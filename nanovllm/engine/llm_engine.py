import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.engine import block_manager
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.speculation import SpeculationMode


class LLMEngine:
    def __init__(
        self,
        model_config: Config,
        speculation_mode: SpeculationMode = SpeculationMode.NONE,
        speculator_config: list[Config] | None = None,
        speculation_length: int | None = None,
        **kwargs,
    ):
        self.speculation_mode = speculation_mode
        self.model_config = model_config
        self.speculator_config = speculator_config
        self.speculation_length = speculation_length
        # tensor parallelism bookkeeping
        # disable TP with specdecode for now
        if (
            speculation_mode is not SpeculationMode.NONE
            and model_config.tensor_parallel_size > 1
        ):
            raise NotImplementedError("Speculation not supported with TP")

        if speculation_mode is not SpeculationMode.NONE and speculator_config is None:
            raise ValueError(
                "Speculator config and model is required for naive speculation"
            )
        if speculation_mode is not SpeculationMode.NONE and (
            speculation_length is None or speculation_length < 1
        ):
            raise ValueError(
                "Speculation length must be a positive integer for speculation"
            )

        self.block_managers: list[BlockManager] = []
        self.model_runners: list[ModelRunner] = []
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, model_config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(
                target=ModelRunner,
                kwargs={
                    "config": model_config,
                    "rank": i,
                    "event": event,
                    "block_managers": self.block_managers,
                    "model_runner_idx": 0,
                },
            )
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.model, use_fast=True
        )
        model_config.eos = self.tokenizer.eos_token_id

        # setup model runner(s), one for each model instance that we need to run
        self.model_runners.append(
            ModelRunner(
                config=model_config,
                rank=0,
                event=self.events,
                block_managers=self.block_managers,
                model_runner_idx=0,
            )
        )

        self.block_managers.append(
            BlockManager(
                num_blocks=model_config.num_kvcache_blocks,
                block_size=model_config.kvcache_block_size,
                block_table_idx=0,
            )
        )

        if speculation_mode is SpeculationMode.NAIVE_SPECULATION:
            if len(speculator_config) > 1:
                raise NotImplementedError(
                    "Naive Speculation with multiple models not supported"
                )
            speculator_config[0].eos = self.tokenizer.eos_token_id
            self.model_runners.append(
                ModelRunner(
                    config=speculator_config[0],
                    rank=0,
                    event=self.events,
                    block_managers=self.block_managers,
                    model_runner_idx=1,
                )
            )
            # idx should be length of current array
            block_table_idx = len(self.block_managers)
            self.block_managers.append(
                BlockManager(
                    num_blocks=speculator_config[0].num_kvcache_blocks,
                    block_size=speculator_config[0].kvcache_block_size,
                    block_table_idx=block_table_idx,
                )
            )

        # setup scheduler with access to all block managers
        self.scheduler = Scheduler(
            config=model_config,
            block_managers=self.block_managers,
            speculation_mode=self.speculation_mode,
            speculator_config=self.speculator_config,
            speculation_length=self.speculation_length,
        )

        # running acceptance stats for naive speculation
        self.spec_drafts_total = 0
        self.spec_accepted_total = 0

        # register cleanup hook for tp processes
        atexit.register(self.exit)

    def exit(self):
        self.model_runners[0].call("exit")
        del self.model_runners
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(
            token_ids=prompt,
            sampling_params=sampling_params,
            num_block_tables=len(self.block_managers),
        )
        self.scheduler.add(seq)

    def _naive_specdec_step(self):
        # todo; split the step method into dispatch pattern
        pass

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        # TODO: gate behavior base don speculation mode
        step_drafts = step_accepted = -1
        if self.speculation_mode is SpeculationMode.NAIVE_SPECULATION:
            # get the two model runners for regular specdec
            verifier_model_idx = 0
            drafter_model_idx = 1
            verifier = self.model_runners[verifier_model_idx]
            drafter = self.model_runners[drafter_model_idx]
            if is_prefill:
                # fill the kv of both models, but ignore the draft token
                drafter.call("run", seqs, is_prefill)
                token_ids, _ = verifier.call("run", seqs, is_prefill)
                token_ids = [[tok] for tok in token_ids]
            else:
                # generate draft tokens
                for _ in range(self.speculation_length):
                    draft_ids, draft_logits = drafter.call("run", seqs, is_prefill)
                    # add draft tokens to the sequence's draft token ids

                    # TODO: update this so that draft_ids is a list of lists and use extend
                    # so we don't do the appending in the draft loop
                    for seq, draft_id, draft_logit in zip(
                        seqs, draft_ids, draft_logits
                    ):
                        seq.draft_token_ids[drafter_model_idx].append(draft_id)
                        seq.draft_token_logits[drafter_model_idx].append(draft_logit)

                # generate logits for the draft tokens
                # ignore the token that gets generated
                verif_token_ids, verif_logits = verifier.call(
                    "verify", seqs, drafter_model_idx
                )

                # accept/reject per sequence
                token_ids = []
                step_drafts = 0
                step_accepted = 0
                for idx, seq in enumerate(seqs):
                    draft_tokens = seq.draft_token_ids[drafter_model_idx]
                    small_logits = seq.draft_token_logits[drafter_model_idx]
                    big_token_ids = verif_token_ids[idx]
                    big_logits = verif_logits[idx][:-1]
                    seq_accept = []
                    step_drafts += len(draft_tokens)
                    seq_accepted_drafts = 0
                    for tok, small, bin in zip(draft_tokens, small_logits, big_logits):
                        small_prob_dist = small.softmax(dim=-1)
                        big_prob_dist = bin.softmax(dim=-1)
                        p_small = small_prob_dist[tok]
                        p_big = big_prob_dist[tok]
                        accept = p_big >= p_small
                        if not accept:
                            accept = p_big.new_empty(()).uniform_() < (
                                p_big / (p_small + 1e-12)
                            )
                        if accept:
                            seq_accept.append(tok)
                            seq_accepted_drafts += 1
                            continue
                        residual = (big_prob_dist - small_prob_dist).clamp_min(0)
                        residual = residual / (residual.sum() + 1e-12)
                        bonus_token = residual.multinomial(1).item()
                        seq_accept.append(bonus_token)
                        break
                    step_accepted += seq_accepted_drafts
                    if not seq_accept:
                        assert big_token_ids
                        seq_accept.append(big_token_ids[0])

                    token_ids.append(seq_accept)

                # empty draft token list
                for seq in seqs:
                    seq.draft_token_ids[1] = []
                    seq.draft_token_logits[1] = []
        else:
            token_ids, _ = self.model_runners[0].call("run", seqs, is_prefill)
            token_ids = [[tok] for tok in token_ids]

        self.scheduler.postprocess(seqs, token_ids)
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        # step_drafts/step_accepted are -1 for prefill and non-spec steps so
        # callers can distinguish "no spec this step" from a genuine 0-draft
        # batch. Caller aggregates; see generate() / bench for reporting.
        # TODO: richer metrics — per-step distribution of accepted-run
        # lengths, time spent in drafter vs verifier, etc.
        return outputs, num_tokens, step_drafts, step_accepted

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        idx = 0
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens, step_drafts, step_accepted = self.step()
            # accumulate spec metrics regardless of use_tqdm — caller dumps
            # aggregates (see bench.py). -1 sentinel = prefill or non-spec.
            if step_drafts > 0:
                self.spec_drafts_total += step_drafts
                self.spec_accepted_total += step_accepted
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                    decode_throughput = 0
                else:
                    prefill_throughput = 0
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
                idx += 1
                print(
                    f"Step {idx}: Prefill {prefill_throughput:.2f}tok/s, Decode {decode_throughput:.2f}tok/s\n"
                )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        if use_tqdm:
            pbar.close()
        return outputs
