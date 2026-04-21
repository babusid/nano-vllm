import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.engine import block_manager
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.speculation import SpeculationMode
from nanovllm.engine.medusa_utils import (
    generate_medusa_buffers,
    generate_candidates,
    evaluate_posterior,
    mc_sim_7b_63,
)


class LLMEngine:
    def __init__(
        self,
        model_config: Config,
        speculation_mode: SpeculationMode = SpeculationMode.NONE,
        speculator_config: list[Config] | None = None,
        speculation_length: int | None = None,
        # MEDUSA-specific params — ignored in none/naive modes
        medusa_model_path: str | None = None,
        medusa_choices: list | None = None,
        medusa_num_heads: int = 4,
        medusa_num_layers: int = 1,
        **kwargs,
    ):
        self.speculation_mode = speculation_mode
        self.model_config = model_config
        self.speculator_config = speculator_config
        self.speculation_length = speculation_length

        # tensor parallelism bookkeeping — TP not supported with any spec-dec mode
        if (
            speculation_mode is not SpeculationMode.NONE
            and model_config.tensor_parallel_size > 1
        ):
            raise NotImplementedError("Speculation not supported with TP")

        # Mode-specific param validation
        if speculation_mode is SpeculationMode.NAIVE_SPECULATION:
            if speculator_config is None:
                raise ValueError(
                    "speculator_config is required for naive speculation"
                )
            if speculation_length is None or speculation_length < 1:
                raise ValueError(
                    "speculation_length must be a positive integer for naive speculation"
                )
        if speculation_mode is SpeculationMode.MEDUSA:
            if medusa_model_path is None:
                raise ValueError(
                    "medusa_model_path is required for MEDUSA speculation"
                )

        # Pre-compute static MEDUSA tree buffers once at engine start.
        # These are passed to ModelRunner (for attention / head computation)
        # and Scheduler (for KV slot reservation).
        medusa_buffers: dict | None = None
        if speculation_mode is SpeculationMode.MEDUSA:
            choices = medusa_choices if medusa_choices is not None else mc_sim_7b_63
            # Clip the tree to paths compatible with the available number of heads.
            # At depth d (path length d), generate_candidates maps those nodes to
            # flat-candidate indices  cur[-1] + TOPK * (d-1) + 1.  The flat vector
            # has size 1 + num_heads * TOPK, so valid depths are 1..num_heads only.
            # Paths longer than num_heads would produce out-of-range indices and
            # cause CUDA index-out-of-bounds assertions at runtime.
            choices = [c for c in choices if len(c) <= medusa_num_heads]
            if not choices:
                raise ValueError(
                    f"No valid medusa_choices for medusa_num_heads={medusa_num_heads}. "
                    "All paths in the topology exceed the number of heads."
                )
            medusa_buffers = generate_medusa_buffers(choices, device="cuda")
        self.medusa_buffers = medusa_buffers

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
        # Tell the verifier how many query tokens per seq to expect during verify
        # so it can capture a matching family of CUDA graphs.
        verify_seqlen_q = (
            speculation_length + 1
            if speculation_mode is SpeculationMode.NAIVE_SPECULATION
            else None
        )
        self.model_runners.append(
            ModelRunner(
                config=model_config,
                rank=0,
                event=self.events,
                block_managers=self.block_managers,
                model_runner_idx=0,
                verify_seqlen_q=verify_seqlen_q,
                # MEDUSA params (all None/0 in non-MEDUSA modes)
                medusa_model_path=medusa_model_path,
                medusa_num_heads=medusa_num_heads,
                medusa_num_layers=medusa_num_layers,
                medusa_buffers=medusa_buffers,
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
            medusa_len=medusa_buffers["medusa_len"] if medusa_buffers else None,
        )

        # running acceptance stats for spec-dec modes
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

    # ------------------------------------------------------------------
    # MEDUSA step logic
    # ------------------------------------------------------------------

    def _medusa_step(
        self, seqs: list[Sequence], is_prefill: bool
    ) -> tuple[list[list[int]], int, int]:
        """Execute one MEDUSA engine step.

        Returns
        -------
        token_ids      : list[list[int]]  — new tokens per sequence (for postprocess)
        step_drafts    : int              — speculative candidates tried (-1 for prefill)
        step_accepted  : int              — speculative tokens accepted (-1 for prefill)
        """
        runner = self.model_runners[0]
        buffers = self.medusa_buffers
        retrieve_indices = buffers["retrieve_indices"]  # [num_paths, depth+1]
        tree_indices = buffers["tree_indices"]          # [medusa_len]

        # ---- Prefill ----
        if is_prefill:
            token_ids, _ = runner.call("run", seqs, True)
            # Clear MEDUSA logits so the first decode step triggers the seed pass.
            for seq in seqs:
                seq.medusa_lm_logits = None
                seq.medusa_head_logits = None
            return [[tok] for tok in token_ids], -1, -1

        # ---- Decode (batch size is always 1 in MEDUSA mode) ----
        assert len(seqs) == 1, "MEDUSA decode requires batch size 1"
        seq = seqs[0]

        # Seed pass: runs once after each prefill.
        # Writes the first generated token's KV to the paged cache and computes
        # LM + MEDUSA-head logits at that position so candidate generation works.
        if seq.medusa_lm_logits is None:
            seed_start = len(seq) - 1   # logical position of the last committed token
            lm_s, med_s = runner.call(
                "run_medusa_kv_and_seed", seq, [seq.last_token], seed_start
            )
            seq.medusa_lm_logits = lm_s.reshape(1, 1, -1)
            seq.medusa_head_logits = med_s.reshape(-1, 1, 1, lm_s.shape[-1])

        # Build the candidate tree from stored per-position logits.
        cart_candidates, tree_candidates = generate_candidates(
            seq.medusa_head_logits,
            seq.medusa_lm_logits,
            tree_indices,
            retrieve_indices,
            temperature=seq.temperature,
        )
        # cart_candidates : [num_paths, depth+1]   cartesian paths
        # tree_candidates : [1, medusa_len]         flat tree layout

        # Run all medusa_len tree candidates through the model in one pass.
        lm_logits, _ = runner.call(
            "run_medusa_tree", seqs, {seq.seq_id: tree_candidates}
        )
        # lm_logits : [medusa_len, vocab]

        # Posterior evaluation: find the best accepted path.
        path_logits = lm_logits[retrieve_indices]   # [num_paths, depth+1, vocab]
        best_candidate, accept_length = evaluate_posterior(
            path_logits,
            cart_candidates,
            temperature=seq.temperature,
        )

        # Accepted tokens: root (position 0 in path) + accept_length speculative.
        accepted = cart_candidates[best_candidate, : accept_length + 1].tolist()

        # Bonus token: greedy / sampled from the model's prediction at the last
        # accepted tree position.  Becomes the root of the next step's tree.
        accept_node = int(retrieve_indices[best_candidate, accept_length])
        if seq.temperature == 0:
            bonus = int(lm_logits[accept_node].argmax())
        else:
            bonus = int(
                (lm_logits[accept_node] / seq.temperature)
                .softmax(dim=-1)
                .multinomial(1)
            )

        all_new_tokens = accepted + [bonus]
        old_committed = len(seq)    # position where the new tokens start

        # Corrective KV pass: re-run the model on the accepted path so that
        # the paged cache has correct sequential K/V for every new position.
        # Also returns seed logits for the next tree-decode step.
        lm_seed, med_seed = runner.call(
            "run_medusa_kv_and_seed", seq, all_new_tokens, old_committed
        )
        seq.medusa_lm_logits = lm_seed.reshape(1, 1, -1)
        seq.medusa_head_logits = med_seed.reshape(-1, 1, 1, lm_seed.shape[-1])

        # step_drafts = speculative candidates (all tree positions minus root)
        # step_accepted = verified speculative tokens
        step_drafts = buffers["medusa_len"] - 1
        step_accepted = accept_length
        return [all_new_tokens], step_drafts, step_accepted

    def step(self):
        with torch.profiler.record_function("llm.step.schedule"):
            seqs, is_prefill = self.scheduler.schedule()
        step_drafts = step_accepted = -1

        if self.speculation_mode is SpeculationMode.MEDUSA:
            with torch.profiler.record_function("medusa.step"):
                token_ids, step_drafts, step_accepted = self._medusa_step(
                    seqs, is_prefill
                )

        elif self.speculation_mode is SpeculationMode.NAIVE_SPECULATION:
            # get the two model runners for regular specdec
            verifier_model_idx = 0
            drafter_model_idx = 1
            verifier = self.model_runners[verifier_model_idx]
            drafter = self.model_runners[drafter_model_idx]
            if is_prefill:
                # fill the kv of both models, but ignore the draft token
                with torch.profiler.record_function("spec.prefill.drafter_run"):
                    drafter.call("run", seqs, is_prefill)
                with torch.profiler.record_function("spec.prefill.verifier_run"):
                    token_ids, _ = verifier.call("run", seqs, is_prefill)
                token_ids = [[tok] for tok in token_ids]
            else:
                # generate draft tokens
                for _ in range(self.speculation_length):
                    with torch.profiler.record_function("spec.decode.drafter_run"):
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
                with torch.profiler.record_function("spec.decode.verifier_verify"):
                    verif_token_ids, verif_logits = verifier.call(
                        "verify", seqs, drafter_model_idx
                    )

                # accept/reject per sequence
                token_ids = []
                step_drafts = 0
                step_accepted = 0
                with torch.profiler.record_function("spec.decode.accept_reject"):
                    for idx, seq in enumerate(seqs):
                        draft_tokens = seq.draft_token_ids[drafter_model_idx]
                        small_logits = seq.draft_token_logits[drafter_model_idx]
                        big_token_ids = verif_token_ids[idx]
                        draft_big_logits = verif_logits[idx][:-1]
                        seq_accept = []
                        step_drafts += len(draft_tokens)
                        seq_accepted_drafts = 0
                        for tok, small, big in zip(
                            draft_tokens, small_logits, draft_big_logits
                        ):
                            # upcast to fp32 — fp16 logits (esp. Vicuna-33B) can
                            # overflow and poison softmax with inf/nan, which
                            # propagates into residual and trips multinomial's
                            # probability-validity assert.
                            small_prob_dist = small.float().softmax(dim=-1)
                            big_prob_dist = big.float().softmax(dim=-1)
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
                            rsum = residual.sum()
                            if rsum <= 0 or not torch.isfinite(rsum):
                                # big ≤ small everywhere (or non-finite): fall
                                # back to sampling from the target distribution
                                bonus_token = big_prob_dist.multinomial(1).item()
                            else:
                                bonus_token = (residual / rsum).multinomial(1).item()
                            seq_accept.append(bonus_token)
                            break
                        if seq_accepted_drafts == len(draft_tokens) and draft_tokens:
                            assert len(big_token_ids) == len(draft_tokens) + 1
                            seq_accept.append(big_token_ids[-1])
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
            with torch.profiler.record_function("base.run"):
                token_ids, _ = self.model_runners[0].call("run", seqs, is_prefill)
            token_ids = [[tok] for tok in token_ids]

        with torch.profiler.record_function("llm.step.postprocess"):
            self.scheduler.postprocess(seqs, token_ids)
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        # For MEDUSA decode, report the actual number of tokens committed this
        # step (root + accepted speculative + bonus) rather than just -1 per seq,
        # so that the throughput display in generate() reflects real token rate.
        if self.speculation_mode is SpeculationMode.MEDUSA and not is_prefill:
            num_tokens = -sum(len(tids) for tids in token_ids)
        else:
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
