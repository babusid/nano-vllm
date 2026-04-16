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

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        # TODO: gate behavior base don speculation mode
        if self.speculation_mode is SpeculationMode.NAIVE_SPECULATION:
            # get the two model runners for regular specdec
            verifier = self.model_runners[0]
            drafter = self.model_runners[1]
            if is_prefill:
                # fill the kv of both models, but ignore the draft token
                drafter.call("run", seqs, is_prefill)
                token_ids = verifier.call("run", seqs, is_prefill)
            else:
                # generate draft tokens
                for _ in range(self.speculation_length):
                    draft_ids = drafter.call("run", seqs, is_prefill)
                    # add draft tokens to the sequence's draft token ids

                    # TODO: update this so that draft_ids is a list of lists and use extend
                    # so we don't do the appending in the draft loop
                    for seq, draft_id in zip(seqs, draft_ids):
                        seq.draft_token_ids[1].append(draft_id)

                token_ids = verifier.call("run", seqs, is_prefill)

                # TODO: need actual verif / memory reclaim logic here
                # draft tokens should be passed to the verifier, which will
                # output the correct tokens, we have to empty the draft token list
                # and reclaim the memory for the ones that got rejected
                # empty draft token list
                for seq in seqs:
                    seq.draft_token_ids[1] = []

        else:
            token_ids = self.model_runners[0].call("run", seqs, is_prefill)

        self.scheduler.postprocess(seqs, token_ids)
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

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
            output, num_tokens = self.step()
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
