import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def step_speculative(self):
        seqs, is_prefill = self.scheduler.schedule()

        if is_prefill:
            token_ids = self.model_runner.call("run", seqs, True)
            self.scheduler.postprocess(seqs, token_ids)
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
            num_tokens = sum(len(seq) for seq in seqs)
            return outputs, num_tokens, {}

        # Allocate KV slots for K+1 draft tokens per sequence
        K = self.config.medusa_num_heads
        self.scheduler.allocate_spec_slots(seqs, K + 1)

        # Run speculative decode
        accepted_tokens_list, metrics = self.model_runner.call("run_speculative", seqs)

        self.scheduler.postprocess_speculative(seqs, accepted_tokens_list)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # Report actual accepted tokens
        num_tokens = -sum(len(toks) for toks in accepted_tokens_list)
        return outputs, num_tokens, metrics or {}

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        return_metrics: bool = False,
    ) -> list[dict]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        use_spec = bool(self.config.speculative_method)
        outputs = {}
        prefill_throughput = decode_throughput = 0.0

        # Cumulative speculative metrics across all steps
        cumulative_metrics: dict = {}

        while not self.is_finished():
            t = perf_counter()

            if use_spec:
                output, num_tokens, step_metrics = self.step_speculative()
                _accumulate_metrics(cumulative_metrics, step_metrics)
            else:
                output, num_tokens = self.step()
                step_metrics = {}

            elapsed = 

            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                postfix = {
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                }
                if use_spec and cumulative_metrics.get("total_steps", 0) > 0:
                    postfix["tau"] = f"{cumulative_metrics['total_accepted'] / cumulative_metrics['total_steps']:.2f}"
                pbar.set_postfix(postfix)

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        result = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        if return_metrics and use_spec:
            for item in result:
                item["metrics"] = cumulative_metrics

        if use_tqdm:
            pbar.close()
        return result

def _accumulate_metrics(cumulative: dict, step: dict) -> None:
    for key, val in step.items():
        if key not in cumulative:
            cumulative[key] = val
        elif isinstance(val, list):
            if not cumulative[key]:
                cumulative[key] = list(val)
            else:
                cumulative[key] = [a + b for a, b in zip(cumulative[key], val)]
        elif isinstance(val, (int, float)):
            cumulative[key] = cumulative.get(key, 0) + val
