import os
import time
from random import randint, seed
from nanovllm.config import Config
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.llm_engine import SpeculationMode


def bench():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    # speculation config comes from env so run_modal.py flags can propagate
    spec_mode_str = os.environ.get("SPEC_MODE", "none").lower()
    spec_length = int(os.environ.get("SPEC_LENGTH", "1"))
    use_spec = spec_mode_str == "naive"
    print(f"Spec: mode={spec_mode_str} length={spec_length if use_spec else '-'}")

    # size memory pool to add up to 90% of GPU memory
    main_model_path = os.path.expanduser(
        os.environ.get("MAIN_MODEL_PATH")
        or os.environ.get("MODEL_PATH", "~/huggingface/Qwen3-0.8B/")
    )
    main_model_config = Config(
        model=main_model_path,
        max_model_len=4096,
        enforce_eager=False,
        gpu_memory_utilization=0.8,
    )
    print("Main Model Path: ", main_model_path)

    # only construct the speculator config when naive spec is requested —
    # Config.__post_init__ hits the filesystem / HF cache, so skipping it
    # lets non-spec runs work without a speculator model present
    spec_kwargs = {}
    if use_spec:
        small_model_path = os.path.expanduser(
            os.environ.get("SPEC_MODEL_PATH", "~/huggingface/Qwen3-0.6B/")
        )
        small_model_config = Config(
            model=small_model_path,
            max_model_len=4096,
            enforce_eager=False,
            gpu_memory_utilization=0.5,
        )
        print("Small Model Path: ", small_model_path)
        spec_kwargs = dict(
            speculation_mode=SpeculationMode.NAIVE_SPECULATION,
            speculator_config=[small_model_config],
            speculation_length=spec_length,
        )

    print("Initializing LLM...")
    llm = LLM(
        model_config=main_model_config,
        **spec_kwargs,
    )

    print("Generating prompts...")
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)
        )
        for _ in range(num_seqs)
    ]

    # Warmup pass so kernel compilation/setup does not pollute benchmark timing.
    print("Warmup")
    warmup_n = min(32, num_seqs)
    llm.generate(
        prompt_token_ids[:warmup_n], sampling_params[:warmup_n], use_tqdm=False
    )
    torch.cuda.synchronize()
    print("Warmup done")

    print("Staring benchmark")
    # snapshot spec counters so warmup's drafts don't leak into the benchmark
    drafts_before = llm.spec_drafts_total
    accepted_before = llm.spec_accepted_total
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    torch.cuda.synchronize()
    t = time.time() - t
    throughput = total_tokens / t
    print(
        f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s"
    )
    drafts = llm.spec_drafts_total - drafts_before
    accepted = llm.spec_accepted_total - accepted_before
    if drafts:
        rate = accepted / drafts
        print(
            f"Spec: drafted={drafts}tok, accepted={accepted}tok, "
            f"acceptance={rate:.2%}"
        )
    print("Benchmark done")


if __name__ == "__main__":
    bench()
