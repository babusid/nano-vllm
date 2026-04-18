import json
import os
import random
import time
from nanovllm.config import Config
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.llm_engine import SpeculationMode


def load_sharegpt_prompts(
    path: str,
    tokenizer,
    num_seqs: int,
    max_input_len: int,
    max_output_len: int,
    seed: int = 0,
    temperature: float = 0.6,
) -> tuple[list[str], list[SamplingParams]]:

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    candidates: list[tuple[str, int]] = []
    for conversation in dataset:
        turns = conversation.get("conversations", [])

        human_value = next(
            (t["value"] for t in turns if t.get("from") == "human"), None
        )
        gpt_value = next((t["value"] for t in turns if t.get("from") == "gpt"), None)
        if human_value is None or gpt_value is None:
            continue

        input_ids = tokenizer.encode(human_value)
        if len(input_ids) > max_input_len:
            continue

        output_ids = tokenizer.encode(gpt_value)
        output_len = min(len(output_ids), max_output_len)
        if output_len == 0:
            continue

        candidates.append((human_value, output_len))

    if len(candidates) < num_seqs:
        raise ValueError(
            f"Only {len(candidates)} valid ShareGPT prompts found after filtering, "
            f"but num_seqs={num_seqs} requested. "
            "Try increasing max_input_len / max_output_len or reducing num_seqs."
        )

    rng = random.Random(seed)
    selected = rng.sample(candidates, num_seqs)

    prompts = [prompt for prompt, _ in selected]
    sampling_params = [
        SamplingParams(temperature=temperature, max_tokens=output_len)
        for _, output_len in selected
    ]
    return prompts, sampling_params


def bench():
    dataset_path = os.environ.get(
        "SHAREGPT_PATH", "ShareGPT_V3_unfiltered_cleaned_split.json"
    )
    num_seqs = int(os.environ.get("BENCH_NUM_SEQS", "256"))
    max_input_len = int(os.environ.get("BENCH_MAX_INPUT_LEN", "1024"))
    max_output_len = int(os.environ.get("BENCH_MAX_OUTPUT_LEN", "1024"))
    sampling_seed = int(os.environ.get("BENCH_SEED", "0"))
    temperature = float(os.environ.get("BENCH_TEMPERATURE", "1e-9"))
    warmup_seqs = int(os.environ.get("BENCH_WARMUP_SEQS", "32"))
    main_max_model_len = int(os.environ.get("BENCH_MAIN_MAX_MODEL_LEN", "4096"))
    main_gpu_memory_utilization = float(
        os.environ.get("BENCH_MAIN_GPU_MEMORY_UTILIZATION", "0.8")
    )
    spec_max_model_len = int(os.environ.get("BENCH_SPEC_MAX_MODEL_LEN", "4096"))
    spec_gpu_memory_utilization = float(
        os.environ.get("BENCH_SPEC_GPU_MEMORY_UTILIZATION", "0.5")
    )

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
        max_model_len=main_max_model_len,
        enforce_eager=os.environ.get("ENFORCE_EAGER", "0") == "1",
        gpu_memory_utilization=main_gpu_memory_utilization,
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
            max_model_len=spec_max_model_len,
            enforce_eager=os.environ.get("ENFORCE_EAGER", "0") == "1",
            gpu_memory_utilization=spec_gpu_memory_utilization,
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

    print(f"Loading ShareGPT prompts from {dataset_path} ...")
    prompts, sampling_params = load_sharegpt_prompts(
        path=dataset_path,
        tokenizer=llm.tokenizer,
        num_seqs=num_seqs,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        seed=sampling_seed,
        temperature=temperature,
    )
    print(f"Loaded {len(prompts)} prompts.")

    # Warmup pass so kernel compilation/setup does not pollute benchmark timing.
    warmup_n = min(warmup_seqs, num_seqs)
    llm.generate(prompts[:warmup_n], sampling_params[:warmup_n], use_tqdm=False)
    torch.cuda.synchronize()
    print("Warmup done")

    print("Staring benchmark")
    # snapshot spec counters so warmup's drafts don't leak into the benchmark
    drafts_before = llm.spec_drafts_total
    accepted_before = llm.spec_accepted_total
    t = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    torch.cuda.synchronize()
    t = time.time() - t

    total_tokens = sum(len(out["token_ids"]) for out in outputs)
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
