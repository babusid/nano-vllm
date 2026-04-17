import argparse
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
        gpt_value = next(
            (t["value"] for t in turns if t.get("from") == "gpt"), None
        )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nano-vllm ShareGPT benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ShareGPT_V3_unfiltered_cleaned_split.json",
        help="Path to the ShareGPT JSON file.",
    )
    parser.add_argument(
        "--num-seqs",
        type=int,
        default=256,
        help="Number of prompts to benchmark.",
    )
    parser.add_argument(
        "--max-input-len",
        type=int,
        default=1024,
        help="Maximum input token length; longer prompts are discarded.",
    )
    parser.add_argument(
        "--max-output-len",
        type=int,
        default=1024,
        help="Cap on output tokens per request (derived from reference reply length).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for prompt sampling.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # size memory pool to add up to 90% of GPU memory
    main_model_path = os.path.expanduser(
        os.environ.get("MAIN_MODEL_PATH")
        or os.environ.get("MODEL_PATH", "~/huggingface/Qwen3-8B/")
    )
    print("Main Model Path: ", main_model_path)
    main_model_config = Config(
        model=main_model_path,
        max_model_len=4096,
        enforce_eager=False,
        gpu_memory_utilization=0.8,
    )
    small_model_path = os.path.expanduser(
        os.environ.get("SPEC_MODEL_PATH", "~/huggingface/Qwen3-0.6B/")
    )
    print("Small Model Path: ", small_model_path)
    small_model_config = Config(
        model=small_model_path,
        max_model_len=4096,
        enforce_eager=False,
        gpu_memory_utilization=0.5,
    )

    llm = LLM(
        model=main_model_path,
        model_config=main_model_config,
        speculation_mode=SpeculationMode.NAIVE_SPECULATION,
        speculation_model=[small_model_path],
        speculator_config=[small_model_config],
    )

    print(f"Loading ShareGPT prompts from {args.dataset} ...")
    prompts, sampling_params = load_sharegpt_prompts(
        path=args.dataset,
        tokenizer=llm.tokenizer,
        num_seqs=args.num_seqs,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        seed=args.seed,
        temperature=args.temperature,
    )
    print(f"Loaded {len(prompts)} prompts.")

    # Warmup pass so kernel compilation/setup does not pollute benchmark timing.
    warmup_n = min(32, args.num_seqs)
    llm.generate(prompts[:warmup_n], sampling_params[:warmup_n], use_tqdm=False)
    torch.cuda.synchronize()

    t = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    torch.cuda.synchronize()
    t = time.time() - t

    total_tokens = sum(len(out["token_ids"]) for out in outputs)
    throughput = total_tokens / t
    print(
        f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s"
    )


if __name__ == "__main__":
    main()
