"""
bench_speculative.py — Ablation benchmark for speculative decoding.

Compares standard autoregressive decoding against MEDUSA (and optionally EAGLE
once implemented) on an identical prompt/max-token workload.

Usage
-----
# Baseline AR only:
MODEL_PATH=~/models/Qwen3-0.6B python bench_speculative.py

# MEDUSA (heads loaded from model dir or medusa_model_path):
MODEL_PATH=~/models/Qwen3-0.6B python bench_speculative.py --method medusa --num-heads 4

# Full comparison (runs AR then MEDUSA, prints side-by-side table):
MODEL_PATH=~/models/Qwen3-0.6B python bench_speculative.py --method medusa --compare

Metrics reported
----------------
  Throughput      tokens / wall-second  (higher is better)
  TTFT            time-to-first-token in seconds  (lower is better)
  τ  (tau)        mean accepted tokens per speculative step  (MEDUSA only)
  Head acceptance fraction of drafts accepted per head index  (MEDUSA only)
  Speedup         throughput(spec) / throughput(AR)
  Memory delta    GPU memory overhead of draft heads in MB  (MEDUSA only)
"""

import argparse
import os
import time
from random import randint, seed

import torch

from nanovllm import LLM, SamplingParams


def run_benchmark(
    llm: LLM,
    prompt_token_ids: list[list[int]],
    sampling_params: list[SamplingParams],
    label: str,
    return_metrics: bool = False,
) -> dict:
    # Warmup
    warmup_n = min(16, len(prompt_token_ids))
    llm.generate(prompt_token_ids[:warmup_n], sampling_params[:warmup_n], use_tqdm=False)
    torch.cuda.synchronize()

    mem_before = torch.cuda.memory_allocated() / 1024 ** 2  # MiB

    t_start = time.perf_counter()
    results = llm.generate(
        prompt_token_ids,
        sampling_params,
        use_tqdm=True,
        return_metrics=return_metrics,
    )
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    mem_after = torch.cuda.memory_allocated() / 1024 ** 2

    total_tokens = sum(len(r["token_ids"]) for r in results)
    elapsed = t_end - t_start
    throughput = total_tokens / elapsed

    stats = {
        "label": label,
        "total_tokens": total_tokens,
        "elapsed_s": elapsed,
        "throughput_tok_s": throughput,
        "mem_mb": mem_after,
        "mem_delta_mb": mem_after - mem_before,
    }

    # Speculative metrics — derive rates from raw accumulated counts so
    # they remain correct regardless of how many decode steps were taken.
    if results and "metrics" in results[0]:
        m = results[0]["metrics"]
        total_steps   = m.get("total_steps", 0)
        total_accepted = m.get("total_accepted", 0)
        per_head_hits  = m.get("per_head_hits", [])
        per_head_tries = m.get("per_head_tries", [])
        stats["total_steps"]   = total_steps
        stats["total_accepted"] = total_accepted
        stats["mean_accepted_per_step"] = (
            total_accepted / total_steps if total_steps else 0.0
        )
        stats["per_head_acceptance_rate"] = [
            h / t if t > 0 else 0.0
            for h, t in zip(per_head_hits, per_head_tries)
        ]

    return stats


def measure_ttft(llm: LLM, prompt_token_ids: list[list[int]]) -> float:
    sp = [SamplingParams(temperature=0.6, max_tokens=1)]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.generate(prompt_token_ids[:1], sp, use_tqdm=False)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def print_stats(stats: dict) -> None:
    label = stats["label"]
    print(f"\n{'─' * 55}")
    print(f"  {label}")
    print(f"{'─' * 55}")
    print(f"  Tokens generated   : {stats['total_tokens']:,}")
    print(f"  Wall time          : {stats['elapsed_s']:.2f} s")
    print(f"  Throughput         : {stats['throughput_tok_s']:.1f} tok/s")
    if "ttft_s" in stats:
        print(f"  TTFT               : {stats['ttft_s'] * 1000:.1f} ms")
    if "mem_delta_mb" in stats:
        print(f"  GPU memory delta   : {stats['mem_delta_mb']:+.1f} MiB")
    if "mean_accepted_per_step" in stats:
        tau = stats["mean_accepted_per_step"]
        print(f"  τ (mean accepted)  : {tau:.3f} tokens/step")
    if "per_head_acceptance_rate" in stats:
        rates = stats["per_head_acceptance_rate"]
        for i, r in enumerate(rates):
            print(f"  Head {i} accept rate : {r * 100:.1f}%")
    if "speedup" in stats:
        print(f"  Speedup vs AR      : {stats['speedup']:.2f}×")
    print(f"{'─' * 55}")


def parse_args():
    p = argparse.ArgumentParser(description="Speculative decoding benchmark")
    p.add_argument("--method", default="", choices=["", "medusa"],
                   help="Speculative decoding strategy (default: AR baseline only)")
    p.add_argument("--num-heads", type=int, default=4,
                   help="Number of MEDUSA draft heads / lookahead depth (default: 4)")
    p.add_argument("--num-layers", type=int, default=1,
                   help="ResBlock layers per MEDUSA head (default: 1)")
    p.add_argument("--medusa-path", default="",
                   help="Path to medusa_heads.safetensors (default: search model dir)")
    p.add_argument("--compare", action="store_true",
                   help="Run both AR and speculative and compare side by side")
    p.add_argument("--num-seqs", type=int, default=128,
                   help="Number of sequences to generate (default: 128)")
    p.add_argument("--max-input-len", type=int, default=512)
    p.add_argument("--max-output-len", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    seed(args.seed)

    model_path = os.path.expanduser(
        os.environ.get("MODEL_PATH", "~/huggingface/Qwen3-0.6B/")
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=args.temperature,
            ignore_eos=True,
            max_tokens=randint(50, args.max_output_len),
        )
        for _ in range(args.num_seqs)
    ]

    results_ar = None
    results_spec = None

    # ---- AR baseline -------------------------------------------------------
    if args.compare or not args.method:
        print("\n[Baseline] Loading AR model …")
        llm_ar = LLM(model_path, enforce_eager=False, max_model_len=4096)
        ttft_ar = measure_ttft(llm_ar, prompt_token_ids)
        results_ar = run_benchmark(llm_ar, prompt_token_ids, sampling_params,
                                   label="Baseline (autoregressive)")
        results_ar["ttft_s"] = ttft_ar
        print_stats(results_ar)
        llm_ar.exit()   # tears down dist process group so the next LLM can init
        del llm_ar
        torch.cuda.empty_cache()

    # ---- Speculative decode ------------------------------------------------
    if args.method:
        print(f"\n[{args.method.upper()}] Loading speculative model …")
        llm_spec = LLM(
            model_path,
            enforce_eager=False,
            max_model_len=4096,
            speculative_method=args.method,
            num_speculative_tokens=args.num_heads,
            medusa_num_layers=args.num_layers,
            medusa_model_path=args.medusa_path,
        )
        ttft_spec = measure_ttft(llm_spec, prompt_token_ids)
        results_spec = run_benchmark(
            llm_spec, prompt_token_ids, sampling_params,
            label=f"Speculative ({args.method}, K={args.num_heads})",
            return_metrics=True,
        )
        results_spec["ttft_s"] = ttft_spec

        if results_ar:
            results_spec["speedup"] = (
                results_spec["throughput_tok_s"] / results_ar["throughput_tok_s"]
            )

        print_stats(results_spec)
        del llm_spec
        torch.cuda.empty_cache()

    if args.compare and results_ar and results_spec:
        print("\n" + "═" * 55)
        print("  COMPARISON SUMMARY")
        print("═" * 55)
        print(f"  AR throughput      : {results_ar['throughput_tok_s']:.1f} tok/s")
        print(f"  Spec throughput    : {results_spec['throughput_tok_s']:.1f} tok/s")
        print(f"  Speedup            : {results_spec.get('speedup', 0):.2f}×")
        if "mean_accepted_per_step" in results_spec:
            print(f"  τ (mean accepted)  : {results_spec['mean_accepted_per_step']:.3f} tokens/step")
        print("═" * 55)


if __name__ == "__main__":
    main()
