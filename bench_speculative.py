"""
bench_speculative.py — Ablation benchmark for speculative decoding.

Compares standard autoregressive decoding against MEDUSA (and optionally EAGLE
once implemented) on an identical prompt/max-token workload.

Usage
-----
# Baseline AR only (synthetic prompts):
MODEL_PATH=~/models/Qwen3-0.6B python bench_speculative.py

# MEDUSA comparison with ShareGPT data:
MODEL_PATH=~/models/vicuna-7b-v1.3 python bench_speculative.py \\
    --method medusa --num-heads 5 --compare \\
    --dataset sharegpt --dataset-path /path/to/ShareGPT_V3.json

# Full comparison (runs AR then MEDUSA, prints side-by-side table):
MODEL_PATH=~/models/Qwen3-0.6B python bench_speculative.py --method medusa --compare

# Single-sequence mode (MEDUSA's intended sweet spot):
MODEL_PATH=~/models/Qwen3-0.6B python bench_speculative.py --method medusa --compare --num-seqs 1

Metrics reported
----------------
  Throughput      tokens / wall-second  (higher is better)
  TTFT            time-to-first-token in seconds  (lower is better)
  τ  (tau)        mean accepted tokens per speculative step  (MEDUSA only)
  Head acceptance fraction of drafts accepted per head index  (MEDUSA only)
  Speedup         throughput(spec) / throughput(AR)
  Memory delta    GPU memory overhead of draft heads in MB  (MEDUSA only)

Notes
-----
  - TTFT is measured after a warmup pass so both AR and MEDUSA see a hot GPU.
  - Prompts are real English sentences replicated to --num-seqs (synthetic mode).
    Using random token IDs causes degenerate model outputs (repeated tokens) which
    inflate τ and acceptance rates to unrealistic values.
  - ShareGPT mode uses real human conversation turns.  Apply --max-input-len to
    filter prompts that would overflow the KV cache.
  - MEDUSA is designed for batch_size=1.  At large batch sizes the verification
    forward pass overhead dominates and speedup is modest.
  - With --dataset sharegpt, ignore_eos is False so sequences finish naturally at
    different lengths, matching real serving conditions.
"""

import argparse
import json
import os
import time
from random import Random
from random import seed

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.config import Config


# ---------------------------------------------------------------------------
# Real prompts — diverse enough to exercise realistic model distributions.
# Repeated / sampled to reach --num-seqs without needing an external dataset.
# ---------------------------------------------------------------------------
_BASE_PROMPTS = [
    "Explain the difference between supervised and unsupervised learning in machine learning.",
    "What are the main causes of the French Revolution and how did they lead to the fall of the monarchy?",
    "Write a short story about a robot who discovers it has emotions for the first time.",
    "Describe the process of photosynthesis and why it is important for life on Earth.",
    "What is the significance of the Turing Test in the field of artificial intelligence?",
    "Compare and contrast the economic systems of capitalism and socialism.",
    "Summarize the key events of the Second World War in chronological order.",
    "How does the human immune system defend against viral infections?",
    "Explain Newton's three laws of motion with real-world examples.",
    "What are the environmental impacts of deforestation in the Amazon rainforest?",
    "Describe the architecture of a transformer model used in modern NLP.",
    "What is quantum entanglement and why did Einstein call it 'spooky action at a distance'?",
    "How do compilers differ from interpreters in the context of programming languages?",
    "What role did the Silk Road play in ancient trade and cultural exchange?",
    "Explain the concept of recursion in computer science with a simple example.",
    "What are the ethical considerations surrounding the use of AI in healthcare?",
    "Describe the water cycle and explain how human activity affects it.",
    "What is the significance of the Higgs boson in particle physics?",
    "How does the blockchain achieve consensus without a central authority?",
    "Explain the concept of natural selection and how it drives evolution.",
]


_VICUNA_TEMPLATE = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
    "\n\nUSER: {text}\nASSISTANT:"
)


def _format_prompt(text: str, tokenizer, chat_template: str) -> str:
    """Apply a chat template to a raw user message.

    chat_template choices:
      "auto"   — use tokenizer.chat_template if available, else vicuna fallback
      "vicuna" — always use USER:/ASSISTANT: format (lmsys/vicuna-* models)
      "none"   — pass raw text through unchanged
    """
    if chat_template == "none":
        return text
    if chat_template == "vicuna":
        return _VICUNA_TEMPLATE.format(text=text)
    # "auto"
    has_template = (
        hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None) is not None
    )
    if has_template:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    # Fallback: Vicuna-style format works reasonably for most instruction models
    return _VICUNA_TEMPLATE.format(text=text)


def _load_sharegpt_prompts(
    path: str,
    tokenizer,
    num_seqs: int,
    max_input_len: int,
    max_output_len: int,
    temperature: float,
    chat_template: str,
    rng: Random,
) -> tuple[list[list[int]], list[SamplingParams]]:
    """Load real prompts from a ShareGPT JSON file.

    Each conversation's first human turn is used as the prompt.  The chat
    template is applied when the tokenizer supports it (required for models
    like Vicuna that expect a specific conversation format).

    Prompts longer than *max_input_len* tokens are discarded so they do not
    overflow the engine's KV cache.  Sequences are allowed to finish at their
    natural EOS so output lengths vary, matching real serving conditions.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Normalise top-level structure — some files wrap the list in a dict.
    if isinstance(data, dict):
        # Try common wrapper keys.
        for key in ("conversations", "data", "items", "records"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            # Last resort: take the first list value found.
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break

    # Extract the first human turn from every conversation.
    # Handles both the standard ShareGPT format {"from": "human", "value": "..."}
    # and the ChatML-style {"role": "user", "content": "..."}.
    raw_texts: list[str] = []
    for conv in data:
        if not isinstance(conv, dict):
            continue
        turns = conv.get("conversations") or conv.get("messages") or []
        human_turn = next(
            (
                t.get("value") or t.get("content", "")
                for t in turns
                if t.get("from") in ("human", "user")
                or t.get("role") in ("human", "user")
            ),
            None,
        )
        if human_turn and human_turn.strip():
            raw_texts.append(human_turn.strip())

    print(f"[ShareGPT] Parsed {len(raw_texts)} human turns from {path}")

    # Apply chat template.
    formatted: list[str] = [
        _format_prompt(text, tokenizer, chat_template) for text in raw_texts
    ]

    # Tokenize and filter by length.
    prompt_token_ids: list[list[int]] = []
    for text in formatted:
        ids = tokenizer.encode(text)
        if len(ids) <= max_input_len:
            prompt_token_ids.append(ids)

    if not prompt_token_ids:
        raise ValueError(
            f"No ShareGPT prompts survived the max_input_len={max_input_len} filter. "
            "Try increasing --max-input-len."
        )

    # Sample with replacement if there are fewer surviving prompts than requested.
    if len(prompt_token_ids) < num_seqs:
        prompt_token_ids = rng.choices(prompt_token_ids, k=num_seqs)
    else:
        prompt_token_ids = rng.sample(prompt_token_ids, num_seqs)

    sampling_params = [
        SamplingParams(
            temperature=temperature,
            ignore_eos=False,       # natural EOS termination for realism
            max_tokens=max_output_len,
        )
        for _ in range(num_seqs)
    ]
    print(
        f"[dataset] ShareGPT: {len(prompt_token_ids)} prompts loaded "
        f"(avg input len {sum(len(p) for p in prompt_token_ids) // len(prompt_token_ids)} tok)"
    )
    return prompt_token_ids, sampling_params


def run_benchmark(
    llm: LLM,
    prompt_token_ids: list[list[int]],
    sampling_params: list[SamplingParams],
    label: str,
    return_metrics: bool = False,
    show_output: bool = False,
    tokenizer=None,
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

    if show_output and results and tokenizer is not None:
        prompt_text = tokenizer.decode(prompt_token_ids[0], skip_special_tokens=True)
        output_text = results[0]["text"]
        print(f"\n{'─'*55}")
        print(f"  [show-output] PROMPT  : {prompt_text[:200]!r}")
        print(f"  [show-output] OUTPUT  : {output_text[:400]!r}")
        print(f"{'─'*55}")

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
    """Measure prefill latency for a single prompt after a warmup pass."""
    sp = [SamplingParams(temperature=0.6, max_tokens=1)]
    # Warmup: one generate call so CUDA kernels are compiled and GPU is hot.
    llm.generate(prompt_token_ids[:1], sp, use_tqdm=False)
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
    p.add_argument("--medusa-path", default=os.environ.get("MEDUSA_HEADS_PATH", ""),
                   help="Path to medusa_lm_head.pt / medusa_heads.safetensors, or a "
                        "directory containing it. Falls back to MEDUSA_HEADS_PATH env var.")
    p.add_argument("--compare", action="store_true",
                   help="Run both AR and speculative and compare side by side")
    p.add_argument("--num-seqs", type=int, default=128,
                   help="Number of sequences to generate (default: 128). "
                        "Use 1 for MEDUSA's intended single-sequence sweet spot.")
    p.add_argument("--max-output-len", type=int, default=256,
                   help="Maximum tokens to generate per sequence (default: 256)")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    # Dataset selection
    p.add_argument("--dataset", default="synthetic",
                   choices=["synthetic", "sharegpt"],
                   help="Prompt dataset: 'synthetic' uses built-in English sentences; "
                        "'sharegpt' loads real conversations from --dataset-path.")
    p.add_argument("--dataset-path",
                   default=os.environ.get("SHAREGPT_PATH", ""),
                   help="Path to ShareGPT JSON file (required when --dataset sharegpt). "
                        "Falls back to SHAREGPT_PATH env var.")
    p.add_argument("--max-input-len", type=int, default=1024,
                   help="Maximum prompt length in tokens; longer ShareGPT entries are "
                        "discarded (default: 1024)")
    p.add_argument("--chat-template", default="auto",
                   choices=["auto", "vicuna", "none"],
                   help="Chat template to wrap prompts with before encoding. "
                        "'auto' uses the tokenizer's built-in template and falls back "
                        "to vicuna-style USER:/ASSISTANT: format. "
                        "'vicuna' always uses USER:/ASSISTANT: (use for lmsys/vicuna-*). "
                        "'none' passes raw text (only useful for models without an "
                        "instruction format). Default: auto")
    p.add_argument("--show-output", action="store_true",
                   help="Print the first prompt and its generated output for "
                        "sanity-checking that the model is generating coherent text "
                        "rather than repeating tokens.")
    return p.parse_args()


def _build_synthetic_prompts(
    tokenizer,
    num_seqs: int,
    max_output_len: int,
    temperature: float,
    chat_template: str,
) -> tuple[list[list[int]], list[SamplingParams]]:
    """Build prompt_token_ids and sampling_params from built-in English sentences."""
    texts: list[str] = []
    while len(texts) < num_seqs:
        texts.extend(_BASE_PROMPTS)
    texts = texts[:num_seqs]

    formatted = [_format_prompt(t, tokenizer, chat_template) for t in texts]
    prompt_token_ids = [tokenizer.encode(t) for t in formatted]
    sampling_params = [
        SamplingParams(
            temperature=temperature,
            ignore_eos=True,
            max_tokens=max_output_len,
        )
        for _ in range(num_seqs)
    ]
    print(
        f"[dataset] Synthetic: {num_seqs} prompts "
        f"(avg input len {sum(len(p) for p in prompt_token_ids) // max(len(prompt_token_ids),1)} tok, "
        f"template={chat_template!r})"
    )
    return prompt_token_ids, sampling_params


def build_prompts(args, tokenizer, rng: Random):
    """Dispatch to the correct prompt loader based on --dataset."""
    if args.dataset == "sharegpt":
        if not args.dataset_path:
            raise ValueError(
                "--dataset-path (or SHAREGPT_PATH env var) is required when "
                "--dataset sharegpt is used."
            )
        return _load_sharegpt_prompts(
            path=args.dataset_path,
            tokenizer=tokenizer,
            num_seqs=args.num_seqs,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            temperature=args.temperature,
            chat_template=args.chat_template,
            rng=rng,
        )
    return _build_synthetic_prompts(
        tokenizer,
        args.num_seqs,
        args.max_output_len,
        args.temperature,
        args.chat_template,
    )


def main():
    args = parse_args()
    seed(args.seed)
    rng = Random(args.seed)

    model_path = os.path.expanduser(
        os.environ.get("MODEL_PATH", "~/huggingface/Qwen3-0.6B/")
    )

    # Tokenizer is needed to encode prompts; load it once here.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    prompt_token_ids, sampling_params = build_prompts(args, tokenizer, rng)

    results_ar = None
    results_spec = None

    # ---- AR baseline -------------------------------------------------------
    if args.compare or not args.method:
        print("\n[Baseline] Loading AR model …")
        ar_config = Config(model=model_path, enforce_eager=False, max_model_len=4096)
        llm_ar = LLM(model=model_path, model_config=ar_config)
        ttft_ar = measure_ttft(llm_ar, prompt_token_ids)
        results_ar = run_benchmark(
            llm_ar, prompt_token_ids, sampling_params,
            label="Baseline (autoregressive)",
            show_output=args.show_output,
            tokenizer=tokenizer,
        )
        results_ar["ttft_s"] = ttft_ar
        print_stats(results_ar)
        llm_ar.exit()   # tears down dist process group so the next LLM can init
        del llm_ar
        torch.cuda.empty_cache()

    # ---- Speculative decode ------------------------------------------------
    if args.method:
        print(f"\n[{args.method.upper()}] Loading speculative model …")
        spec_config = Config(
            model=model_path,
            enforce_eager=False,
            max_model_len=4096,
            speculative_method=args.method,
            num_speculative_tokens=args.num_heads,
            medusa_num_layers=args.num_layers,
            medusa_model_path=args.medusa_path,
        )
        llm_spec = LLM(model=model_path, model_config=spec_config)
        ttft_spec = measure_ttft(llm_spec, prompt_token_ids)
        results_spec = run_benchmark(
            llm_spec, prompt_token_ids, sampling_params,
            label=f"Speculative ({args.method}, K={args.num_heads})",
            return_metrics=True,
            show_output=args.show_output,
            tokenizer=tokenizer,
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
