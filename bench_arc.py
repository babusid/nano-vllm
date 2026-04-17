"""
ARC-Easy accuracy benchmark for nano-vllm.

Modes
-----
  plain   – single model, no speculation
  spec    – two-model naive speculative decoding
  medusa  – single Medusa checkpoint (self-speculative)

Environment variables
---------------------
  MODEL_PATH          path to main (or Medusa) checkpoint
  SPEC_MODEL_PATH     path to draft checkpoint (spec mode only)
  BENCH_MODE          plain | spec | medusa   (default: plain)
  ARC_NUM_EXAMPLES    how many test questions to evaluate (default: 200)

Usage (local)
-------------
  MODEL_PATH=~/huggingface/vicuna-7b-v1.5 python bench_arc.py

Usage (Modal)
-------------
  modal run run_modal.py --target arc --model "lmsys/vicuna-7b-v1.5"
  modal run run_modal.py --target arc --mode spec \
      --main-model "Qwen/Qwen3-8B" --spec-model "Qwen/Qwen3-0.6B"
"""

import os
import time

from datasets import load_dataset

from nanovllm import LLM, SamplingParams
from nanovllm.config import Config
from nanovllm.engine.llm_engine import SpeculationMode


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

_LETTER_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}


def _format_prompt(row: dict) -> tuple[str, str]:
    """Return (prompt, correct_letter)."""
    question = row["question"]
    labels = row["choices"]["label"]
    texts = row["choices"]["text"]
    choices_str = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
    prompt = (
        f"The following is a multiple-choice question. "
        f"Reply with only the letter of the correct answer.\n\n"
        f"Question: {question}\n{choices_str}\nAnswer:"
    )
    answer = row["answerKey"].strip()
    answer = _LETTER_MAP.get(answer, answer)
    return prompt, answer


# ---------------------------------------------------------------------------
# LLM construction
# ---------------------------------------------------------------------------

def _build_llm(mode: str, model_path: str, spec_path: str) -> LLM:
    main_config = Config(
        model=model_path,
        max_model_len=2048,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
    )

    if mode == "plain":
        return LLM(model=model_path, model_config=main_config)

    if mode == "medusa":
        from nanovllm.engine.llm_engine import SpeculationMode
        return LLM(
            model=model_path,
            model_config=main_config,
            speculation_mode=SpeculationMode.MEDUSA,
        )

    if mode == "spec":
        if not spec_path:
            raise ValueError("SPEC_MODEL_PATH must be set for spec mode")
        spec_config = Config(
            model=spec_path,
            max_model_len=2048,
            enforce_eager=True,
            gpu_memory_utilization=0.5,
        )
        return LLM(
            model=model_path,
            model_config=main_config,
            speculation_mode=SpeculationMode.NAIVE_SPECULATION,
            speculation_model=[spec_path],
            speculator_config=[spec_config],
        )

    raise ValueError(f"Unknown mode {mode!r}. Choose plain / spec / medusa.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mode = os.environ.get("BENCH_MODE", "plain").lower()
    model_path = os.path.expanduser(
        os.environ.get("MODEL_PATH", "~/huggingface/Qwen3-0.6B/")
    )
    spec_path = os.path.expanduser(os.environ.get("SPEC_MODEL_PATH", ""))
    num_examples = int(os.environ.get("ARC_NUM_EXAMPLES", "200"))

    print(f"Mode      : {mode}")
    print(f"Model     : {model_path}")
    if mode == "spec":
        print(f"Draft     : {spec_path}")
    print(f"Examples  : {num_examples}")
    print()

    # Load dataset
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    dataset = dataset.select(range(min(num_examples, len(dataset))))

    prompts, answers = zip(*[_format_prompt(row) for row in dataset])

    # Build LLM (agnostic to spec strategy after this point)
    llm = _build_llm(mode, model_path, spec_path)

    # Single token is enough — we just want the answer letter
    sampling_params = SamplingParams(temperature=0.01, max_tokens=1)

    # Warmup
    warmup_n = min(16, len(prompts))
    llm.generate(list(prompts[:warmup_n]), [sampling_params] * warmup_n, use_tqdm=False)

    # Benchmark
    t0 = time.time()
    outputs = llm.generate(list(prompts), [sampling_params] * len(prompts))
    elapsed = time.time() - t0

    # Score
    correct = 0
    for answer, output in zip(answers, outputs):
        # Take the first non-space character of the completion
        generated = output["text"].strip()[:1].upper()
        if generated == answer:
            correct += 1

    accuracy = correct / len(answers) * 100
    print(f"\nARC-Easy accuracy : {correct}/{len(answers)} = {accuracy:.1f}%")
    print(f"Total time        : {elapsed:.1f}s")
    print(f"Throughput        : {len(answers) / elapsed:.1f} questions/s")


if __name__ == "__main__":
    main()
