import os
import random
import time

from datasets import load_dataset
import torch

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


def _build_llm(
    model_path: str,
    spec_path: str,
    spec_mode: str,
    spec_length: int,
    main_max_model_len: int,
    main_gpu_memory_utilization: float,
    spec_max_model_len: int,
    spec_gpu_memory_utilization: float,
    enforce_eager: bool,
) -> LLM:
    main_config = Config(
        model=model_path,
        max_model_len=main_max_model_len,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=main_gpu_memory_utilization,
    )

    spec_kwargs = {}
    if spec_mode == "naive":
        if not spec_path:
            raise ValueError("SPEC_MODEL_PATH must be set when SPEC_MODE=naive")
        spec_config = Config(
            model=spec_path,
            max_model_len=spec_max_model_len,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=spec_gpu_memory_utilization,
        )
        spec_kwargs = dict(
            speculation_mode=SpeculationMode.NAIVE_SPECULATION,
            speculator_config=[spec_config],
            speculation_length=spec_length,
        )

    return LLM(model_config=main_config, **spec_kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def bench_arc():
    num_examples = int(os.environ.get("ARC_NUM_EXAMPLES", "200"))
    warmup_examples = int(os.environ.get("ARC_WARMUP_EXAMPLES", "16"))
    sampling_seed = int(os.environ.get("ARC_SEED", "0"))
    temperature = float(os.environ.get("ARC_TEMPERATURE", "1e-9"))
    random.seed(sampling_seed)
    torch.manual_seed(sampling_seed)
    torch.cuda.manual_seed_all(sampling_seed)

    dataset_name = os.environ.get("ARC_DATASET_NAME", "allenai/ai2_arc")
    dataset_config = os.environ.get("ARC_DATASET_CONFIG", "ARC-Easy")
    dataset_split = os.environ.get("ARC_DATASET_SPLIT", "test")
    dataset_cache_dir = os.environ.get("ARC_CACHE_DIR", "") or None

    spec_mode = os.environ.get("SPEC_MODE", "none").lower()
    if spec_mode not in {"none", "naive"}:
        raise ValueError(
            f"SPEC_MODE must be one of ['none', 'naive'], got {spec_mode!r}"
        )
    spec_length = int(os.environ.get("SPEC_LENGTH", "1"))
    if spec_length < 1:
        raise ValueError(f"SPEC_LENGTH must be >= 1, got {spec_length}")

    model_path = os.path.expanduser(
        os.environ.get("MAIN_MODEL_PATH")
        or os.environ.get("MODEL_PATH", "~/huggingface/Qwen3-8B/")
    )
    spec_path = os.path.expanduser(os.environ.get("SPEC_MODEL_PATH", ""))
    main_max_model_len = int(os.environ.get("ARC_MAIN_MAX_MODEL_LEN", "4096"))
    main_gpu_memory_utilization = float(
        os.environ.get("ARC_MAIN_GPU_MEMORY_UTILIZATION", "0.8")
    )
    spec_max_model_len = int(os.environ.get("ARC_SPEC_MAX_MODEL_LEN", "4096"))
    spec_gpu_memory_utilization = float(
        os.environ.get("ARC_SPEC_GPU_MEMORY_UTILIZATION", "0.5")
    )
    enforce_eager = os.environ.get("ENFORCE_EAGER", "0") == "1"

    print(f"Spec mode          : {spec_mode}")
    print(f"Spec length        : {spec_length if spec_mode == 'naive' else '-'}")
    print(f"Model              : {model_path}")
    if spec_mode == "naive":
        print(f"Draft              : {spec_path}")
    print(f"Examples           : {num_examples}")
    print(f"Warmup examples    : {warmup_examples}")
    print(f"Seed               : {sampling_seed}")
    print(f"Temperature        : {temperature}")
    print(f"Dataset            : {dataset_name}/{dataset_config}:{dataset_split}")
    if dataset_cache_dir:
        print(f"Dataset cache dir  : {dataset_cache_dir}")
    print()

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=dataset_split,
        cache_dir=dataset_cache_dir,
    )
    dataset = dataset.select(range(min(num_examples, len(dataset))))

    prompts, answers = zip(*[_format_prompt(row) for row in dataset])

    llm = _build_llm(
        model_path=model_path,
        spec_path=spec_path,
        spec_mode=spec_mode,
        spec_length=spec_length,
        main_max_model_len=main_max_model_len,
        main_gpu_memory_utilization=main_gpu_memory_utilization,
        spec_max_model_len=spec_max_model_len,
        spec_gpu_memory_utilization=spec_gpu_memory_utilization,
        enforce_eager=enforce_eager,
    )

    sampling_params = SamplingParams(temperature=temperature, max_tokens=1)

    warmup_n = min(warmup_examples, len(prompts))
    llm.generate(list(prompts[:warmup_n]), [sampling_params] * warmup_n, use_tqdm=False)
    torch.cuda.synchronize()

    drafts_before = llm.spec_drafts_total
    accepted_before = llm.spec_accepted_total
    t0 = time.time()
    outputs = llm.generate(
        list(prompts), [sampling_params] * len(prompts), use_tqdm=True
    )
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    correct = 0
    for answer, output in zip(answers, outputs):
        generated = output["text"].strip()[:1].upper()
        if generated == answer:
            correct += 1

    accuracy = correct / len(answers) * 100
    print(f"\nARC-Easy accuracy : {correct}/{len(answers)} = {accuracy:.1f}%")
    print(f"Total time        : {elapsed:.1f}s")
    print(f"Throughput        : {len(answers) / elapsed:.1f} questions/s")
    drafts = llm.spec_drafts_total - drafts_before
    accepted = llm.spec_accepted_total - accepted_before
    if drafts:
        rate = accepted / drafts
        print(
            f"Spec: drafted={drafts}tok, accepted={accepted}tok, "
            f"acceptance={rate:.2%}"
        )


if __name__ == "__main__":
    bench_arc()
