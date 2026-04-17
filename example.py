import os
import time
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.config import Config
from transformers import AutoTokenizer
from nanovllm.engine.llm_engine import SpeculationMode
from nanovllm.config import Config


def example():
    example = os.path.expanduser(
        os.environ.get("MAIN_MODEL_PATH", "~/huggingface/Qwen3-0.6B/")
    )
    tokenizer = AutoTokenizer.from_pretrained(example)

    # speculation config comes from env so run_modal.py flags can propagate
    spec_mode_str = os.environ.get("SPEC_MODE", "none").lower()
    spec_length = int(os.environ.get("SPEC_LENGTH", "1"))
    use_spec = spec_mode_str == "naive"
    print(f"Spec: mode={spec_mode_str} length={spec_length if use_spec else '-'}")

    # size memory pool to add up to 90% of GPU memory
    main_model_path = os.path.expanduser(
        os.environ.get("MAIN_MODEL_PATH")
        or os.environ.get("MODEL_PATH", "~/huggingface/Qwen3-8B/")
    )
    main_model_config = Config(
        model=main_model_path,
        max_model_len=4096,
        enforce_eager=False,
        gpu_memory_utilization=0.8,
    )
    print("Main Model Path: ", main_model_path)

    # only construct the speculator config when naive spec is requested —
    # Config.__post_init__ hits the filesystem / HF cache
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

    llm = LLM(
        model_config=main_model_config,
        **spec_kwargs,
    )

    sampling_params = SamplingParams(temperature=0.1, max_tokens=256)
    prompts = [
        "who are you?",
    ]
    if tokenizer.chat_template:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
    else:
        # Vicuna-style template (no chat_template set in tokenizer)
        system = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        for prompt in prompts
    ]
    # snapshot spec counters so we only report drafts from this run
    drafts_before = llm.spec_drafts_total
    accepted_before = llm.spec_accepted_total
    t = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    t = time.time() - t

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

    total_tokens = sum(len(output["token_ids"]) for output in outputs)
    throughput = total_tokens / t if t > 0 else 0.0
    print(
        f"\nTotal: {total_tokens}tok, Time: {t:.2f}s, "
        f"Throughput: {throughput:.2f}tok/s"
    )
    drafts = llm.spec_drafts_total - drafts_before
    accepted = llm.spec_accepted_total - accepted_before
    if drafts:
        rate = accepted / drafts
        print(
            f"Spec: drafted={drafts}tok, accepted={accepted}tok, "
            f"acceptance={rate:.2%}"
        )


if __name__ == "__main__":
    example()
