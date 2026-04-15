import os
from nanovllm import LLM, SamplingParams
from nanovllm.config import Config
from transformers import AutoTokenizer
from nanovllm.engine.llm_engine import SpeculationMode
from nanovllm.config import Config


def main():
    path = os.path.expanduser(
        os.environ.get("MAIN_MODEL_PATH", "~/huggingface/Qwen3-0.6B/")
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
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
        model_config=main_model_config,
        speculation_mode=SpeculationMode.NAIVE_SPECULATION,
        speculator_config=[small_model_config],
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
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
