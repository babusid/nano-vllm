import os
from nanovllm import LLM, SamplingParams
from nanovllm.config import Config
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser(os.environ.get("MODEL_PATH", "~/huggingface/Qwen3-0.6B/"))
    tokenizer = AutoTokenizer.from_pretrained(path)
    model_config = Config(model=path, enforce_eager=True, tensor_parallel_size=1)
    llm = LLM(model=path, model_config=model_config)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
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
        prompts = [f"{system} USER: {prompt} ASSISTANT:" for prompt in prompts]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
