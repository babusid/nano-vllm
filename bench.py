import os
import time
from random import randint, seed
from nanovllm.config import Config
import torch
from nanovllm import LLM, SamplingParams

# from vllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser(os.environ.get("MODEL_PATH", "~/huggingface/Qwen3-0.6B/"))
    main_model_config = Config(model=path, max_model_len=4096, enforce_eager=False)
    llm = LLM(
        model=path,
        model_config=main_model_config,
    )

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
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    # Warmup pass so kernel compilation/setup does not pollute benchmark timing.
    warmup_n = min(32, num_seqs)
    llm.generate(
        prompt_token_ids[:warmup_n], sampling_params[:warmup_n], use_tqdm=False
    )
    torch.cuda.synchronize()

    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    torch.cuda.synchronize()
    t = time.time() - t
    throughput = total_tokens / t
    print(
        f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s"
    )


if __name__ == "__main__":
    main()
