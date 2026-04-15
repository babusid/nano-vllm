"""Run nano-vLLM scripts remotely on Modal.

Setup (one-time):
    pip install modal
    modal setup

Usage:
    # Run benchmark script (8B main + 0.6B speculator by default)
    modal run run_modal.py --target bench

    # Run example script (single model)
    modal run run_modal.py --target example

    # Run benchmark with custom models
    modal run run_modal.py --target bench --main-model "Qwen/Qwen3-8B" --spec-model "Qwen/Qwen3-0.6B"

    # Run example with custom model and revision
    modal run run_modal.py --target example --model "Qwen/Qwen3-0.6B" --revision "main"

    # Run MEDUSA vs AR comparison benchmark (bench_speculative.py)
    modal run run_modal.py --target bench_speculative --model "Qwen/Qwen3-0.6B" --extra-args "--method medusa --num-heads 4 --compare"

    # MEDUSA with pretrained Vicuna-7B heads (FasterDecoding/medusa-vicuna-7b-v1.3)
    modal run run_modal.py --target bench_speculative --model "lmsys/vicuna-7b-v1.3" --medusa-heads-model "FasterDecoding/medusa-vicuna-7b-v1.3" --extra-args "--method medusa --num-heads 5 --compare"

    # AR baseline only
    modal run run_modal.py --target bench_speculative --model "Qwen/Qwen3-0.6B"
"""

from __future__ import annotations

import sys

import modal

app = modal.App("nano-vllm-runner")

hf_volume = modal.Volume.from_name("nano-vllm-hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends git build-essential && rm -rf /var/lib/apt/lists/*",
    )
    .run_commands("python -m pip install --upgrade pip")
    .run_commands(
        "python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.8.*'",
    )
    .pip_install(
        "transformers==4.51.0",
        "huggingface_hub>=0.25.0",
        "xxhash",
        "tiktoken",
        "sentencepiece",
    )
    .run_commands(
        "python -m pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl' || python -m pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'",
    )
    .add_local_dir(
        ".",
        remote_path="/workspace",
        ignore=[".git", "__pycache__", ".venv", "*.pyc"],
    )
)


def _download_model(repo_id: str, revision: str = "") -> str:
    from huggingface_hub import snapshot_download

    model_name = repo_id.rstrip("/").split("/")[-1]
    model_path = f"/root/huggingface/{model_name}"
    download_kwargs = {"repo_id": repo_id, "local_dir": model_path}
    if revision:
        download_kwargs["revision"] = revision
    snapshot_download(**download_kwargs)
    return model_path


def _download_medusa_heads(repo_id: str, revision: str = "") -> str:
    """Download just the MEDUSA heads file and return the directory path.

    The official FasterDecoding checkpoints store heads in medusa_lm_head.pt
    inside the repo.  We download the full snapshot so the caller can point
    load_medusa_heads() at the directory.
    """
    from huggingface_hub import snapshot_download

    repo_name = repo_id.rstrip("/").split("/")[-1]
    heads_path = f"/root/huggingface/{repo_name}"
    download_kwargs = {
        "repo_id": repo_id,
        "local_dir": heads_path,
        # Only fetch the heads file and config; skip large model shards
        "ignore_patterns": ["*.bin", "*.safetensors", "pytorch_model*"],
    }
    if revision:
        download_kwargs["revision"] = revision
    snapshot_download(**download_kwargs)
    return heads_path


@app.function(
    image=image,
    gpu="B200:1",
    timeout=7200,
    volumes={"/root/huggingface": hf_volume},
)
def run_target(
    target: str,
    model: str,
    revision: str = "",
    main_model: str = "",
    main_revision: str = "",
    spec_model: str = "",
    spec_revision: str = "",
    extra_args: str = "",
    medusa_heads_model: str = "",
    medusa_heads_revision: str = "",
) -> tuple[int, str, str]:
    import os
    import subprocess

    valid_targets = {"bench", "example", "bench_speculative"}
    if target not in valid_targets:
        raise ValueError(f"target must be one of {sorted(valid_targets)}, got {target!r}")
    print("Target: ", target)
    env = os.environ.copy()
    if target == "bench":
        main_repo = main_model or "Qwen/Qwen3-8B"
        spec_repo = spec_model or "Qwen/Qwen3-0.6B"
        env["MAIN_MODEL_PATH"] = _download_model(main_repo, main_revision)
        env["SPEC_MODEL_PATH"] = _download_model(spec_repo, spec_revision)
    elif target == "bench_speculative":
        # MEDUSA uses a single model; MODEL_PATH drives bench_speculative.py
        repo = model or "Qwen/Qwen3-0.6B"
        env["MODEL_PATH"] = _download_model(repo, revision)
        # Optionally download pretrained MEDUSA heads from a separate HF repo
        if medusa_heads_model:
            heads_dir = _download_medusa_heads(medusa_heads_model, medusa_heads_revision)
            env["MEDUSA_HEADS_PATH"] = heads_dir
            print(f"[MEDUSA] Heads downloaded to {heads_dir}")
    else:
        env["MODEL_PATH"] = _download_model(model, revision)

    cmd = ["python", f"/workspace/{target}.py"]
    if extra_args:
        cmd += extra_args.split()

    result = subprocess.run(
        cmd,
        cwd="/workspace",
        env=env,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


@app.local_entrypoint()
def main(
    target: str = "bench",
    model: str = "Qwen/Qwen3-0.6B",
    revision: str = "",
    main_model: str = "Qwen/Qwen3-8B",
    main_revision: str = "",
    spec_model: str = "Qwen/Qwen3-0.6B",
    spec_revision: str = "",
    extra_args: str = "",
    medusa_heads_model: str = "",
    medusa_heads_revision: str = "",
):
    try:
        returncode, stdout, stderr = run_target.remote(
            target,
            model,
            revision,
            main_model,
            main_revision,
            spec_model,
            spec_revision,
            extra_args,
            medusa_heads_model,
            medusa_heads_revision,
        )
    except Exception as exc:  # pragma: no cover
        print(f"Modal execution failed: {exc}")
        sys.exit(1)

    if stdout:
        print(stdout, end="")
    if stderr:
        print(stderr, file=sys.stderr, end="")

    if returncode != 0:
        print(f"\n{target} failed with exit code {returncode}", file=sys.stderr)
        sys.exit(returncode)
