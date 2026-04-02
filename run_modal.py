"""Run nano-vLLM scripts remotely on Modal.

Setup (one-time):
    pip install modal
    modal setup

Usage:
    # Run benchmark script
    modal run run_modal.py --target bench

    # Run example script
    modal run run_modal.py --target example

    # Use a different Hugging Face model and revision
    modal run run_modal.py --target bench --model "Qwen/Qwen3-0.6B" --revision "main"
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


@app.function(
    image=image,
    gpu="B200:1",
    timeout=7200,
    volumes={"/root/huggingface": hf_volume},
)
def run_target(target: str, model: str, revision: str = "") -> tuple[int, str, str]:
    import os
    import subprocess

    from huggingface_hub import snapshot_download

    if target not in {"bench", "example"}:
        raise ValueError(f"target must be one of ['bench', 'example'], got {target!r}")

    model_name = model.rstrip("/").split("/")[-1]
    model_path = f"/root/huggingface/{model_name}"

    download_kwargs = {
        "repo_id": model,
        "local_dir": model_path,
    }
    if revision:
        download_kwargs["revision"] = revision
    snapshot_download(**download_kwargs)

    env = os.environ.copy()
    env["MODEL_PATH"] = model_path

    result = subprocess.run(
        ["python", f"/workspace/{target}.py"],
        cwd="/workspace",
        env=env,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


@app.local_entrypoint()
def main(target: str = "bench", model: str = "Qwen/Qwen3-0.6B", revision: str = ""):
    try:
        returncode, stdout, stderr = run_target.remote(target, model, revision)
    except Exception as exc:  # pragma: no cover
        print(f"Modal execution failed: {exc}")
        sys.exit(1)

    if stdout:
        print(stdout, end="")
    if stderr:
        print(stderr, file=sys.stderr, end="")

    if returncode != 0:
        print(f"\n{target}.py failed with exit code {returncode}", file=sys.stderr)
        sys.exit(returncode)
