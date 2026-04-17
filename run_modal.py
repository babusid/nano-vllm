"""Run nano-vLLM scripts remotely on Modal.

Setup (one-time):
    pip install modal
    modal setup

Usage:
    # Run benchmark script (8B main, no speculation by default)
    modal run run_modal.py --target bench

    # Run example script (single model)
    modal run run_modal.py --target example

    # Run benchmark with naive speculative decoding (default length 1)
    modal run run_modal.py --target bench --spec-mode naive

    # Run benchmark with naive speculation, length 8
    modal run run_modal.py --target bench --spec-mode naive --spec-length 8

    # Run benchmark with custom models
    modal run run_modal.py --target bench --main-model "Qwen/Qwen3-8B" --spec-model "Qwen/Qwen3-0.6B"

    # Run example with custom model and revision
    modal run run_modal.py --target example --main-model "Qwen/Qwen3-0.6B" --main-revision "main"
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


def _download_model(repo_id: str, revision: str = "", enable_cache: bool = True) -> str:
    import os
    from huggingface_hub import snapshot_download

    model_name = repo_id.rstrip("/").split("/")[-1]
    model_path = f"/root/huggingface/{model_name}"
    if os.path.isfile(os.path.join(model_path, "config.json")) and enable_cache:
        print(f"using cached model: {model_path}")
        return model_path

    download_kwargs = {"repo_id": repo_id, "local_dir": model_path}
    if revision:
        download_kwargs["revision"] = revision
    snapshot_download(**download_kwargs)
    return model_path


@app.function(
    image=image,
    gpu="B200:1",
    timeout=7200,
    volumes={"/root/huggingface": hf_volume},
)
def run_target(
    target: str,
    main_model: str = "",
    main_revision: str = "",
    spec_model: str = "",
    spec_revision: str = "",
    spec_mode: str = "none",
    spec_length: int = 1,
) -> None:
    import os
    import runpy
    import sys

    if target not in {"bench", "example"}:
        raise ValueError(f"target must be one of ['bench', 'example'], got {target!r}")
    spec_mode_norm = spec_mode.lower()
    if spec_mode_norm not in {"none", "naive"}:
        raise ValueError(
            f"spec_mode must be one of ['none', 'naive'], got {spec_mode!r}"
        )
    if spec_length < 1:
        raise ValueError(f"spec_length must be >= 1, got {spec_length}")
    print("Target: ", target)
    print(f"Spec: mode={spec_mode_norm} length={spec_length}")

    main_repo = main_model or "Qwen/Qwen3-8B"
    os.environ["MAIN_MODEL_PATH"] = _download_model(main_repo, main_revision)
    # only pull the speculator when we're actually going to use it
    if spec_mode_norm != "none":
        spec_repo = spec_model or "Qwen/Qwen3-0.6B"
        os.environ["SPEC_MODEL_PATH"] = _download_model(spec_repo, spec_revision)

    # propagate spec config to the target script via env
    os.environ["SPEC_MODE"] = spec_mode_norm
    os.environ["SPEC_LENGTH"] = str(spec_length)

    workspace_dir = "/workspace"
    if workspace_dir not in sys.path:
        sys.path.insert(0, workspace_dir)

    script_path = os.path.join(workspace_dir, f"{target}.py")
    if not os.path.isfile(script_path):
        script_path = f"/{target}.py"

    runpy.run_path(script_path, run_name="__main__")


@app.local_entrypoint()
def main(
    target: str = "bench",
    main_model: str = "Qwen/Qwen3-8B",
    main_revision: str = "",
    spec_model: str = "Qwen/Qwen3-0.6B",
    spec_revision: str = "",
    spec_mode: str = "none",
    spec_length: int = 1,
):
    try:
        run_target.remote(
            target,
            main_model,
            main_revision,
            spec_model,
            spec_revision,
            spec_mode,
            spec_length,
        )
    except Exception as exc:  # pragma: no cover
        print(f"Modal execution failed: {exc}")
        sys.exit(1)
