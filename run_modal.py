"""Run nano-vLLM scripts remotely on Modal.

Setup (one-time):
    pip install modal
    modal setup

Usage:
    # Throughput benchmark (8B main + 0.6B speculator by default)
    modal run run_modal.py --target bench

    # Single-model example
    modal run run_modal.py --target example --model "Qwen/Qwen3-0.6B"

    # Medusa self-speculative benchmark
    modal run run_modal.py --target medusa --model "FasterDecoding/medusa-vicuna-7b-v1.3"

    # ARC-Easy accuracy — plain single model
    modal run run_modal.py --target arc --model "Qwen/Qwen3-0.6B"

    # ARC-Easy accuracy — two-model speculative decoding
    modal run run_modal.py --target arc --mode spec \
        --main-model "Qwen/Qwen3-8B" --spec-model "Qwen/Qwen3-0.6B"

    # ARC-Easy accuracy — Medusa self-speculative
    modal run run_modal.py --target arc --mode medusa \
        --model "FasterDecoding/medusa-vicuna-7b-v1.3"
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
        "datasets",
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
    import os
    from huggingface_hub import snapshot_download

    model_name = repo_id.rstrip("/").split("/")[-1]
    model_path = f"/root/huggingface/{model_name}"

    # Skip the HuggingFace metadata round-trip when already cached in the volume.
    if os.path.isfile(os.path.join(model_path, "config.json")):
        print(f"Using cached model: {model_path}")
        return model_path

    print(f"Downloading {repo_id} → {model_path}")
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
    model: str,
    revision: str = "",
    main_model: str = "",
    main_revision: str = "",
    spec_model: str = "",
    spec_revision: str = "",
    mode: str = "",
) -> tuple[int, str, str]:
    import os
    import subprocess

    _VALID = {"bench", "example", "arc", "medusa"}
    if target not in _VALID:
        raise ValueError(f"target must be one of {sorted(_VALID)}, got {target!r}")
    print("Target:", target)
    env = os.environ.copy()

    if target == "bench":
        main_repo = main_model or "Qwen/Qwen3-8B"
        spec_repo = spec_model or "Qwen/Qwen3-0.6B"
        env["MAIN_MODEL_PATH"] = _download_model(main_repo, main_revision)
        env["SPEC_MODEL_PATH"] = _download_model(spec_repo, spec_revision)

    elif target == "medusa":
        medusa_repo = model or "FasterDecoding/medusa-vicuna-7b-v1.3"
        env["MEDUSA_MODEL_PATH"] = _download_model(medusa_repo, revision)

    elif target == "arc":
        # mode determines which LLM strategy bench_arc.py uses.
        bench_mode = mode or "plain"
        env["BENCH_MODE"] = bench_mode
        if bench_mode == "spec":
            main_repo = main_model or "Qwen/Qwen3-8B"
            spec_repo = spec_model or "Qwen/Qwen3-0.6B"
            env["MODEL_PATH"] = _download_model(main_repo, main_revision)
            env["SPEC_MODEL_PATH"] = _download_model(spec_repo, spec_revision)
        elif bench_mode == "medusa":
            medusa_repo = model or "FasterDecoding/medusa-vicuna-7b-v1.3"
            env["MODEL_PATH"] = _download_model(medusa_repo, revision)
        else:
            # plain
            env["MODEL_PATH"] = _download_model(model or "Qwen/Qwen3-0.6B", revision)

    else:
        # example / any single-model script
        env["MODEL_PATH"] = _download_model(model, revision)

    script_name = {"arc": "bench_arc", "medusa": "bench_medusa"}.get(target, target)
    result = subprocess.run(
        ["python", f"/workspace/{script_name}.py"],
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
    mode: str = "",
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
            mode,
        )
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
