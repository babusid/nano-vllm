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

    # MEDUSA vs AR — greedy, single-sequence (intended sweet spot, expect ~1.5-2.5x speedup)
    modal run run_modal.py --target bench_speculative \
        --model "lmsys/vicuna-7b-v1.3" \
        --medusa-heads-model "FasterDecoding/medusa-vicuna-7b-v1.3" \
        --extra-args "--method medusa --num-heads 5 --compare --num-seqs 1 --temperature 0 --show-output --chat-template vicuna --max-output-len 256"

    # MEDUSA with Vicuna heads + ShareGPT dataset (most realistic benchmark)
    modal run run_modal.py --target bench_speculative \
        --model "lmsys/vicuna-7b-v1.3" \
        --medusa-heads-model "FasterDecoding/medusa-vicuna-7b-v1.3" \
        --sharegpt-dataset "anon8231489123/ShareGPT_Vicuna_unfiltered" \
        --extra-args "--method medusa --num-heads 5 --compare --num-seqs 1 --temperature 0 --dataset sharegpt --chat-template vicuna"

    # AR baseline only (Vicuna)
    modal run run_modal.py --target bench_speculative \
        --model "lmsys/vicuna-7b-v1.3" \
        --extra-args "--num-seqs 1 --temperature 0 --show-output --chat-template vicuna"
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


def _download_sharegpt(repo_id: str, revision: str = "") -> str:
    """Download the ShareGPT dataset file and return the path to the JSON.

    The canonical benchmark source used by vLLM / SGLang is:
      anon8231489123/ShareGPT_Vicuna_unfiltered
    which contains ShareGPT_V3_unfiltered_cleaned_split.json.

    We attempt to download that specific file first; if it is not present in
    the repo we fall back to a full snapshot and pick the best JSON candidate.
    """
    from huggingface_hub import hf_hub_download, snapshot_download
    import glob as _glob
    import os as _os

    repo_name = repo_id.rstrip("/").split("/")[-1]
    dataset_dir = f"/root/sharegpt/{repo_name}"
    _os.makedirs(dataset_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Attempt 1: download the canonical vLLM benchmark file directly.
    # ------------------------------------------------------------------ #
    canonical_name = "ShareGPT_V3_unfiltered_cleaned_split.json"
    try:
        json_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=canonical_name,
            local_dir=dataset_dir,
            **({"revision": revision} if revision else {}),
        )
        print(f"[ShareGPT] Using dataset file: {json_path}")
        return json_path
    except Exception:
        pass  # file not in repo, fall through to snapshot

    # ------------------------------------------------------------------ #
    # Attempt 2: snapshot the whole repo (skip large non-JSON files).
    # ------------------------------------------------------------------ #
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=dataset_dir,
        ignore_patterns=["*.bin", "*.safetensors", "*.parquet", "*.arrow"],
        **({"revision": revision} if revision else {}),
    )

    candidates = sorted(_glob.glob(f"{dataset_dir}/**/*.json", recursive=True))
    # Prefer files that look like the vLLM benchmark file.
    priority = [
        p for p in candidates
        if "V3" in p or ("unfiltered" in p and "cleaned" in p and "split" in p)
    ]
    json_path = priority[0] if priority else (candidates[0] if candidates else None)

    if not json_path:
        raise FileNotFoundError(
            f"No JSON file found in downloaded ShareGPT dataset at {dataset_dir}"
        )
    print(f"[ShareGPT] Using dataset file: {json_path}")
    return json_path


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
    sharegpt_dataset: str = "",
    sharegpt_revision: str = "",
    mode: str = "",
) -> tuple[int, str, str]:
    import os
    import subprocess

    valid_targets = {"bench", "example", "bench_speculative", "arc", "medusa"}
    if target not in valid_targets:
        raise ValueError(f"target must be one of {sorted(valid_targets)}, got {target!r}")
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

    elif target == "bench_speculative":
        # MEDUSA uses a single model; MODEL_PATH drives bench_speculative.py
        repo = model or "Qwen/Qwen3-0.6B"
        env["MODEL_PATH"] = _download_model(repo, revision)
        # Optionally download pretrained MEDUSA heads from a separate HF repo
        if medusa_heads_model:
            heads_dir = _download_medusa_heads(medusa_heads_model, medusa_heads_revision)
            env["MEDUSA_HEADS_PATH"] = heads_dir
            print(f"[MEDUSA] Heads downloaded to {heads_dir}")
        # Optionally download ShareGPT dataset from HuggingFace
        if sharegpt_dataset:
            json_path = _download_sharegpt(sharegpt_dataset, sharegpt_revision)
            env["SHAREGPT_PATH"] = json_path
            print(f"[ShareGPT] Dataset path set to {json_path}")
    else:
        # example / any single-model script
        env["MODEL_PATH"] = _download_model(model, revision)

    script_name = {"arc": "bench_arc", "medusa": "bench_medusa"}.get(target, target)
    cmd = ["python", f"/workspace/{script_name}.py"]
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
    sharegpt_dataset: str = "",
    sharegpt_revision: str = "",
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
            extra_args,
            medusa_heads_model,
            medusa_heads_revision,
            sharegpt_dataset,
            sharegpt_revision,
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
        print(f"\n{target} failed with exit code {returncode}", file=sys.stderr)
        sys.exit(returncode)
