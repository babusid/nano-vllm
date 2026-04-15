"""Run nano-vLLM scripts remotely on Modal.

Setup (one-time):
    pip install modal
    modal setup

Usage
-----
# Original benchmarks (unchanged):
    modal run run_modal.py --target bench
    modal run run_modal.py --target example

# Train MEDUSA heads (saves heads to the HF volume next to the model):
    modal run run_modal.py --target train --model "Qwen/Qwen3-0.6B"

# Benchmark MEDUSA vs AR baseline (full comparison table):
    modal run run_modal.py --target bench_speculative --model "Qwen/Qwen3-0.6B" \\
        --extra-args "--method medusa --num-heads 4 --compare"

# Train heads and immediately benchmark in one Modal call
# (avoids downloading the model twice):
    modal run run_modal.py --target train_and_bench --model "Qwen/Qwen3-0.6B" \\
        --extra-args "--num-heads 4"

# Sweep over number of heads (K=2,3,4,5):
    modal run run_modal.py --target sweep --model "Qwen/Qwen3-0.6B"
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
        "safetensors",
        "tqdm",
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_TARGETS = {"bench", "example", "bench_speculative", "train", "train_and_bench", "sweep"}


def _download_model(model: str, revision: str) -> str:
    """Download a HF model to the volume and return its local path."""
    from huggingface_hub import snapshot_download

    model_name = model.rstrip("/").split("/")[-1]
    model_path = f"/root/huggingface/{model_name}"
    kwargs = {"repo_id": model, "local_dir": model_path}
    if revision:
        kwargs["revision"] = revision
    snapshot_download(**kwargs)
    return model_path


def _run(cmd: list[str], env: dict) -> tuple[int, str, str]:
    import subprocess
    result = subprocess.run(cmd, cwd="/workspace", env=env, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def _heads_path(model_path: str) -> str:
    import os
    return os.path.join(model_path, "medusa_heads.safetensors")


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------

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
    extra_args: str = "",
) -> tuple[int, str, str]:
    """Generic runner used by all targets except sweep."""
    import os

    if target not in VALID_TARGETS:
        raise ValueError(f"target must be one of {sorted(VALID_TARGETS)}, got {target!r}")

    model_path = _download_model(model, revision)
    env = os.environ.copy()
    env["MODEL_PATH"] = model_path
    heads_path = _heads_path(model_path)

    extra = extra_args.split() if extra_args.strip() else []

    # ---- Original targets (unchanged) -----------------------------------
    if target in {"bench", "example"}:
        return _run(["python", f"/workspace/{target}.py"] + extra, env)

    # ---- Train MEDUSA heads ---------------------------------------------
    if target == "train":
        # Use a small slice of the ShareGPT dataset from HF hub as training text
        print("[train] Downloading training data …", flush=True)
        _fetch_sharegpt(model_path)

        num_heads = _parse_flag(extra, "--num-heads", default="4")
        num_layers = _parse_flag(extra, "--num-layers", default="1")
        epochs = _parse_flag(extra, "--epochs", default="2")

        cmd = [
            "python", "/workspace/train_medusa_heads.py",
            "--model", model_path,
            "--data", f"{model_path}/train_data.jsonl",
            "--num-heads", num_heads,
            "--num-layers", num_layers,
            "--epochs", epochs,
            "--output", heads_path,
        ]
        rc, out, err = _run(cmd, env)
        hf_volume.commit()          # persist trained heads to the volume
        return rc, out, err

    # ---- bench_speculative ----------------------------------------------
    if target == "bench_speculative":
        cmd = ["python", "/workspace/bench_speculative.py"] + extra
        # If heads checkpoint exists on the volume, tell the script where it is
        if _heads_exist(heads_path) and "--medusa-path" not in extra:
            cmd += ["--medusa-path", heads_path]
        return _run(cmd, env)

    # ---- train_and_bench ------------------------------------------------
    if target == "train_and_bench":
        num_heads = _parse_flag(extra, "--num-heads", default="4")
        num_layers = _parse_flag(extra, "--num-layers", default="1")

        # 1. Fetch training data
        _fetch_sharegpt(model_path)

        # 2. Train
        train_cmd = [
            "python", "/workspace/train_medusa_heads.py",
            "--model", model_path,
            "--data", f"{model_path}/train_data.jsonl",
            "--num-heads", num_heads,
            "--num-layers", num_layers,
            "--epochs", "2",
            "--output", heads_path,
        ]
        rc, out, err = _run(train_cmd, env)
        if rc != 0:
            return rc, out, err

        hf_volume.commit()          # persist heads

        # 3. Benchmark AR vs MEDUSA
        bench_cmd = [
            "python", "/workspace/bench_speculative.py",
            "--method", "medusa",
            "--num-heads", num_heads,
            "--medusa-path", heads_path,
            "--compare",
        ]
        rc2, out2, err2 = _run(bench_cmd, env)
        return rc2, out + out2, err + err2

    # ---- sweep ----------------------------------------------------------
    # (handled by run_sweep below; should not reach here)
    raise RuntimeError(f"Unexpected target {target!r}")


@app.function(
    image=image,
    gpu="B200:1",
    timeout=7200,
    volumes={"/root/huggingface": hf_volume},
)
def run_sweep(model: str, revision: str = "") -> tuple[int, str, str]:
    """Run bench_speculative.py for K = 1, 2, 3, 4, 5 and print a summary table.

    Assumes heads have already been trained (run --target train first).
    """
    import os

    model_path = _download_model(model, revision)
    env = os.environ.copy()
    env["MODEL_PATH"] = model_path
    heads_path = _heads_path(model_path)

    all_stdout = ""
    all_stderr = ""

    for k in [1, 2, 3, 4, 5]:
        cmd = [
            "python", "/workspace/bench_speculative.py",
            "--method", "medusa",
            "--num-heads", str(k),
            "--compare",
            "--num-seqs", "64",
        ]
        if _heads_exist(heads_path):
            cmd += ["--medusa-path", heads_path]

        print(f"\n{'='*50}\n  Sweep: K = {k}\n{'='*50}", flush=True)
        rc, out, err = _run(cmd, env)
        all_stdout += f"\n--- K={k} ---\n{out}"
        all_stderr += err
        if rc != 0:
            return rc, all_stdout, all_stderr

    return 0, all_stdout, all_stderr


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _heads_exist(path: str) -> bool:
    import os
    return os.path.isfile(path)


def _parse_flag(args: list[str], flag: str, default: str) -> str:
    """Extract --flag VALUE from a list of args, with fallback to default."""
    try:
        idx = args.index(flag)
        return args[idx + 1]
    except (ValueError, IndexError):
        return default


def _fetch_sharegpt(model_path: str, num_samples: int = 5000) -> None:
    """Download a small slice of ShareGPT and write it as a JSONL training file.

    Uses only the 'human' turns so the model learns to predict natural text.
    Falls back to a tiny synthetic dataset if the download fails.
    """
    import json
    import os

    out_path = os.path.join(model_path, "train_data.jsonl")
    if os.path.isfile(out_path):
        print(f"[train] Training data already exists at {out_path}", flush=True)
        return

    try:
        from datasets import load_dataset
        ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
        texts = []
        for row in ds:
            for turn in row.get("conversations", []):
                if turn.get("from") == "human":
                    texts.append(turn["value"])
                if len(texts) >= num_samples:
                    break
            if len(texts) >= num_samples:
                break
        print(f"[train] Fetched {len(texts)} training samples from ShareGPT", flush=True)
    except Exception as e:
        print(f"[train] ShareGPT download failed ({e}), using synthetic data", flush=True)
        import random
        words = ["the", "model", "predicts", "tokens", "based", "on", "context"]
        texts = [" ".join(random.choices(words, k=100)) for _ in range(500)]

    with open(out_path, "w") as f:
        for text in texts:
            f.write(json.dumps({"text": text}) + "\n")
    print(f"[train] Wrote {len(texts)} samples to {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    target: str = "bench",
    model: str = "Qwen/Qwen3-0.6B",
    revision: str = "",
    extra_args: str = "",
):
    try:
        if target == "sweep":
            returncode, stdout, stderr = run_sweep.remote(model, revision)
        else:
            returncode, stdout, stderr = run_target.remote(target, model, revision, extra_args)
    except Exception as exc:
        print(f"Modal execution failed: {exc}")
        sys.exit(1)

    if stdout:
        print(stdout, end="")
    if stderr:
        print(stderr, file=sys.stderr, end="")

    if returncode != 0:
        print(f"\n{target} failed with exit code {returncode}", file=sys.stderr)
        sys.exit(returncode)
