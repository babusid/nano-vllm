"""Run nano-vLLM scripts remotely on Modal.

Setup (one-time):
    pip install modal
    modal setup

Usage:
    # Run benchmark script (8B main, no speculation by default)
    modal run run_modal.py --target bench

    # Run example script (single model)
    modal run run_modal.py --target example

    # Run ARC-Easy accuracy benchmark
    modal run run_modal.py --target arc

    # Run benchmark with naive speculative decoding (default length 1)
    modal run run_modal.py --target bench --spec-mode naive

    # Run benchmark with naive speculation, length 8
    modal run run_modal.py --target bench --spec-mode naive --spec-length 8

    # Run benchmark with MEDUSA speculative decoding
    modal run run_modal.py --target bench --spec-mode medusa --spec-model "FasterDecoding/medusa-vicuna-7b-v1.3"

    # Run benchmark with custom models
    modal run run_modal.py --target bench --main-model "Qwen/Qwen3-8B" --spec-model "Qwen/Qwen3-0.6B"

    # Run example with custom model and revision
    modal run run_modal.py --target example --main-model "Qwen/Qwen3-0.6B" --main-revision "main"

    # Profile full bench.py execution (CPU-only by default) and save trace JSON to Modal volume
    modal run run_modal.py --target bench --profile

    # Include CUDA activity events too (larger trace, slower export)
    modal run run_modal.py --target bench --profile --profile-cuda

    # Profile with eager kernels (no CUDA graph replay)
    modal run run_modal.py --target bench --profile --enforce-eager

    # Capture per-step prefill/decode throughput CSV
    modal run run_modal.py --target bench --data-capture

    # Disabled (can deadlock): Python stack traces in profiler events
    # modal run run_modal.py --target bench --profile --profile-with-stack

    # Pull traces locally
    modal volume get nano-vllm-profiler-traces / ./traces

Profiling flags:
    --profile
        Enable full-script PyTorch profiling and export a .pt.trace.json file.
    --profile-label
        Optional output trace label used in the trace directory name.
    --profile-record-shapes
        Record tensor shapes for profiled ops.
    --profile-memory
        Record memory usage events.
    --profile-cuda
        Include CUDA activity events in the trace (disabled by default).
    # --profile-with-stack (disabled)
    #     Record Python stack traces for events; useful for attribution but slower.
    #     Disabled due to deadlocks/stalls during profiling finalization/export.
    --enforce-eager
        Disable CUDA graph replay during execution (independent of --profile).
    --data-capture / --trace-throughput
        Capture per-step throughput metrics into data.csv on the Modal trace volume.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4
import sys
import modal
import shutil

app = modal.App("nano-vllm-runner")

hf_volume = modal.Volume.from_name("nano-vllm-hf-cache", create_if_missing=True)
trace_volume = modal.Volume.from_name(
    "nano-vllm-profiler-traces", create_if_missing=True
)
sharegpt_volume = modal.Volume.from_name("nano-vllm-sharegpt", create_if_missing=True)
arc_volume = modal.Volume.from_name("nano-vllm-arc", create_if_missing=True)
TRACE_DIR = Path("/traces")
ARC_DIR = Path("/arc")
ARC_CACHE_DIR = ARC_DIR / "hf_datasets"

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
        "datasets>=2.20.0",
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
        ignore=[
            ".git",
            "__pycache__",
            ".venv",
            "*.pyc",
            # Local ablation scripts write logs here while other `modal run` syncs the tree;
            # tracking it causes "modified during build process" race errors.
            "modal_throughput_ablations",
        ],
    )
)


SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered"
    "/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)
SHAREGPT_DIR = "/sharegpt"
SHAREGPT_PATH = "/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"


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


def _download_sharegpt() -> str:
    import urllib.request

    if not __import__("os").path.exists(SHAREGPT_PATH):
        print(f"Downloading ShareGPT dataset to {SHAREGPT_PATH} ...")
        urllib.request.urlretrieve(SHAREGPT_URL, SHAREGPT_PATH)
        print("ShareGPT download complete.")
    else:
        print("ShareGPT dataset already present, skipping download.")
    return SHAREGPT_PATH


def _download_arc_easy(cache_dir: Path = ARC_CACHE_DIR) -> str:
    import os
    from datasets import load_dataset

    os.makedirs(cache_dir, exist_ok=True)
    marker = cache_dir / ".arc_easy_test_ready"
    if not marker.exists():
        print(f"Downloading ARC-Easy dataset to cache dir {cache_dir} ...")
        load_dataset(
            "allenai/ai2_arc", "ARC-Easy", split="test", cache_dir=str(cache_dir)
        )
        marker.touch()
        print("ARC-Easy dataset download complete.")
    else:
        print("ARC-Easy dataset already present in cache, skipping download.")
    return str(cache_dir)


@app.function(
    image=image,
    gpu="H200:1",
    timeout=7200,
    volumes={
        "/root/huggingface": hf_volume,
        str(TRACE_DIR): trace_volume,
        str(SHAREGPT_DIR): sharegpt_volume,
        str(ARC_DIR): arc_volume,
    },
)
def run_target(
    target: str,
    main_model: str = "",
    main_revision: str = "",
    spec_model: str = "",
    spec_revision: str = "",
    spec_mode: str = "none",
    spec_length: int = 1,
    # MEDUSA-specific
    medusa_choices: str = "",
    medusa_num_heads: int = 4,
    medusa_num_layers: int = 1,
    profile: bool = False,
    profile_label: str = "",
    profile_record_shapes: bool = True,
    profile_memory: bool = False,
    profile_cuda: bool = False,
    # profile_with_stack: bool = False,
    enforce_eager: bool = False,
    data_capture: bool = False,
    trace_throughput: bool = False,
    bench_num_seqs: int = 64,
    bench_max_input_len: int = 1024,
    bench_max_output_len: int = 1024,
    bench_seed: int = 0,
    bench_temperature: float = 0.0,
    bench_warmup_seqs: int = 32,
    bench_max_batch_size: int = 512,
    bench_main_max_model_len: int = 4096,
    bench_main_gpu_memory_utilization: float = 0.8,
    bench_spec_max_model_len: int = 4096,
    bench_spec_gpu_memory_utilization: float = 0.5,
    example_seed: int = 0,
    example_temperature: float = 0.0,
    example_max_tokens: int = 256,
    example_prompt: str = "write me a longform poem about Pittsburgh. About a page. ",
    example_warmup_base_tokens: int = 8,
    example_main_max_model_len: int = 4096,
    example_main_gpu_memory_utilization: float = 0.8,
    example_spec_max_model_len: int = 4096,
    example_spec_gpu_memory_utilization: float = 0.5,
    arc_num_examples: int = 200,
    arc_warmup_examples: int = 16,
    arc_seed: int = 0,
    arc_temperature: float = 1e-9,
    arc_main_max_model_len: int = 4096,
    arc_main_gpu_memory_utilization: float = 0.8,
    arc_spec_max_model_len: int = 4096,
    arc_spec_gpu_memory_utilization: float = 0.5,
) -> None:
    import os
    import runpy
    import sys
    import torch

    if target not in {"bench", "example", "arc"}:
        raise ValueError(
            f"target must be one of ['bench', 'example', 'arc'], got {target!r}"
        )
    spec_mode_norm = spec_mode.lower()
    if spec_mode_norm not in {"none", "naive", "medusa"}:
        raise ValueError(
            f"spec_mode must be one of ['none', 'naive', 'medusa'], got {spec_mode!r}"
        )
    if spec_length < 0:
        raise ValueError(f"spec_length must be >= 0, got {spec_length}")
    if spec_length < 1 and spec_mode_norm in {"naive", "medusa"}:
        raise ValueError(
            f"spec_length must be >= 1 when spec_mode is naive or medusa, got {spec_length}"
        )
    if spec_mode_norm == "medusa" and not spec_model:
        raise ValueError("--spec-model is required when --spec-mode medusa")
    # if profile and profile_with_stack and not enforce_eager:
    #     print(
    #         "Profiler warning: disabling --profile-with-stack in CUDA graph mode "
    #         "(known to stall during trace finalization). Use --enforce-eager "
    #         "if you need Python stacks."
    #     )
    #     profile_with_stack = False

    print("Target: ", target)
    print(
        f"Spec: mode={spec_mode_norm} length={spec_length if spec_mode_norm == 'naive' else '-'}"
    )
    print(f"Profiler: enabled={profile}")
    print(f"Profiler: cuda_events={profile_cuda}")
    print(f"Enforce eager: {enforce_eager}")
    throughput_capture_enabled = data_capture or trace_throughput
    throughput_tmp_path = "/tmp/data.csv"
    print(f"Throughput trace: enabled={throughput_capture_enabled}")
    os.environ["CAPTURE_THROUGHPUT_TRACE"] = "1" if throughput_capture_enabled else "0"
    if throughput_capture_enabled:
        os.environ["THROUGHPUT_TRACE_PATH"] = throughput_tmp_path

    main_repo = main_model or "Qwen/Qwen3-8B"
    os.environ["MAIN_MODEL_PATH"] = _download_model(main_repo, main_revision)

    # Pull the speculator / MEDUSA checkpoint only when needed.
    if spec_mode_norm == "naive":
        spec_repo = spec_model or "Qwen/Qwen3-0.6B"
        os.environ["SPEC_MODEL_PATH"] = _download_model(spec_repo, spec_revision)
    elif spec_mode_norm == "medusa":
        os.environ["MEDUSA_MODEL_PATH"] = _download_model(spec_model, spec_revision)
        if medusa_choices:
            os.environ["MEDUSA_CHOICES"] = medusa_choices
        os.environ["MEDUSA_NUM_HEADS"] = str(medusa_num_heads)
        os.environ["MEDUSA_NUM_LAYERS"] = str(medusa_num_layers)

    # propagate spec config to the target script via env
    os.environ["SPEC_MODE"] = spec_mode_norm
    os.environ["SPEC_LENGTH"] = str(spec_length)
    os.environ["ENFORCE_EAGER"] = "1" if enforce_eager else "0"
    if target == "bench":
        os.environ["BENCH_NUM_SEQS"] = str(bench_num_seqs)
        os.environ["BENCH_MAX_INPUT_LEN"] = str(bench_max_input_len)
        os.environ["BENCH_MAX_OUTPUT_LEN"] = str(bench_max_output_len)
        os.environ["BENCH_SEED"] = str(bench_seed)
        os.environ["BENCH_TEMPERATURE"] = str(bench_temperature)
        os.environ["BENCH_WARMUP_SEQS"] = str(bench_warmup_seqs)
        os.environ["BENCH_MAX_BATCH_SIZE"] = str(bench_max_batch_size)
        os.environ["BENCH_MAIN_MAX_MODEL_LEN"] = str(bench_main_max_model_len)
        os.environ["BENCH_MAIN_GPU_MEMORY_UTILIZATION"] = str(
            bench_main_gpu_memory_utilization
        )
        os.environ["BENCH_SPEC_MAX_MODEL_LEN"] = str(bench_spec_max_model_len)
        os.environ["BENCH_SPEC_GPU_MEMORY_UTILIZATION"] = str(
            bench_spec_gpu_memory_utilization
        )
        os.environ["SHAREGPT_PATH"] = _download_sharegpt()
    elif target == "example":
        os.environ["EXAMPLE_SEED"] = str(example_seed)
        os.environ["EXAMPLE_TEMPERATURE"] = str(example_temperature)
        os.environ["EXAMPLE_MAX_TOKENS"] = str(example_max_tokens)
        os.environ["EXAMPLE_PROMPT"] = example_prompt
        os.environ["EXAMPLE_WARMUP_BASE_TOKENS"] = str(example_warmup_base_tokens)
        os.environ["EXAMPLE_MAIN_MAX_MODEL_LEN"] = str(example_main_max_model_len)
        os.environ["EXAMPLE_MAIN_GPU_MEMORY_UTILIZATION"] = str(
            example_main_gpu_memory_utilization
        )
        os.environ["EXAMPLE_SPEC_MAX_MODEL_LEN"] = str(example_spec_max_model_len)
        os.environ["EXAMPLE_SPEC_GPU_MEMORY_UTILIZATION"] = str(
            example_spec_gpu_memory_utilization
        )
    else:
        os.environ["ARC_NUM_EXAMPLES"] = str(arc_num_examples)
        os.environ["ARC_WARMUP_EXAMPLES"] = str(arc_warmup_examples)
        os.environ["ARC_SEED"] = str(arc_seed)
        os.environ["ARC_TEMPERATURE"] = str(arc_temperature)
        os.environ["ARC_MAIN_MAX_MODEL_LEN"] = str(arc_main_max_model_len)
        os.environ["ARC_MAIN_GPU_MEMORY_UTILIZATION"] = str(
            arc_main_gpu_memory_utilization
        )
        os.environ["ARC_SPEC_MAX_MODEL_LEN"] = str(arc_spec_max_model_len)
        os.environ["ARC_SPEC_GPU_MEMORY_UTILIZATION"] = str(
            arc_spec_gpu_memory_utilization
        )
        os.environ["ARC_CACHE_DIR"] = _download_arc_easy()

    workspace_dir = "/workspace"
    if workspace_dir not in sys.path:
        sys.path.insert(0, workspace_dir)

    script_name = "bench_arc.py" if target == "arc" else f"{target}.py"
    script_path = os.path.join(workspace_dir, script_name)
    if not os.path.isfile(script_path):
        script_path = f"/{script_name}"

    profile_tag = (profile_label.strip() or target).replace("/", "-")
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_id = uuid4().hex[:8]
    output_dir = None
    if profile or throughput_capture_enabled:
        output_dir = TRACE_DIR / f"{profile_tag}-{timestamp}-{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

    wrote_artifact = False
    if not profile:
        runpy.run_path(script_path, run_name="__main__")
    else:
        trace_path = output_dir / "trace.pt.trace.json"
        activities = [torch.profiler.ProfilerActivity.CPU]
        if profile_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(
            activities=activities,
            record_shapes=profile_record_shapes,
            profile_memory=profile_memory,
            # with_stack=profile_with_stack,
            with_stack=False,
        ) as prof:
            runpy.run_path(script_path, run_name="__main__")

        torch.cuda.synchronize()
        print(f"Exporting profiler trace to {trace_path}...")
        # prof.export_chrome_trace(str(trace_path))
        prof.export_chrome_trace("/tmp/run.trace.pt.trace.json")
        print("Finished exporting, copying to modal volume...")
        _ = shutil.copy("/tmp/run.trace.pt.trace.json", str(trace_path))
        print(f"Profiler trace saved to modal volume: {trace_path}")
        wrote_artifact = True

    if throughput_capture_enabled:
        data_path = output_dir / "data.csv"
        if not os.path.isfile(throughput_tmp_path):
            raise FileNotFoundError(
                f"Throughput trace requested but no file found at {throughput_tmp_path}"
            )
        _ = shutil.copy(throughput_tmp_path, str(data_path))
        print(f"Throughput data saved to modal volume: {data_path}")
        wrote_artifact = True

    if wrote_artifact:
        print("Committing artifacts to modal volume...")
        trace_volume.commit()
        print(f"Artifacts committed under {output_dir}")


@app.local_entrypoint()
def main(
    target: str = "bench",
    main_model: str = "Qwen/Qwen3-8B",
    main_revision: str = "",
    spec_model: str = "",
    spec_revision: str = "",
    spec_mode: str = "none",
    spec_length: int = 1,
    # MEDUSA-specific
    medusa_choices: str = "",
    medusa_num_heads: int = 4,
    medusa_num_layers: int = 1,
    profile: bool = False,
    profile_label: str = "",
    profile_record_shapes: bool = True,
    profile_memory: bool = False,
    profile_cuda: bool = False,
    # profile_with_stack: bool = False,
    enforce_eager: bool = False,
    data_capture: bool = False,
    trace_throughput: bool = False,
    bench_num_seqs: int = 64,
    bench_max_input_len: int = 1024,
    bench_max_output_len: int = 1024,
    bench_seed: int = 0,
    bench_temperature: float = 0.0,
    bench_warmup_seqs: int = 32,
    bench_max_batch_size: int = 512,
    bench_main_max_model_len: int = 4096,
    bench_main_gpu_memory_utilization: float = 0.8,
    bench_spec_max_model_len: int = 4096,
    bench_spec_gpu_memory_utilization: float = 0.5,
    example_seed: int = 0,
    example_temperature: float = 0.0,
    example_max_tokens: int = 256,
    example_prompt: str = "write me a longform poem about Pittsburgh. About a page. ",
    example_warmup_base_tokens: int = 8,
    example_main_max_model_len: int = 4096,
    example_main_gpu_memory_utilization: float = 0.8,
    example_spec_max_model_len: int = 4096,
    example_spec_gpu_memory_utilization: float = 0.5,
    arc_num_examples: int = 200,
    arc_warmup_examples: int = 16,
    arc_seed: int = 0,
    arc_temperature: float = 1e-9,
    arc_main_max_model_len: int = 4096,
    arc_main_gpu_memory_utilization: float = 0.8,
    arc_spec_max_model_len: int = 4096,
    arc_spec_gpu_memory_utilization: float = 0.5,
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
            medusa_choices,
            medusa_num_heads,
            medusa_num_layers,
            profile,
            profile_label,
            profile_record_shapes,
            profile_memory,
            profile_cuda,
            # profile_with_stack,
            enforce_eager,
            data_capture,
            trace_throughput,
            bench_num_seqs,
            bench_max_input_len,
            bench_max_output_len,
            bench_seed,
            bench_temperature,
            bench_warmup_seqs,
            bench_max_batch_size,
            bench_main_max_model_len,
            bench_main_gpu_memory_utilization,
            bench_spec_max_model_len,
            bench_spec_gpu_memory_utilization,
            example_seed,
            example_temperature,
            example_max_tokens,
            example_prompt,
            example_warmup_base_tokens,
            example_main_max_model_len,
            example_main_gpu_memory_utilization,
            example_spec_max_model_len,
            example_spec_gpu_memory_utilization,
            arc_num_examples,
            arc_warmup_examples,
            arc_seed,
            arc_temperature,
            arc_main_max_model_len,
            arc_main_gpu_memory_utilization,
            arc_spec_max_model_len,
            arc_spec_gpu_memory_utilization,
        )
    except Exception as exc:  # pragma: no cover
        print(f"Modal execution failed: {exc}")
        sys.exit(1)
