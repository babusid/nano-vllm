"""One-shot dump of the EAGLE-3 head checkpoint's key/shape inventory.

Run on Modal where the head was downloaded:

    modal run run_modal.py --target inspect-eagle \
        --eagle-head "AngelSlim/Qwen3-32B_eagle3"

(or locally if the path is set):

    EAGLE_HEAD_PATH=~/huggingface/Qwen3-32B_eagle3 python inspect_eagle.py
"""

import json
import os
from glob import glob

from safetensors import safe_open


def main() -> None:
    path = os.path.expanduser(
        os.environ.get("EAGLE_HEAD_PATH", "~/huggingface/Qwen3-32B_eagle3/")
    )
    print(f"Inspecting: {path}")
    cfg = os.path.join(path, "config.json")
    if os.path.isfile(cfg):
        with open(cfg) as f:
            print("config.json:")
            print(json.dumps(json.load(f), indent=2))

    print(f"\nDirectory contents (top-level):")
    for entry in sorted(os.listdir(path)):
        full = os.path.join(path, entry)
        kind = "dir " if os.path.isdir(full) else "file"
        size = "" if os.path.isdir(full) else f"  ({os.path.getsize(full):,}B)"
        print(f"  [{kind}] {entry}{size}")

    # Recursive in case snapshot_download stashed weights in a subdir.
    safetensors_files = sorted(
        glob(os.path.join(path, "**/*.safetensors"), recursive=True)
    )
    if not safetensors_files:
        bin_files = sorted(
            glob(os.path.join(path, "**/pytorch_model*.bin"), recursive=True)
        )
        if not bin_files:
            raise FileNotFoundError(
                f"No *.safetensors or pytorch_model*.bin files under {path}. "
                "Re-run after deleting the cache dir, or check the directory "
                "listing above."
            )
        print(f"\nFound {len(bin_files)} .bin files; loading via torch")
        import torch

        rows: list[tuple[str, tuple[int, ...], str]] = []
        for fp in bin_files:
            sd = torch.load(fp, map_location="cpu", weights_only=True)
            for k, v in sd.items():
                rows.append((k, tuple(v.shape), str(v.dtype)))
        rows.sort()
        for k, shape, dtype in rows:
            print(f"  {k:60s}  {str(shape):30s}  {dtype}")
        return

    print(f"\nFound {len(safetensors_files)} safetensors file(s)")
    print("weights (key, shape, dtype):")
    rows: list[tuple[str, tuple[int, ...], str]] = []
    for fp in safetensors_files:
        with safe_open(fp, framework="pt") as h:
            for k in h.keys():
                ts = h.get_slice(k)
                rows.append((k, tuple(ts.get_shape()), str(ts.get_dtype())))
    rows.sort()
    for k, shape, dtype in rows:
        print(f"  {k:60s}  {str(shape):30s}  {dtype}")


if __name__ == "__main__":
    main()
