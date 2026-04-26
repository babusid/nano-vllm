"""CUDA memory profiling helpers.

Wraps ``torch.cuda.memory._record_memory_history`` so a target script can
opt in via env vars set by ``run_modal.py``. The resulting snapshot is a
pickle file viewable at https://pytorch.org/memory_viz.
"""

from __future__ import annotations

import os
from functools import wraps
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

import torch

P = ParamSpec("P")
R = TypeVar("R")


def is_enabled() -> bool:
    return os.environ.get("MEMORY_PROFILE", "0") == "1"


@contextmanager
def memory_profile():
    """Record CUDA allocator history for the wrapped block when enabled.

    Honors:
        MEMORY_PROFILE              ("1" to enable; otherwise no-op)
        MEMORY_PROFILE_PATH         output snapshot path (default: ./memory_snapshot.pickle)
        MEMORY_PROFILE_MAX_ENTRIES  alloc-event ring buffer size (default: 100000)
    """
    if not is_enabled():
        yield
        return

    if not torch.cuda.is_available():
        print("[memory_profile] CUDA unavailable, skipping memory profiling")
        yield
        return

    max_entries = int(os.environ.get("MEMORY_PROFILE_MAX_ENTRIES", "100000"))
    snapshot_path = Path(
        os.environ.get("MEMORY_PROFILE_PATH", "memory_snapshot.pickle")
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[memory_profile] recording CUDA memory history "
        f"(max_entries={max_entries}); snapshot -> {snapshot_path}"
    )
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    try:
        yield
    finally:
        try:
            torch.cuda.synchronize()
        except Exception as exc:
            print(f"[memory_profile] synchronize failed before snapshot: {exc}")
        try:
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
            print(f"[memory_profile] snapshot saved: {snapshot_path}")
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)


def memory_profiled(fn: Callable[P, R]) -> Callable[P, R]:
    @wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        with memory_profile():
            return fn(*args, **kwargs)

    return wrapped
