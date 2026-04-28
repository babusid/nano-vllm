#!/usr/bin/env python3
"""
Download ``data.csv`` throughput artifacts from the Modal profiler trace volume.

Remote layout (from ``run_modal.py``)::

    <profile_label>-<YYYYMMDD>-<HHMMSS>-<8hex>/data.csv

Example::

    modal volume get nano-vllm-profiler-traces \\
      naive-speclen_1-bs_2-n128-20260428-054906-1fa3ede9/data.csv \\
      ./out/data.csv

This script lists the volume root, parses those directory names, and either
downloads the **latest** run per ``profile_label`` (default) or **every** run.

Run from the repo root::

    python scripts/download_modal_profiler_csvs.py
    python scripts/download_modal_profiler_csvs.py --all-runs --output-dir ./modal_throughput_ablations/csvs
    python scripts/download_modal_profiler_csvs.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOLUME = "nano-vllm-profiler-traces"
DEFAULT_OUT = REPO_ROOT / "modal_throughput_ablations" / "csvs"

# Directory name from run_modal: f"{profile_tag}-{timestamp}-{run_id}" with timestamp %Y%m%d-%H%M%S
RE_TRACE_DIR = re.compile(
    r"^(?P<tag>.+)-(?P<ymd>\d{8})-(?P<hms>\d{6})-(?P<rid>[0-9a-f]{8})$"
)


def _env_for_modal_cli() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("NO_COLOR", "1")
    env.setdefault("TERM", "dumb")
    return env


def _modal_volume_ls_json(volume: str) -> list[dict]:
    r = subprocess.run(
        ["modal", "volume", "ls", volume, "--json"],
        capture_output=True,
        text=True,
        check=False,
        env=_env_for_modal_cli(),
        cwd=str(REPO_ROOT),
    )
    if r.returncode != 0:
        print(r.stderr or r.stdout, file=sys.stderr)
        raise SystemExit(f"`modal volume ls` failed with exit {r.returncode}")
    return json.loads(r.stdout)


def _sort_key(row: tuple[str, str, str, str, str]) -> tuple[str, str, str]:
    """dirname, tag, ymd, hms, rid -> sort by time then rid."""
    _dirname, _tag, ymd, hms, rid = row
    return (ymd, hms, rid)


def _pick_runs(
    entries: list[dict],
    *,
    all_runs: bool,
) -> list[tuple[str, str]]:
    """Return list of (remote_dirname, local_basename.csv) to fetch."""
    rows: list[tuple[str, str, str, str, str]] = []
    for e in entries:
        if e.get("Type") != "dir":
            continue
        name = e.get("Filename") or ""
        m = RE_TRACE_DIR.match(name)
        if not m:
            continue
        tag = m.group("tag")
        ymd, hms, rid = m.group("ymd"), m.group("hms"), m.group("rid")
        rows.append((name, tag, ymd, hms, rid))

    if not rows:
        return []

    out: list[tuple[str, str]] = []
    if all_runs:
        for name, tag, ymd, hms, rid in sorted(rows, key=_sort_key):
            local_name = f"{name}.data.csv"
            out.append((name, local_name))
        return out

    by_tag: dict[str, list[tuple[str, str, str, str, str]]] = defaultdict(list)
    for row in rows:
        by_tag[row[1]].append(row)

    for tag in sorted(by_tag.keys()):
        group = by_tag[tag]
        best = max(group, key=_sort_key)
        dirname = best[0]
        # Same convention as manual one-off: stable name from profile_label only.
        safe = tag.replace("/", "-")
        out.append((dirname, f"{safe}.data.csv"))
    return out


def _run_get(volume: str, remote_dir: str, local_path: Path, *, dry_run: bool) -> bool:
    remote = f"{remote_dir}/data.csv"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["modal", "volume", "get", volume, remote, str(local_path)]
    if dry_run:
        print(shlex.join(cmd))
        return True
    r = subprocess.run(
        cmd,
        env=_env_for_modal_cli(),
        cwd=str(REPO_ROOT),
    )
    return r.returncode == 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--volume",
        default=DEFAULT_VOLUME,
        help=f"Modal volume name (default: {DEFAULT_VOLUME})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Local directory for downloaded CSVs (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--all-runs",
        action="store_true",
        help="Download every run's data.csv; filenames include full remote dirname",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print modal volume get commands without running",
    )
    args = p.parse_args()
    out_dir = args.output_dir.resolve()

    print(f"Listing volume {args.volume!r} ...")
    raw = _modal_volume_ls_json(args.volume)
    planned = _pick_runs(raw, all_runs=args.all_runs)
    if not planned:
        print("No matching trace directories found (expected *-YYYYMMDD-HHMMSS-<8hex>/).")
        return 1

    mode = "all runs" if args.all_runs else "latest per profile_label"
    print(f"Planned downloads ({mode}): {len(planned)}")
    print(f"Output directory: {out_dir}")
    print()

    ok = 0
    fail = 0
    for remote_dir, local_name in planned:
        dest = out_dir / local_name
        if not _run_get(args.volume, remote_dir, dest, dry_run=args.dry_run):
            print(f"FAILED: {remote_dir}", file=sys.stderr)
            fail += 1
        else:
            ok += 1

    if args.dry_run:
        print()
        print("`--dry-run`: no files downloaded.")
        return 0

    print()
    print(f"Done. Succeeded: {ok}, failed: {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
