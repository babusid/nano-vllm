#!/usr/bin/env python3
"""
Launch Modal throughput ablations as separate ``modal run`` processes (each job uses
its own Modal GPU allocation).

Each job writes stdout/stderr to ``<state-dir>/logs/<job-id>.log`` and updates
``<state-dir>/dashboard.json``. Poll with ``python scripts/run_modal_throughput_ablations.py --status``
(optional ``--state-dir``).

Run from the repository root::

    # Blocking: wait for every Modal job (default).
    python scripts/run_modal_throughput_ablations.py

    # Background: launcher exits immediately; one child Python process runs the sweep.
    python scripts/run_modal_throughput_ablations.py --detach

    # Inspect progress written by the launcher
    python scripts/run_modal_throughput_ablations.py --status

Detached runs append child stdout/err to ``<state-dir>/orchestrator.log``.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_FILE = Path(__file__).resolve()
# Keep this path in sync with ``ignore=`` in ``run_modal.py`` ``add_local_dir`` so logs
# written during parallel ``modal run`` jobs are not part of the image sync hash.
DEFAULT_STATE_DIR = REPO_ROOT / "modal_throughput_ablations"

MAIN_MODEL = "lmsys/vicuna-33b-v1.3"
SPEC_MODEL = "Jiayi-Pan/Tiny-Vicuna-1B"
MEDUSA_MODEL = "FasterDecoding/medusa-vicuna-33b-v1.3"
MEDUSA_CONFIG_DIR = REPO_ROOT / "medusa_tree_configs"
BATCH_SIZES = (2, 4, 16, 32, 64, 128)
# SPEC_LENGTHS_NAIVE = (1, 2, 3, 4, 5)
SPEC_LENGTHS_NAIVE = (32,)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(fragment: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in fragment)


def _build_none_jobs() -> list[tuple[str, list[str]]]:
    jobs: list[tuple[str, list[str]]] = []
    for bs in BATCH_SIZES:
        label = f"none-bs_{bs}-n128"
        cmd = [
            "modal",
            "run",
            "--detach",
            "run_modal.py",
            "--target",
            "bench",
            "--spec-mode",
            "none",
            "--spec-length",
            "0",
            "--main-model",
            MAIN_MODEL,
            "--bench-num-seqs",
            "128",
            "--bench-max-batch-size",
            str(bs),  # batch size
            "--bench-warmup-seqs",
            "8",
            "--trace-throughput",
            "--profile-label",
            label,
        ]
        jobs.append((_safe_id(label), cmd))
    return jobs


def _build_naive_jobs() -> list[tuple[str, list[str]]]:
    jobs: list[tuple[str, list[str]]] = []
    for sl in SPEC_LENGTHS_NAIVE:
        for bs in BATCH_SIZES:
            label = f"naive-speclen_{sl}-bs_{bs}-n128"
            cmd = [
                "modal",
                "run",
                "--detach",
                "run_modal.py",
                "--target",
                "bench",
                "--spec-mode",
                "naive",
                "--main-model",
                MAIN_MODEL,
                "--spec-model",
                SPEC_MODEL,
                "--bench-num-seqs",
                "128",
                "--bench-max-batch-size",
                str(bs),  # batch size
                "--bench-warmup-seqs",
                "8",
                "--spec-length",
                str(sl),  # spec length
                "--trace-throughput",
                "--profile-label",
                label,
            ]
            jobs.append((_safe_id(label), cmd))
    return jobs


def _build_medusa_jobs() -> list[tuple[str, list[str]]]:
    jobs: list[tuple[str, list[str]]] = []
    if not MEDUSA_CONFIG_DIR.is_dir():
        raise FileNotFoundError(f"Missing medusa config dir: {MEDUSA_CONFIG_DIR}")

    config_paths = sorted(MEDUSA_CONFIG_DIR.glob("*.json"))
    if not config_paths:
        raise ValueError(f"No JSON configs found in {MEDUSA_CONFIG_DIR}")

    for config_path in config_paths:
        m = re.match(r"^tree_(\d+)_(\d+)\.json$", config_path.name)
        if m is None:
            continue
        c1, c2 = m.group(1), m.group(2)
        medusa_choices = config_path.read_text(encoding="utf-8").strip()
        for bs in BATCH_SIZES:
            label = f"medusa_bs{bs}_c1_{c1}_c2_{c2}_n128"
            cmd = [
                "modal",
                "run",
                "run_modal.py",
                "--target",
                "bench",
                "--spec-mode",
                "medusa",
                "--main-model",
                MAIN_MODEL,
                "--spec-model",
                MEDUSA_MODEL,
                "--medusa-num-heads",
                "2",
                "--medusa-choices",
                medusa_choices,
                "--bench-num-seqs",
                "128",
                "--bench-max-batch-size",
                str(bs),
                "--bench-warmup-seqs",
                "8",
                "--trace-throughput",
                "--profile-label",
                label,
            ]
            jobs.append((_safe_id(label), cmd))
    return jobs


def _apply_job_filters(
    jobs: list[tuple[str, list[str]]], patterns: list[str] | None
) -> list[tuple[str, list[str]]]:
    if not patterns:
        return jobs

    kept: list[tuple[str, list[str]]] = []
    for jid, cmd in jobs:
        cmd_str = shlex_like(cmd)
        if any(
            fnmatch.fnmatch(jid, p) or fnmatch.fnmatch(cmd_str, p) for p in patterns
        ):
            kept.append((jid, cmd))
    return kept


@dataclass
class JobRecord:
    job_id: str
    command: list[str]
    status: str  # pending | running | completed | failed
    pid: int | None
    started_at: str | None
    finished_at: str | None
    exit_code: int | None
    log_rel: str | None


def _write_dashboard(
    lock: threading.Lock,
    state_dir: Path,
    jobs: list[JobRecord],
    meta: dict,
) -> None:
    payload = {
        "updated": _utc_now_iso(),
        "meta": meta,
        "summary": _summarize(jobs),
        "jobs": [asdict(j) for j in jobs],
    }
    dashboard = state_dir / "dashboard.json"
    tmp = dashboard.with_suffix(".json.tmp")
    with lock:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(dashboard)


def _summarize(jobs: list[JobRecord]) -> dict[str, int]:
    out = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
    for j in jobs:
        out[j.status] = out.get(j.status, 0) + 1
    return out


def _resolve_modal_exe(modal_bin: str) -> str:
    p = Path(modal_bin)
    if p.is_file():
        return str(p.resolve())
    w = shutil.which(modal_bin)
    return w or modal_bin


def _modal_cli_available(modal_bin: str) -> bool:
    p = Path(modal_bin)
    if p.is_file():
        return True
    return shutil.which(modal_bin) is not None


def _env_for_modal_subprocess() -> dict[str, str]:
    """Avoid Windows cp1252 / 'charmap' crashes when Modal's CLI prints unicode (✓, box chars)."""
    env = os.environ.copy()
    # Python-based CLIs (including ``modal``, Rich/Typer): stdout must be UTF-8 when redirected to a file.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Prefer plain ASCII-ish output where supported (fewer glyphs in Panels/borders).
    env.setdefault("NO_COLOR", "1")
    env.setdefault("TERM", "dumb")
    return env


def _spawn_detached_orchestrator(state_dir: Path) -> None:
    argv = [sys.executable, str(SCRIPT_FILE)]
    argv.extend(a for a in sys.argv[1:] if a != "--detach")
    state_dir.mkdir(parents=True, exist_ok=True)
    log_path = state_dir / "orchestrator.log"
    lf = open(log_path, "a", encoding="utf-8", newline="\n")
    kwargs: dict = {
        "cwd": REPO_ROOT,
        "stdout": lf,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "env": _env_for_modal_subprocess(),
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # noqa: S404
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(argv, **kwargs)
    lf.close()
    print(f"Detached orchestrator appending stdout/stderr to:\n  {log_path}")
    print(f"Job dashboard (JSON):\n  {state_dir / 'dashboard.json'}")
    try:
        rel = SCRIPT_FILE.relative_to(REPO_ROOT)
    except ValueError:
        rel = SCRIPT_FILE
    print(f"Status table:\n  python {rel} --status")


def run_orchestration(
    state_dir: Path,
    *,
    modal_bin: str,
    max_parallel: int,
    dry_run: bool,
    only: str | None,
    job_filter: list[str] | None,
) -> int:
    if not _modal_cli_available(modal_bin):
        print(
            f"fatal: Modal CLI `{modal_bin}` not found (PATH or invalid file).",
            file=sys.stderr,
        )
        return 126

    run_modal = REPO_ROOT / "run_modal.py"
    if not run_modal.is_file():
        print(f"fatal: expected {run_modal}", file=sys.stderr)
        return 2

    jobs: list[tuple[str, list[str]]] = []
    if only == "none" or only is None:
        jobs.extend(_build_none_jobs())
    if only == "naive" or only is None:
        jobs.extend(_build_naive_jobs())
    if only == "medusa" or only is None:
        jobs.extend(_build_medusa_jobs())

    jobs = _apply_job_filters(jobs, job_filter)

    logs_dir = state_dir / "logs"
    state_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    records: dict[str, JobRecord] = {
        jid: JobRecord(
            job_id=jid,
            command=cmd,
            status="pending",
            pid=None,
            started_at=None,
            finished_at=None,
            exit_code=None,
            log_rel=str((logs_dir / f"{jid}.log").relative_to(state_dir)),
        )
        for jid, cmd in jobs
    }
    ordered = list(records[jid] for jid, _ in jobs)

    modal_exe = _resolve_modal_exe(modal_bin)
    meta = {
        "repo_root": str(REPO_ROOT),
        "max_parallel": max_parallel,
        "dry_run": dry_run,
        "total_jobs": len(jobs),
        "modal_bin": modal_bin,
        "modal_exe": modal_exe,
        "job_filter": job_filter or [],
    }

    print(f"State dir: {state_dir}")
    print(f"Jobs:      {len(jobs)}  (dashboard: {state_dir / 'dashboard.json'})")
    print()

    if dry_run:
        for jid, cmd in jobs:
            print(shlex_like(cmd))
        print()
        print("`--dry-run` set; nothing launched.")
        return 0

    _write_dashboard(lock, state_dir, ordered, meta)

    # Bounded parallel execution of local `modal run` wrappers.
    from collections import deque

    queue_cmd: deque[tuple[str, JobRecord]] = deque(
        (jid, records[jid]) for jid, _ in jobs
    )
    running: dict[int, tuple[subprocess.Popen, JobRecord]] = {}
    failures = 0

    def reap_finished() -> None:
        nonlocal failures
        for pid in list(running.keys()):
            proc, record = running[pid]
            code = proc.poll()
            if code is None:
                continue
            del running[pid]
            record.finished_at = _utc_now_iso()
            record.exit_code = int(code)
            if code == 0:
                record.status = "completed"
            else:
                record.status = "failed"
                failures += 1
            _write_dashboard(lock, state_dir, ordered, meta)

    while queue_cmd or running:
        while queue_cmd and len(running) < max_parallel:
            jid, record = queue_cmd.popleft()
            log_path = logs_dir / f"{jid}.log"
            record.status = "running"
            record.started_at = _utc_now_iso()
            record.log_rel = str(log_path.relative_to(state_dir))
            cmd = [modal_exe] + record.command[1:]
            lf = open(log_path, "w", encoding="utf-8", newline="\n", errors="replace")
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=lf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=_env_for_modal_subprocess(),
            )
            lf.close()
            record.pid = proc.pid
            running[proc.pid] = (proc, record)
            _write_dashboard(lock, state_dir, ordered, meta)

        reap_finished()
        if queue_cmd and len(running) >= max_parallel:
            time.sleep(0.4)
        elif running:
            time.sleep(0.4)
        else:
            break

    assert not running
    _write_dashboard(lock, state_dir, ordered, meta)

    if failures:
        print(f"Done with {failures} failed job(s). See logs under {logs_dir}")
        return 1
    print(f"All {len(jobs)} jobs completed successfully.")
    return 0


def shlex_like(argv: list[str]) -> str:
    try:
        import shlex

        return shlex.join(argv)
    except Exception:
        return " ".join(argv)


def cmd_status(state_dir: Path) -> int:
    dashboard = state_dir / "dashboard.json"
    if not dashboard.is_file():
        print(f"No dashboard at {dashboard}", file=sys.stderr)
        return 1
    data = json.loads(dashboard.read_text(encoding="utf-8"))
    jobs = data.get("jobs", [])
    print(f"Updated: {data.get('updated')}")
    print(f"Summary: {data.get('summary')}")
    print()
    w_id = max(len("job_id"), max((len(j["job_id"]) for j in jobs), default=0))
    w_st = 9
    print(f"{'job_id':<{w_id}}  {'status':<{w_st}}  exit  log")
    print("-" * (w_id + w_st + 24))
    for j in jobs:
        ec = "" if j.get("exit_code") is None else str(j["exit_code"])
        print(
            f"{j['job_id']:<{w_id}}  {j['status']:<{w_st}}  {ec:>4}  {j.get('log_rel', '')}"
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--state-dir",
        type=Path,
        default=DEFAULT_STATE_DIR,
        help=(
            f"Directory for dashboard.json and logs (default: {DEFAULT_STATE_DIR}). "
            "If inside the repo, add that dirname to ``run_modal.py`` ``add_local_dir(..., ignore=[...])`` "
            "or parallel ``modal run`` snapshots will race ablation logs."
        ),
    )
    p.add_argument(
        "--modal-bin",
        default="modal",
        help="Modal CLI executable name or path (default: modal)",
    )
    p.add_argument(
        "--max-parallel",
        type=int,
        default=64,
        metavar="N",
        help="Max concurrent local `modal run` processes (default: 64)",
    )
    p.add_argument(
        "--only",
        choices=("none", "naive", "medusa"),
        default=None,
        help="Run only one sweep family",
    )
    p.add_argument(
        "--job-filter",
        action="append",
        default=None,
        metavar="GLOB",
        help=(
            "Glob filter over job_id or full command; repeatable. "
            "Examples: --job-filter 'medusa_*' --job-filter '*bs128*'"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running",
    )
    p.add_argument(
        "--detach",
        action="store_true",
        help="Launch the sweep in a background subprocess (returns immediately)",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Print dashboard table from --state-dir and exit",
    )
    args = p.parse_args()
    state_dir = args.state_dir.resolve()
    if args.status:
        return cmd_status(state_dir)
    if args.detach and not args.dry_run:
        _spawn_detached_orchestrator(state_dir)
        return 0
    return run_orchestration(
        state_dir,
        modal_bin=args.modal_bin,
        max_parallel=max(1, args.max_parallel),
        dry_run=args.dry_run,
        only=args.only,
        job_filter=args.job_filter,
    )


if __name__ == "__main__":
    raise SystemExit(main())
