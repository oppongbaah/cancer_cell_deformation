"""Keep a log of every run of a script, however it was started.

``start_run_log("features/nnunet1000_loocv", "extract_features_all")`` copies everything the process
writes to stdout and stderr -- its own prints, tracebacks, and the output of libraries and child
processes (nnU-Net's workers, C extensions) -- into ``logs/<category>/<name>_<YYYYmmdd-HHMMSS>.log``,
while still showing it in the terminal. The copy is made at the file-descriptor level by a ``tee``
process in its own session, so Ctrl+C stops the script but not the logging, and the traceback of an
interrupted or crashed run still lands in the log.

The log opens with the command line, start time, host and git commit, and closes with the end time.

Only the runs that compute results call it (feature extraction, model training, tuning, the classification
benchmark, the deformation analysis); figure/table redraws and quick summaries do not, so logs/ holds the
runs worth keeping.
"""
from __future__ import annotations

import atexit
import os
import shlex
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_active: Path | None = None


def _git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return f"{sha}{' (uncommitted changes)' if dirty else ''}" if sha else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def start_run_log(category: str, name: str) -> Path:
    """Start teeing this process's stdout/stderr into logs/<category>/<name>_<timestamp>.log; return its path.
    Calling it twice in one process keeps the first log."""
    global _active
    if _active is not None:
        return _active
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log = ROOT / "logs" / category / f"{name}_{stamp}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    sys.stdout.flush()
    sys.stderr.flush()
    tee = subprocess.Popen(["tee", "-a", str(log)], stdin=subprocess.PIPE, start_new_session=True)
    os.dup2(tee.stdin.fileno(), 1)
    os.dup2(tee.stdin.fileno(), 2)
    tee.stdin.close()
    # a pipe, not a terminal, is now behind fd 1/2: flush every line so the log and screen stay current
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print(f"[run log] {log.relative_to(ROOT)}")
    print(f"[run log] command: {shlex.join([Path(sys.executable).name, *sys.argv])}")
    print(f"[run log] started {datetime.now():%Y-%m-%d %H:%M:%S} on {socket.gethostname()}, "
          f"git {_git_commit()}, cwd {os.getcwd()}")

    def _close() -> None:
        print(f"[run log] ended {datetime.now():%Y-%m-%d %H:%M:%S}")
        sys.stdout.flush()
        sys.stderr.flush()
        os.close(1)
        os.close(2)
        try:
            tee.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    atexit.register(_close)
    _active = log
    return log
