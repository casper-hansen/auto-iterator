"""Unit tests for ``ai ls`` / ``ai show`` status reconciliation.

The key invariant: ``meta.status`` is never trusted. Every row goes
through :func:`auto_iterator.ls.reconcile_status` which looks at the
event tail, the state snapshot, pid liveness, and the heartbeat mtime
in that order. Getting this wrong means operators can't tell a stuck
run from a crashed one — exactly the mode filesystem.md calls out."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_iterator.events import EventLog, RunState  # noqa: E402
from auto_iterator.heartbeat import STUCK_THRESHOLD_SECONDS  # noqa: E402
from auto_iterator.ls import reconcile_status, summarize_run  # noqa: E402
from auto_iterator.meta import write_meta  # noqa: E402
from auto_iterator.run_dir import (  # noqa: E402
    atomic_write_json,
    create_run_dir,
    new_run_id,
    now_iso,
    touch,
)


def _fresh(tmp: Path):
    paths = create_run_dir(tmp, new_run_id())
    state = RunState(run_id=paths.run_id, prompt="p", workspace="/tmp/ws")
    log = EventLog(paths, state)
    return paths, state, log


def test_status_running() -> None:
    """pid alive + fresh heartbeat → running."""
    with tempfile.TemporaryDirectory() as tmp:
        paths, state, log = _fresh(Path(tmp))
        # Start a benign child whose pid we can use as "runner".
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            write_meta(paths, {
                "run_id": paths.run_id,
                "pid": child.pid,
                "status": "running",
                "workspace": "/tmp/ws",
                "started_at": now_iso(),
            })
            touch(paths.heartbeat)
            assert reconcile_status(paths, {
                "pid": child.pid, "status": "running",
            }) == "running"
        finally:
            child.terminate()
            child.wait(timeout=5)
    print("  test_status_running PASS")


def test_status_stuck_when_heartbeat_stale() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths, state, log = _fresh(Path(tmp))
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            write_meta(paths, {
                "run_id": paths.run_id, "pid": child.pid,
                "status": "running", "workspace": "/tmp/ws",
            })
            # Create heartbeat and rewind its mtime past the staleness bar.
            touch(paths.heartbeat)
            stale = time.time() - (STUCK_THRESHOLD_SECONDS + 5)
            os.utime(paths.heartbeat, (stale, stale))
            assert reconcile_status(paths, {
                "pid": child.pid, "status": "running",
            }) == "stuck"
        finally:
            child.terminate()
            child.wait(timeout=5)
    print("  test_status_stuck_when_heartbeat_stale PASS")


def test_status_crashed_when_pid_dead() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths, state, log = _fresh(Path(tmp))
        # Use a pid that's very unlikely to be live (PID_MAX is typically
        # < 4M on Linux; 2**24 is outside that and unused).
        dead_pid = 99_999_999
        write_meta(paths, {
            "run_id": paths.run_id, "pid": dead_pid,
            "status": "running", "workspace": "/tmp/ws",
        })
        touch(paths.heartbeat)
        assert reconcile_status(paths, {
            "pid": dead_pid, "status": "running",
        }) == "crashed"
    print("  test_status_crashed_when_pid_dead PASS")


def test_status_terminal_from_events() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths, state, log = _fresh(Path(tmp))
        # Record a run_finished event and a finished state — should
        # reconcile as a terminal state regardless of pid.
        state.approved = True
        state.finished = True
        state.exit_code = 0
        log.emit("run_finished", approved=True, exit_code=0)
        assert reconcile_status(paths, {
            "pid": 1, "status": "exited",
        }) == "approved"
    print("  test_status_terminal_from_events PASS")


def test_status_killed_from_meta_after_signal() -> None:
    """When meta records killed + run_finished event present, status is killed."""
    with tempfile.TemporaryDirectory() as tmp:
        paths, state, log = _fresh(Path(tmp))
        state.finished = True
        state.exit_code = 143  # SIGTERM (128 + 15)
        state.approved = False
        log.emit("run_finished", approved=False, exit_code=143)
        assert reconcile_status(paths, {
            "pid": 1, "status": "killed",
        }) == "killed"
    print("  test_status_killed_from_meta_after_signal PASS")


def test_summarize_orphan_meta() -> None:
    """A meta.json with no pid / no heartbeat surfaces as crashed."""
    with tempfile.TemporaryDirectory() as tmp:
        paths, state, log = _fresh(Path(tmp))
        write_meta(paths, {
            "run_id": paths.run_id, "status": "running",
            "workspace": "/tmp/ws", "pid": 999_999_999,
        })
        row = summarize_run(paths)
        assert row is not None
        assert row.status == "crashed"
    print("  test_summarize_orphan_meta PASS")


def main() -> None:
    print("=" * 60)
    print("Test: ls status reconciliation (running / stuck / crashed / terminal)")
    print("-" * 60)
    test_status_running()
    test_status_stuck_when_heartbeat_stale()
    test_status_crashed_when_pid_dead()
    test_status_terminal_from_events()
    test_status_killed_from_meta_after_signal()
    test_summarize_orphan_meta()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
