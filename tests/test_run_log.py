"""Unit tests for the per-run filesystem layout.

Covers the small, easy-to-regress primitives that every higher-level
feature sits on:

* ``meta.json`` / ``spec.json`` round-trips.
* Atomic writes and append-jsonl ordering.
* Control-file parse for ``rewind --to`` shorthand.

No subprocesses; everything runs in-process against tmpdirs. These
tests are the safety net the integration tests rely on — if any one of
them fails, the rest of the suite is untrustworthy."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_iterator.control import parse_rewind_to  # noqa: E402
from auto_iterator.events import EventLog, RunState  # noqa: E402
from auto_iterator.feature.config import RunConfig  # noqa: E402
from auto_iterator.meta import read_meta, update_meta, write_meta  # noqa: E402
from auto_iterator.run_dir import (  # noqa: E402
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    create_run_dir,
    new_run_id,
    now_iso,
    read_last_jsonl,
)
from auto_iterator.runner import cfg_to_spec, spec_to_cfg  # noqa: E402


def _run_paths(tmp: Path):
    run_id = new_run_id()
    paths = create_run_dir(tmp, run_id)
    assert paths.run_dir.exists()
    assert paths.control_dir.exists()
    assert paths.logs_dir.exists()
    return paths


def test_meta_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = _run_paths(Path(tmp))
        write_meta(paths, {
            "run_id": paths.run_id,
            "pid": 1234,
            "status": "running",
            "workspace": "/tmp/ws",
        })
        got = read_meta(paths)
        assert got["run_id"] == paths.run_id
        assert got["pid"] == 1234
        assert got["status"] == "running"

        updated = update_meta(paths, status="exited", finished_at=now_iso())
        assert updated["status"] == "exited"
        again = read_meta(paths)
        assert again["status"] == "exited"
        assert again["pid"] == 1234  # untouched fields survive
    print("  test_meta_roundtrip PASS")


def test_spec_roundtrip() -> None:
    cfg = RunConfig(
        task="implement X",
        impl_model="impl-m",
        fix_model="fix-m",
        reviewer_model="rev-m",
        max_outer=3,
        max_inner=4,
        workspace="/tmp/ws",
        skip_impl=False,
        extra_flags=("--foo", "--bar"),
        agent_cmd="claude",
        backend="claude-code",
        context="some context",
    )
    spec = cfg_to_spec(cfg, agent_type="review-loop")
    # Must be JSON-serializable.
    encoded = json.dumps(spec)
    decoded = json.loads(encoded)
    round_tripped = spec_to_cfg(decoded)
    assert round_tripped == cfg
    print("  test_spec_roundtrip PASS")


def test_atomic_write_and_append() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = _run_paths(Path(tmp))
        # Atomic JSON: partial crash would leave a tmp file, never half.
        atomic_write_json(paths.state, {"hello": "world"})
        assert json.loads(paths.state.read_text()) == {"hello": "world"}

        # Append-jsonl preserves order and parses line-by-line.
        for i in range(5):
            append_jsonl(paths.events, {"seq": i, "type": "probe"})
        lines = paths.events.read_text().splitlines()
        assert [json.loads(line)["seq"] for line in lines] == [0, 1, 2, 3, 4]

        # Last-line reader picks the most recent event.
        assert read_last_jsonl(paths.events) == {"seq": 4, "type": "probe"}
    print("  test_atomic_write_and_append PASS")


def test_event_log_seq_and_snapshot() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = _run_paths(Path(tmp))
        state = RunState(run_id=paths.run_id, prompt="do the thing",
                         workspace="/tmp/ws")
        log = EventLog(paths, state)
        evt1 = log.emit("run_started", workspace="/tmp/ws")
        evt2 = log.emit("inner_started", outer=1, inner=1)
        assert evt1["seq"] == 1
        assert evt2["seq"] == 2
        # State snapshot should exist and match current state.
        snap = json.loads(paths.state.read_text())
        assert snap["run_id"] == paths.run_id
        assert snap["phase"] == "init"
        # A second EventLog (simulating crash-restart) resumes sequence.
        log2 = EventLog(paths, state)
        evt3 = log2.emit("review_started", outer=1, inner=1, model="m")
        assert evt3["seq"] == 3
    print("  test_event_log_seq_and_snapshot PASS")


def test_parse_rewind_to() -> None:
    # Happy path
    r = parse_rewind_to("outer=1,inner=2")
    assert (r.outer, r.inner, r.phase) == (1, 2, "review")

    r2 = parse_rewind_to("outer=2,inner=3,phase=fix")
    assert (r2.outer, r2.inner, r2.phase) == (2, 3, "fix")

    r3 = parse_rewind_to("phase=after_impl")
    assert (r3.outer, r3.inner, r3.phase) == (0, 0, "after_impl")

    # Malformed inputs
    for bad in ["outer=1", "inner=1", "phase=review", "outer=x,inner=1",
                "outer=0,inner=1", "outer=1,inner=0,phase=fix",
                "outer=1,inner=1,phase=weird", "nope"]:
        try:
            parse_rewind_to(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for '{bad}'")
    print("  test_parse_rewind_to PASS")


def test_run_dir_permissions() -> None:
    import os, stat
    with tempfile.TemporaryDirectory() as tmp:
        paths = _run_paths(Path(tmp))
        # Per-run dir, control/, logs/ are all 0700 (or tighter).
        for d in (paths.run_dir, paths.control_dir, paths.logs_dir):
            m = stat.S_IMODE(os.stat(d).st_mode)
            assert m & 0o077 == 0, f"world/group bits on {d}: {oct(m)}"

        atomic_write_json(paths.meta, {"run_id": paths.run_id})
        m = stat.S_IMODE(os.stat(paths.meta).st_mode)
        assert m & 0o077 == 0, f"world/group bits on meta.json: {oct(m)}"

        atomic_write_text(paths.spec, "probe")
        m = stat.S_IMODE(os.stat(paths.spec).st_mode)
        assert m & 0o077 == 0, f"world/group bits on spec.json: {oct(m)}"
    print("  test_run_dir_permissions PASS")


def main() -> None:
    print("=" * 60)
    print("Test: run-dir primitives (meta / spec / events / control / perms)")
    print("-" * 60)
    test_meta_roundtrip()
    test_spec_roundtrip()
    test_atomic_write_and_append()
    test_event_log_seq_and_snapshot()
    test_parse_rewind_to()
    test_run_dir_permissions()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
