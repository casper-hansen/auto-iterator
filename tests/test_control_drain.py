"""Unit tests for the control-file drain.

Exercises every intent type (guidance, rewind, prompt, context, pause)
in isolation from the runner. The drain is the single bottleneck through
which every operator intent flows, so we hit the happy paths *and* the
edge cases (two guidance writes in the same window, malformed rewind,
missing-file no-ops, phase=after_impl normalisation)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_iterator.control import (  # noqa: E402
    drain_control,
    parse_rewind_to,
    wait_while_paused,
)
from auto_iterator.events import EventLog, RunState  # noqa: E402
from auto_iterator.run_dir import (  # noqa: E402
    CTL_CONTEXT,
    CTL_GUIDANCE,
    CTL_PAUSE,
    CTL_PROMPT,
    CTL_REWIND,
    create_run_dir,
    new_run_id,
)


def _setup(tmp: Path):
    paths = create_run_dir(tmp, new_run_id())
    state = RunState(run_id=paths.run_id, prompt="orig prompt",
                     context="orig context", workspace="/tmp/ws",
                     outer=2, inner=3)
    log = EventLog(paths, state)
    return paths, state, log


def test_drain_noop_when_empty() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths, state, log = _setup(Path(tmp))
        res = drain_control(paths, state, log)
        assert res is None
        # No events should have been emitted by an empty drain (only
        # events.jsonl's file existence is OK since EventLog may seed
        # nothing; we assert no mutation happened).
        assert state.prompt == "orig prompt"
        assert state.context == "orig context"
        assert state.guidance_queue == []
    print("  test_drain_noop_when_empty PASS")


def test_drain_guidance_single() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths, state, log = _setup(Path(tmp))
        f = paths.control_file(CTL_GUIDANCE)
        f.write_text("2026-04-24T10:00:00Z\tfocus on X\n")
        res = drain_control(paths, state, log)
        assert res is None
        assert state.guidance_queue == ["focus on X"]
        assert not f.exists()
        # control-applied audit has the guidance_received line.
        audit_lines = paths.control_applied.read_text().strip().splitlines()
        assert any(json.loads(line).get("event") == "guidance_received"
                   for line in audit_lines)
    print("  test_drain_guidance_single PASS")


def test_drain_guidance_multi_concurrent() -> None:
    """Two back-to-back appends both survive the drain."""
    with tempfile.TemporaryDirectory() as tmp:
        paths, state, log = _setup(Path(tmp))
        f = paths.control_file(CTL_GUIDANCE)
        # Simulate two `ai send` calls appending to the same file
        # before the runner has a chance to drain.
        fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.write(fd, b"2026-04-24T10:00:00Z\tfirst\n")
        os.close(fd)
        fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.write(fd, b"2026-04-24T10:00:01Z\tsecond\n")
        os.close(fd)
        drain_control(paths, state, log)
        assert state.guidance_queue == ["first", "second"]
    print("  test_drain_guidance_multi_concurrent PASS")


def test_drain_prompt_and_context() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths, state, log = _setup(Path(tmp))
        paths.control_file(CTL_PROMPT).write_text("new prompt text")
        paths.control_file(CTL_CONTEXT).write_text("new context blob")
        drain_control(paths, state, log)
        assert state.prompt == "new prompt text"
        assert state.context == "new context blob"
        assert not paths.control_file(CTL_PROMPT).exists()
        assert not paths.control_file(CTL_CONTEXT).exists()
    print("  test_drain_prompt_and_context PASS")


def test_drain_rewind_valid_phases() -> None:
    for phase, out, inn in [
        ("review", 1, 2),
        ("fix", 2, 3),
    ]:
        with tempfile.TemporaryDirectory() as tmp:
            paths, state, log = _setup(Path(tmp))
            paths.control_file(CTL_REWIND).write_text(json.dumps({
                "outer": out, "inner": inn, "phase": phase,
            }))
            intent = drain_control(paths, state, log)
            assert intent is not None
            assert (intent.outer, intent.inner, intent.phase) == (out, inn, phase)
            assert not paths.control_file(CTL_REWIND).exists()
    print("  test_drain_rewind_valid_phases PASS")


def test_drain_rewind_after_impl_normalizes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths, state, log = _setup(Path(tmp))
        paths.control_file(CTL_REWIND).write_text(json.dumps({
            "outer": 7, "inner": 9, "phase": "after_impl",
        }))
        intent = drain_control(paths, state, log)
        assert intent is not None
        assert (intent.outer, intent.inner, intent.phase) == (0, 0, "after_impl")
    print("  test_drain_rewind_after_impl_normalizes PASS")


def test_drain_rewind_malformed_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths, state, log = _setup(Path(tmp))
        # Invalid JSON
        paths.control_file(CTL_REWIND).write_text("not json {")
        intent = drain_control(paths, state, log)
        assert intent is None
        # Invalid phase
        paths.control_file(CTL_REWIND).write_text(json.dumps({
            "outer": 1, "inner": 1, "phase": "weird",
        }))
        intent = drain_control(paths, state, log)
        assert intent is None
        # Zero/negative
        paths.control_file(CTL_REWIND).write_text(json.dumps({
            "outer": 0, "inner": 0, "phase": "review",
        }))
        intent = drain_control(paths, state, log)
        assert intent is None
        # control-applied audit has control_rejected entries.
        audit = paths.control_applied.read_text()
        assert audit.count("control_rejected") >= 3
    print("  test_drain_rewind_malformed_rejected PASS")


def test_drain_rewind_concurrent_rewrite_survives() -> None:
    """A concurrent ``ai rewind`` landing during the drain window must
    leave its payload on disk for the next drain tick. Regression test
    for the read-then-unlink race: monkey-patched to inject a second
    atomic write in the window between rename and unlink."""
    with tempfile.TemporaryDirectory() as tmp:
        paths, state, log = _setup(Path(tmp))
        rw = paths.control_file(CTL_REWIND)
        rw.write_text(json.dumps({"outer": 1, "inner": 1, "phase": "review"}))

        orig = Path.read_text
        second = json.dumps({"outer": 5, "inner": 7, "phase": "fix"})

        def racy(self, *a, **kw):
            if ".draining." in self.name:
                rw.write_text(second)
            return orig(self, *a, **kw)

        Path.read_text = racy
        try:
            intent = drain_control(paths, state, log)
        finally:
            Path.read_text = orig

        assert intent is not None
        assert (intent.outer, intent.inner, intent.phase) == (1, 1, "review")
        assert rw.exists(), "concurrent writer's rewind.json was clobbered"
        assert json.loads(rw.read_text()) == {
            "outer": 5, "inner": 7, "phase": "fix",
        }

        state2 = RunState(run_id=paths.run_id, prompt="", context="",
                          workspace="/tmp/ws", outer=0, inner=0)
        log2 = EventLog(paths, state2)
        intent2 = drain_control(paths, state2, log2)
        assert intent2 is not None
        assert (intent2.outer, intent2.inner, intent2.phase) == (5, 7, "fix")
    print("  test_drain_rewind_concurrent_rewrite_survives PASS")


def test_drain_prompt_concurrent_rewrite_survives() -> None:
    """Same race class as rewind, exercised through ``_take_text`` via
    the prompt consumer. A concurrent ``ai set-prompt`` landing during
    the drain window must survive for the next tick."""
    with tempfile.TemporaryDirectory() as tmp:
        paths, state, log = _setup(Path(tmp))
        pf = paths.control_file(CTL_PROMPT)
        pf.write_text("first prompt")

        orig = Path.read_text

        def racy(self, *a, **kw):
            if ".draining." in self.name:
                pf.write_text("second prompt")
            return orig(self, *a, **kw)

        Path.read_text = racy
        try:
            drain_control(paths, state, log)
        finally:
            Path.read_text = orig

        assert state.prompt == "first prompt"
        assert pf.exists(), "concurrent writer's prompt.txt was clobbered"
        assert pf.read_text() == "second prompt"
    print("  test_drain_prompt_concurrent_rewrite_survives PASS")


def test_drain_pause_edge_transitions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths, state, log = _setup(Path(tmp))
        # Set pause file and drain: should flip state.paused True.
        paths.control_file(CTL_PAUSE).touch()
        drain_control(paths, state, log)
        assert state.paused is True
        # Remove pause file, drain again: should flip False.
        paths.control_file(CTL_PAUSE).unlink()
        drain_control(paths, state, log)
        assert state.paused is False
    print("  test_drain_pause_edge_transitions PASS")


def test_wait_while_paused_returns_quick_when_no_pause() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths, state, log = _setup(Path(tmp))
        t0 = time.time()
        wait_while_paused(paths, state, log, poll_interval=0.01)
        assert time.time() - t0 < 1.0
    print("  test_wait_while_paused_returns_quick_when_no_pause PASS")


def main() -> None:
    print("=" * 60)
    print("Test: control-file drain (guidance / prompt / context / rewind / pause)")
    print("-" * 60)
    test_drain_noop_when_empty()
    test_drain_guidance_single()
    test_drain_guidance_multi_concurrent()
    test_drain_prompt_and_context()
    test_drain_rewind_valid_phases()
    test_drain_rewind_after_impl_normalizes()
    test_drain_rewind_malformed_rejected()
    test_drain_rewind_concurrent_rewrite_survives()
    test_drain_prompt_concurrent_rewrite_survives()
    test_drain_pause_edge_transitions()
    test_wait_while_paused_returns_quick_when_no_pause()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
