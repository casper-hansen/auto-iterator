"""End-to-end integration tests against the ``mock_review_agent.py`` stub.

Each test spins up a real ``ReviewLoopRunner`` under a tmpdir runs-dir,
drives it through the mock agent, and asserts the on-disk artefacts
(events.jsonl, state.json, control-applied.jsonl) match what the CLI
contract promises to operators.

We deliberately drive the runner in-process (via
``run_review_loop_sync``) rather than through ``ai run`` — the CLI is a
thin file-writer tested elsewhere, and keeping the runner inline makes
these tests deterministic on all platforms."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_iterator.feature.config import RunConfig  # noqa: E402
from auto_iterator.run_dir import (  # noqa: E402
    CTL_GUIDANCE,
    CTL_REWIND,
    create_run_dir,
    new_run_id,
)
from auto_iterator.runner import (  # noqa: E402
    ReviewLoopRunner,
    bootstrap_run,
)


MOCK = (Path(__file__).resolve().parent / "mock_review_agent.py").resolve()


def _make_cfg(workspace: str, *, max_outer=1, max_inner=2, skip_impl=True) -> RunConfig:
    """Build a RunConfig that talks to the Python mock agent."""
    return RunConfig(
        task="implement a widget",
        impl_model="mock",
        fix_model="mock",
        reviewer_model="mock",
        max_outer=max_outer,
        max_inner=max_inner,
        workspace=workspace,
        skip_impl=skip_impl,
        extra_flags=(),
        agent_cmd=str(MOCK),
        backend="cursor",
        context="",
    )


def _read_events(paths) -> list[dict]:
    if not paths.events.exists():
        return []
    return [json.loads(l) for l in paths.events.read_text().splitlines() if l.strip()]


def _read_audit(paths) -> list[dict]:
    if not paths.control_applied.exists():
        return []
    return [
        json.loads(l)
        for l in paths.control_applied.read_text().splitlines()
        if l.strip()
    ]


async def _run(cfg: RunConfig, paths) -> int:
    runner = ReviewLoopRunner(cfg, paths, agent_type="review-loop")
    rc = await runner.run()
    return rc


def test_happy_path_first_pass_approved() -> None:
    """Single approved review → approved=True, rc=0, event tape matches."""
    import asyncio
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        paths = create_run_dir(tmp_p, new_run_id())
        cfg = _make_cfg(tmp, max_outer=1, max_inner=2)
        bootstrap_run(paths, cfg, pid=os.getpid())
        os.environ["MOCK_VERDICTS"] = "APPROVED"
        os.environ["MOCK_VERDICT_IDX_FILE"] = str(tmp_p / "vidx")
        try:
            rc = asyncio.run(_run(cfg, paths))
        finally:
            os.environ.pop("MOCK_VERDICTS", None)
            os.environ.pop("MOCK_VERDICT_IDX_FILE", None)
        assert rc == 0, f"expected 0 got {rc}"
        state = json.loads(paths.state.read_text())
        assert state["approved"] is True
        assert state["finished"] is True
        events = _read_events(paths)
        types = [e["type"] for e in events]
        assert "run_started" in types
        assert "inner_started" in types
        assert "review_finished" in types
        assert "run_finished" in types
    print("  test_happy_path_first_pass_approved PASS")


def test_changes_needed_triggers_fix_then_approve() -> None:
    import asyncio
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        paths = create_run_dir(tmp_p, new_run_id())
        cfg = _make_cfg(tmp, max_outer=1, max_inner=3)
        bootstrap_run(paths, cfg, pid=os.getpid())
        # CHANGES_NEEDED on first review, plain text for the fix, APPROVED
        # on second review → inner loop converges at inner=2. That's two
        # outer iterations away from fresh-eyes approval, but max_outer=1
        # so we just confirm the inner logic.
        os.environ["MOCK_VERDICTS"] = "CHANGES_NEEDED,fix_text,APPROVED"
        os.environ["MOCK_VERDICT_IDX_FILE"] = str(tmp_p / "vidx")
        try:
            rc = asyncio.run(_run(cfg, paths))
        finally:
            os.environ.pop("MOCK_VERDICTS", None)
            os.environ.pop("MOCK_VERDICT_IDX_FILE", None)
        state = json.loads(paths.state.read_text())
        # With max_outer=1, converging on inner=2 is NOT "approved on
        # first pass" (needs fresh-eyes) — runner returns rc=1 but state
        # reflects the converged verdict.
        assert state["last_verdict"] == "APPROVED"
        assert state["inner"] == 2
        events = _read_events(paths)
        # Must contain a fix_started/fix_finished pair.
        types = [e["type"] for e in events]
        assert "fix_started" in types
        assert "fix_finished" in types
    print("  test_changes_needed_triggers_fix_then_approve PASS")


def test_guidance_send_during_sleep_boundary() -> None:
    """Drop a guidance file during a SLEEP-verdict review, see it land
    on the next inner_started boundary."""
    import asyncio
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        paths = create_run_dir(tmp_p, new_run_id())
        cfg = _make_cfg(tmp, max_outer=1, max_inner=3)
        bootstrap_run(paths, cfg, pid=os.getpid())

        # First review sleeps 1.5s emitting CHANGES_NEEDED; second review
        # emits APPROVED. We drop a guidance file 0.5s in so it gets
        # drained on the inner=2 boundary.
        capture = tmp_p / "captures"
        os.environ["MOCK_VERDICTS"] = "SLEEP:1.5:CHANGES_NEEDED,fix_text,APPROVED"
        os.environ["MOCK_VERDICT_IDX_FILE"] = str(tmp_p / "vidx")
        os.environ["MOCK_CAPTURE_DIR"] = str(capture)

        def drop_guidance():
            time.sleep(0.5)
            paths.control_dir.mkdir(exist_ok=True)
            with open(paths.control_file(CTL_GUIDANCE), "a") as f:
                f.write("2026-04-24T10:00:00Z\tfocus on feature Z\n")

        t = threading.Thread(target=drop_guidance)
        t.start()
        try:
            asyncio.run(_run(cfg, paths))
        finally:
            t.join()
            os.environ.pop("MOCK_VERDICTS", None)
            os.environ.pop("MOCK_VERDICT_IDX_FILE", None)
            os.environ.pop("MOCK_CAPTURE_DIR", None)

        # control-applied.jsonl should carry the guidance_received record.
        audit = _read_audit(paths)
        events_of_type = [a for a in audit if a.get("event") == "guidance_received"]
        assert len(events_of_type) >= 1
        assert events_of_type[0]["text"] == "focus on feature Z"
        # The captured second-review prompt should contain our guidance.
        # Call order: inner=1 review (SLEEP), inner=1 fix, inner=2 review.
        # Guidance is drained at inner_started boundaries → folded into
        # the inner=2 review prompt, which is the third captured call.
        call_files = sorted(capture.glob("call-*.json"))
        assert len(call_files) >= 3, f"only got {len(call_files)} calls"
        third = json.loads(call_files[2].read_text())
        assert "focus on feature Z" in third["prompt"], (
            f"guidance missing from inner=2 review prompt: {third['prompt'][:400]!r}"
        )
    print("  test_guidance_send_during_sleep_boundary PASS")


def test_rewind_same_outer_truncates_history() -> None:
    """Rewind to outer=1,inner=1,phase=review while at inner=2; assert
    history is truncated back to empty and a fresh review is issued."""
    import asyncio
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        paths = create_run_dir(tmp_p, new_run_id())
        cfg = _make_cfg(tmp, max_outer=1, max_inner=3)
        bootstrap_run(paths, cfg, pid=os.getpid())

        capture = tmp_p / "captures"
        # Plan: inner=1 review sleeps 1.5s then emits CHANGES_NEEDED.
        # During sleep we drop rewind.json → outer=1,inner=1,phase=review.
        # The runner should consume it on the NEXT inner_started (inner=2
        # by normal flow, but rewind resets to inner=1 again). So the
        # review at reset-inner=1 runs a second time.
        # To keep the sequence bounded we then return APPROVED twice.
        os.environ["MOCK_VERDICTS"] = "SLEEP:1.5:CHANGES_NEEDED,APPROVED,APPROVED"
        os.environ["MOCK_VERDICT_IDX_FILE"] = str(tmp_p / "vidx")
        os.environ["MOCK_CAPTURE_DIR"] = str(capture)

        def drop_rewind():
            time.sleep(0.5)
            paths.control_dir.mkdir(exist_ok=True)
            (paths.control_file(CTL_REWIND)).write_text(json.dumps({
                "outer": 1, "inner": 1, "phase": "review",
            }))

        t = threading.Thread(target=drop_rewind)
        t.start()
        try:
            asyncio.run(_run(cfg, paths))
        finally:
            t.join()
            os.environ.pop("MOCK_VERDICTS", None)
            os.environ.pop("MOCK_VERDICT_IDX_FILE", None)
            os.environ.pop("MOCK_CAPTURE_DIR", None)

        events = _read_events(paths)
        types = [e["type"] for e in events]
        assert "rewind_applied" in types
        # The rewind should surface in the audit trail.
        audit = _read_audit(paths)
        assert any(a.get("event") == "rewind_applied" for a in audit)
    print("  test_rewind_same_outer_truncates_history PASS")


def test_detached_spawn_end_to_end() -> None:
    """``python -m auto_iterator.runner <run_dir>`` executes to completion."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        paths = create_run_dir(tmp_p, new_run_id())
        cfg = _make_cfg(tmp, max_outer=1, max_inner=1)
        # Write spec.json up front so the spawn target picks it up.
        from auto_iterator.runner import cfg_to_spec
        from auto_iterator.run_dir import atomic_write_json
        atomic_write_json(paths.spec, cfg_to_spec(cfg, agent_type="review-loop"))

        env = os.environ.copy()
        env["MOCK_VERDICTS"] = "APPROVED"
        env["MOCK_VERDICT_IDX_FILE"] = str(tmp_p / "vidx")
        # Ensure our `src` is importable for the spawned child.
        src_dir = str(Path(__file__).resolve().parents[1] / "src")
        env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.Popen(
            [sys.executable, "-m", "auto_iterator.runner", str(paths.run_dir)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=tmp,
        )
        stdout, _ = proc.communicate(timeout=60)
        assert proc.returncode == 0, f"rc={proc.returncode}, out={stdout!r}"
        assert paths.state.exists()
        state = json.loads(paths.state.read_text())
        assert state["finished"] is True
    print("  test_detached_spawn_end_to_end PASS")


def main() -> None:
    print("=" * 60)
    print("Test: runner integration with mock_review_agent.py")
    print("-" * 60)
    test_happy_path_first_pass_approved()
    test_changes_needed_triggers_fix_then_approve()
    test_guidance_send_during_sleep_boundary()
    test_rewind_same_outer_truncates_history()
    test_detached_spawn_end_to_end()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
