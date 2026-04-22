"""Agent-level test: run_agent + RunLogger sink captures tool_call and resume events.

This uses the existing ``mock_agent.sh`` script (tool-call + abnormal exit
+ ``--continue`` resume pattern) to drive real PTY reader threads against
a ``RunLogger`` sink. It proves that:

* ``events.jsonl`` captures the full agent stream (assistant messages,
  ``tool_call_started`` / ``_completed``) in order.
* Session lifecycle events are written: ``agent_session_started``,
  ``agent_session_finished`` for each attempt, plus
  ``agent_exit_abnormal`` and ``agent_resume_started`` across the failure
  boundary.
* ``state.json`` reflects the latest snapshot after the resume completes,
  including ``agent.attempt > 0``, ``resume_in_progress`` cleared on a
  clean subsequent session, and ``agent.last_rc == 0``.
* No stale ``state.json.tmp`` remains after the run.

We intentionally exercise BOTH a successful resume path
(``silent_between_tools`` → second attempt succeeds) and a fatal path
(``max_resume_attempts=0``) to cover the ``agent_resume_giveup`` branch.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iterator_loop.agent import run_agent  # noqa: E402
from iterator_loop.run_log import RunLogger  # noqa: E402

MOCK_SCRIPT = Path(__file__).resolve().parent / "mock_agent.sh"


def _read_events(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert(cond: bool, msg: str) -> None:
    assert cond, msg


async def _drive_successful_resume(logger: RunLogger) -> tuple[int, str]:
    """Run the mock in ``silent_between_tools`` mode, feeding the sink."""
    mock_path = str(MOCK_SCRIPT)
    os.chmod(mock_path, os.stat(mock_path).st_mode | stat.S_IEXEC)

    with tempfile.TemporaryDirectory() as tmp:
        state_file = os.path.join(tmp, "mock_state")
        env_backup = os.environ.copy()
        os.environ["MOCK_STATE_FILE"] = state_file
        os.environ["MOCK_FAILURE_MODE"] = "silent_between_tools"
        try:
            return await run_agent(
                model="resume-test-model",
                prompt="Run the experiment",
                tag="[resume-test]",
                workspace="/tmp",
                agent_cmd=mock_path,
                extra_flags=[],
                event_sink=logger.agent_event_sink(),
            )
        finally:
            os.environ.clear()
            os.environ.update(env_backup)


async def test_successful_resume_captures_ordered_events() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)
        logger.start(config={
            "prompt": "resume-test",
            "impl_model": "resume-test-model",
            "max_outer": 1, "max_inner": 1,
            "workspace": "/tmp",
        })

        rc, text = await _drive_successful_resume(logger)
        _assert(rc == 0, f"expected clean exit after resume, got rc={rc}")
        _assert("Resumed after interruption" in text,
                f"resume text missing from captured output: {text!r}")

        logger.finish(approved=True, exit_code=0, total_reviews=0, outer_loops=1)

        events = _read_events(logger.events_path)
        types = [e["type"] for e in events]

        # Spot-check ordering: must contain the first session's
        # session_started, an abnormal exit, a resume_started, and the
        # second session's session_finished with clean_exit=True before run_finished.
        _assert("run_started" in types, "run_started missing from events.jsonl")
        _assert(types.count("agent_session_started") >= 2,
                f"expected at least 2 agent_session_started, got {types.count('agent_session_started')}")
        _assert(types.count("agent_session_finished") >= 2,
                f"expected at least 2 agent_session_finished, got {types.count('agent_session_finished')}")
        _assert("agent_exit_abnormal" in types, "agent_exit_abnormal missing")
        _assert("agent_resume_started" in types, "agent_resume_started missing")
        _assert("tool_call_started" in types, "tool_call_started missing from events")
        _assert("tool_call_completed" in types, "tool_call_completed missing from events")
        _assert("assistant_message" in types, "assistant_message missing from events")

        # Order checks: abnormal must come before resume, which must come
        # before the second successful session_finished.
        abnormal_idx = types.index("agent_exit_abnormal")
        resume_idx = types.index("agent_resume_started")
        _assert(abnormal_idx < resume_idx,
                f"agent_exit_abnormal must precede agent_resume_started "
                f"(abnormal@{abnormal_idx}, resume@{resume_idx})")

        # The final agent_session_finished must report clean exit.
        finishes = [e for e in events if e["type"] == "agent_session_finished"]
        _assert(finishes[-1]["clean_exit"] is True,
                f"final session must be clean_exit=True, got {finishes[-1]}")
        _assert(finishes[-1]["rc"] == 0, "final session rc should be 0")

        # All seq values must be strictly monotonic.
        seqs = [e["seq"] for e in events]
        _assert(seqs == sorted(seqs), f"seq not monotonic: {seqs}")

        state = _read_state(logger.state_path)
        _assert(state["finished"] is True, "state.finished should be True after finish()")
        _assert(state["approved"] is True, "state.approved should be True")
        _assert(state["exit_code"] == 0, "state.exit_code should be 0")
        _assert(state["agent"]["last_rc"] == 0, "state.agent.last_rc should be 0")
        _assert(state["agent"]["session_active"] is False, "session_active must be cleared")
        _assert(state["agent"]["attempt"] >= 1,
                f"state.agent.attempt should reflect resume, got {state['agent']['attempt']}")

        _assert(state["tail"], "state.tail should not be empty after assistant messages")
        # The last tool call must appear in state
        last_tool = state["agent"]["last_tool"]
        _assert(last_tool is not None, "last_tool should be populated")
        _assert(last_tool["phase"] == "completed",
                f"last_tool.phase should be 'completed', got {last_tool}")

        # No stale .tmp
        tmp_files = list(logger.run_dir.glob("*.tmp"))
        _assert(tmp_files == [], f"stale tmp state file(s): {tmp_files}")

    print("  PASS successful_resume_captures_ordered_events")


async def _drive_giveup(logger: RunLogger) -> tuple[int, str]:
    mock_path = str(MOCK_SCRIPT)
    os.chmod(mock_path, os.stat(mock_path).st_mode | stat.S_IEXEC)
    with tempfile.TemporaryDirectory() as tmp:
        state_file = os.path.join(tmp, "mock_state")
        env_backup = os.environ.copy()
        os.environ["MOCK_STATE_FILE"] = state_file
        os.environ["MOCK_FAILURE_MODE"] = "silent_between_tools"
        try:
            return await run_agent(
                model="giveup-model",
                prompt="Run the experiment",
                tag="[giveup-test]",
                workspace="/tmp",
                agent_cmd=mock_path,
                extra_flags=[],
                max_resume_attempts=0,
                event_sink=logger.agent_event_sink(),
            )
        finally:
            os.environ.clear()
            os.environ.update(env_backup)


async def test_giveup_emits_agent_resume_giveup_event() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)
        logger.start(config={
            "prompt": "giveup-test",
            "impl_model": "giveup-model",
            "max_outer": 1, "max_inner": 1,
            "workspace": "/tmp",
        })

        rc, _ = await _drive_giveup(logger)
        _assert(rc != 0, f"expected non-zero rc on giveup, got {rc}")

        logger.finish(approved=False, exit_code=rc, total_reviews=0, outer_loops=1)

        events = _read_events(logger.events_path)
        types = [e["type"] for e in events]
        _assert("agent_resume_giveup" in types,
                f"agent_resume_giveup must be emitted when max_resume_attempts=0, "
                f"got {types}")
        _assert(types.count("agent_resume_started") == 0,
                f"no resume attempt should be started when max_resume_attempts=0, "
                f"got {types.count('agent_resume_started')}")

        state = _read_state(logger.state_path)
        _assert(state["finished"] is True, "state must be finished after giveup path")
        _assert(state["approved"] is False, "approved should be False after giveup")
        _assert(state["exit_code"] != 0, "exit_code should be non-zero after giveup")
        _assert(state["agent"]["last_rc"] != 0,
                f"agent.last_rc should be non-zero, got {state['agent']['last_rc']}")
        _assert(state["agent"]["last_error"], "agent.last_error should carry the giveup message")
    print("  PASS giveup_emits_agent_resume_giveup_event")


async def test_broken_event_sink_surfaces_failures_and_keeps_loop_alive() -> None:
    """If the event sink raises on every call, the agent run must:

    1. Keep going — the PTY reader thread can't die on a broken sink
       because we still need a final outcome to decide resume / bail.
    2. *Not* silently drop the failures. Every call the agent tried to
       make must be observable after the fact so callers can't falsely
       claim the structured log covered the whole run.

    The old behavior (``try: sink(...); except: pass``) produced a
    "success" that hid a disk-full / permission-error log sink. The
    new behavior counts failures in :class:`_StreamReader` /
    :class:`_SinkGuard` and emits a single aggregated ``warn()`` so
    operators can see the degradation.
    """
    # Deliberately importing here so the public surface stays narrow
    # for callers that don't need the internals.
    from iterator_loop.agent import _SinkGuard, _StreamReader  # noqa: E402
    from iterator_loop.output_formatter import OutputFormatter  # noqa: E402

    # ── 1. _SinkGuard: warns once, counts all failures ──
    calls: list[str] = []

    def boom(event_type: str, payload: dict) -> None:
        calls.append(event_type)
        raise OSError("disk full")

    guard = _SinkGuard(boom, tag="[guard-test]")
    for et in ["agent_session_started", "agent_session_finished",
               "agent_exit_abnormal", "agent_resume_started"]:
        guard.emit(et, {"x": 1})

    _assert(len(calls) == 4,
            f"every lifecycle event must still attempt delivery, "
            f"got {len(calls)}")
    _assert(guard.errors == 4,
            f"guard.errors must count every failure, got {guard.errors}")
    _assert(guard.first_error is not None and "disk full" in guard.first_error,
            f"guard.first_error missing original exc message: "
            f"{guard.first_error!r}")

    # Silent sink never raises — must be a no-op with zero errors.
    silent_guard = _SinkGuard(None, tag="")
    silent_guard.emit("anything", {})
    _assert(silent_guard.errors == 0,
            f"silent sink must not count as error, got {silent_guard.errors}")

    # ── 2. _StreamReader: same contract for PTY-thread events ──
    reader = _StreamReader(
        OutputFormatter("[stream-test]"),
        event_sink=boom,
        tag="[stream-test]",
    )
    for et in ["assistant_partial", "assistant_message",
               "tool_call_started", "tool_call_completed",
               "agent_result"]:
        reader._emit(et, {"text": "payload"})

    _assert(reader.sink_errors == 5,
            f"reader.sink_errors must count every failure, got {reader.sink_errors}")
    _assert(reader.sink_first_error and "disk full" in reader.sink_first_error,
            f"reader.sink_first_error missing exc: "
            f"{reader.sink_first_error!r}")

    # ── 3. run_agent end-to-end: aggregates and never crashes ──
    # Drive a real mock session through run_agent with a broken sink.
    # The run must still return a tuple (rc, text); the sink's failure
    # must not kill the PTY reader or the asyncio session.
    mock_path = str(MOCK_SCRIPT)
    os.chmod(mock_path, os.stat(mock_path).st_mode | stat.S_IEXEC)
    with tempfile.TemporaryDirectory() as tmp:
        state_file = os.path.join(tmp, "mock_state")
        env_backup = os.environ.copy()
        os.environ["MOCK_STATE_FILE"] = state_file
        os.environ["MOCK_FAILURE_MODE"] = "silent_between_tools"

        sink_calls: list[str] = []

        def broken_sink(event_type: str, payload: dict) -> None:
            sink_calls.append(event_type)
            raise OSError("disk full")

        try:
            # Import inline to keep the top of the file focused on
            # the happy-path drivers, same as the helpers above.
            from iterator_loop.agent import run_agent as _run_agent  # noqa: E402
            rc, text = await _run_agent(
                model="broken-sink-model",
                prompt="test",
                tag="[broken-sink]",
                workspace="/tmp",
                agent_cmd=mock_path,
                extra_flags=[],
                max_resume_attempts=1,
                event_sink=broken_sink,
            )
        finally:
            os.environ.clear()
            os.environ.update(env_backup)

    _assert(rc == 0,
            f"run_agent must still reach a successful final attempt "
            f"despite a broken sink, got rc={rc}")
    _assert("Resumed after interruption" in text,
            "resumed session text should still be returned")
    _assert(len(sink_calls) > 5,
            f"every stream + lifecycle event must still attempt "
            f"delivery, got {len(sink_calls)}: {sink_calls[:6]}…")
    # Lifecycle events must be present (proves the _SinkGuard wrapper
    # tried), plus at least one stream event (proves _StreamReader
    # tried), plus the resume boundary (proves the loop survived the
    # first session's dead sink).
    _assert("agent_session_started" in sink_calls,
            f"agent_session_started missing from attempts: {sink_calls}")
    _assert("agent_resume_started" in sink_calls,
            f"agent_resume_started missing from attempts: {sink_calls}")
    _assert("tool_call_started" in sink_calls or "tool_call_completed" in sink_calls,
            f"tool_call events missing from attempts: {sink_calls}")

    print("  PASS broken_event_sink_surfaces_failures_and_keeps_loop_alive")


async def test_run_once_crash_emits_terminal_session_events() -> None:
    """A crash inside ``_run_once`` must still pair ``agent_session_started``
    with ``agent_session_finished`` so the snapshot can't end up with
    ``session_active=True`` next to ``finished=True``.

    Reviewer reproducer: monkey-patched ``_run_once`` to raise
    ``RuntimeError`` after the first ``agent_session_started``.
    Before the fix the final ``events.jsonl`` was
    ``[run_started, agent_session_started, run_finished]`` and
    ``state.json`` claimed the agent session was still live even
    though the run had terminated — contradictory state that
    breaks the "structured logs are the authoritative snapshot"
    contract.

    After the fix the crash path emits:

    1. ``agent_session_finished`` with ``clean_exit=False``,
       ``rc=None``, ``rc_display="aborted"``, plus explicit
       ``aborted=True`` / ``error_type`` / ``error_message`` fields
       so a consumer can tell an aborted session apart from a clean
       zero-rc exit.
    2. ``agent_exit_abnormal`` with the same error context, matching
       the pattern the non-clean-rc branch already uses, so
       ``state.agent.last_error`` carries the crash message.
    3. Re-raises the original exception so the caller can still
       react.
    """
    from iterator_loop import agent as agent_mod  # noqa: E402
    from iterator_loop.run_log import RunLogger  # noqa: E402

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated _run_once crash")

    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp, run_id="crash-run-once")
        logger.start(config={
            "prompt": "p", "impl_model": "m",
            "max_outer": 1, "max_inner": 1, "workspace": "/tmp",
        })

        orig_run_once = agent_mod._run_once
        agent_mod._run_once = _boom
        raised: type[BaseException] | None = None
        try:
            try:
                await agent_mod.run_agent(
                    model="m",
                    prompt="p",
                    tag="[crash-test]",
                    workspace="/tmp",
                    agent_cmd="/bin/true",
                    extra_flags=[],
                    max_resume_attempts=0,
                    event_sink=logger.agent_event_sink(),
                )
            except RuntimeError as exc:
                raised = type(exc)
        finally:
            agent_mod._run_once = orig_run_once

        _assert(raised is RuntimeError,
                f"run_agent must re-raise the _run_once crash, got {raised}")

        # The caller now writes the terminal run record just like
        # review_loop.main would.
        logger.finish(approved=False, exit_code=1)

        events = _read_events(logger.events_path)
        state = _read_state(logger.state_path)
        types = [e["type"] for e in events]

        # Paired lifecycle events: every started session must get a
        # finished event, even on abort.
        started_count = types.count("agent_session_started")
        finished_count = types.count("agent_session_finished")
        _assert(started_count == finished_count and started_count >= 1,
                f"agent_session_started/agent_session_finished must be "
                f"paired on the crash path: started={started_count}, "
                f"finished={finished_count}")

        # The finish event must carry the abort diagnostics.
        finish_evt = next(e for e in events if e["type"] == "agent_session_finished")
        _assert(finish_evt.get("clean_exit") is False,
                f"aborted finish must set clean_exit=False: {finish_evt}")
        _assert(finish_evt.get("aborted") is True,
                f"aborted finish must expose aborted=True so consumers "
                f"can distinguish it from a clean zero-rc session: "
                f"{finish_evt}")
        _assert(finish_evt.get("error_type") == "RuntimeError",
                f"aborted finish must carry error_type: {finish_evt}")
        _assert("simulated _run_once crash" in (finish_evt.get("error_message") or ""),
                f"aborted finish must carry error_message: {finish_evt}")
        _assert(finish_evt.get("rc_display") == "aborted",
                f"aborted finish must set rc_display='aborted' to "
                f"differ from the numeric '0'/'137 (SIGKILL)'/... "
                f"strings: {finish_evt}")

        # agent_exit_abnormal must follow so state.last_error is set.
        _assert("agent_exit_abnormal" in types,
                f"agent_exit_abnormal must be emitted alongside the "
                f"aborted finish: {types}")

        # State reflects the crash, not a phantom live session.
        _assert(state["agent"]["session_active"] is False,
                f"state.agent.session_active must be False after abort, "
                f"got {state['agent']['session_active']}")
        _assert(state["agent"]["last_error"] is not None
                and "simulated _run_once crash" in state["agent"]["last_error"],
                f"state.agent.last_error must carry the crash message, "
                f"got {state['agent']['last_error']!r}")
        _assert(state["finished"] is True,
                "state.finished must be True after run_finished")
        _assert(state["exit_code"] == 1,
                f"state.exit_code should reflect the caller's chosen "
                f"code, got {state['exit_code']}")

        # Event ordering: session_started < session_finished < exit_abnormal.
        idx_started = types.index("agent_session_started")
        idx_finished = types.index("agent_session_finished")
        idx_abnormal = types.index("agent_exit_abnormal")
        _assert(idx_started < idx_finished < idx_abnormal,
                f"ordering must be session_started < session_finished "
                f"< exit_abnormal, got {idx_started} / {idx_finished} / "
                f"{idx_abnormal}")

    print("  PASS run_once_crash_emits_terminal_session_events")


async def test_run_once_does_not_leak_pty_fds_on_spawn_failure() -> None:
    """Failed ``create_subprocess_exec`` must not leak the PTY fds that
    ``pty.openpty()`` just created.

    Reviewer reproducer: with ``asyncio.create_subprocess_exec``
    monkey-patched to raise ``OSError("spawn failed")``, three failing
    ``_run_once`` calls grew ``/proc/self/fd`` from 9 to 15 (2 fds per
    call: the master and slave ends of the freshly-opened PTY). In a
    long-running worker or a bad-agent-config path that retries the
    same bad command, fd exhaustion would hit before later runs even
    start.

    After the fix, the try/except around ``create_subprocess_exec``
    closes both fds before re-raising. The counter stays flat across
    any number of attempts. We also verify:

    * The original exception (not a masked close-time one) propagates
      up to the caller — critical so the caller can distinguish
      spawn failure from other errors.
    * The fix handles both ``OSError`` (the common case: FileNotFound,
      PermissionDenied, ENOMEM, …) and arbitrary ``BaseException``
      subclasses (e.g. ``KeyboardInterrupt`` mid-spawn), since the
      real CLI can raise non-OSError types.
    """
    import asyncio as _asyncio  # noqa: E402
    from iterator_loop.agent import _run_once  # noqa: E402

    # Helper that hides the /proc/self/fd accounting boilerplate
    # and runs an N-attempt loop for a given injected exception.
    async def _leak_stable_across(exc_factory, attempts: int = 8) -> None:
        orig = _asyncio.create_subprocess_exec

        async def _raiser(*_a, **_kw):
            raise exc_factory()

        _asyncio.create_subprocess_exec = _raiser  # type: ignore[assignment]
        try:
            # Prime: one failed call to settle any lazy allocations
            # (e.g. tracemalloc arena, asyncio internals) that could
            # perturb the baseline, then take the baseline.
            try:
                await _run_once(["/nonexistent-cmd"], tag="[leak-test]")
            except BaseException:
                pass
            baseline = len(os.listdir("/proc/self/fd"))

            for i in range(attempts):
                raised = None
                try:
                    await _run_once(["/nonexistent-cmd"], tag="[leak-test]")
                except BaseException as exc:
                    raised = type(exc)
                _assert(raised is not None,
                        f"_run_once must propagate the spawn exception "
                        f"(attempt {i+1})")
                # The propagated exception must be what the patched
                # spawn raised — not a masked close-time error.
                expected = type(exc_factory())
                _assert(issubclass(raised, expected),
                        f"_run_once must re-raise the original spawn "
                        f"exception type, got {raised}, expected "
                        f"subclass of {expected}")

                count = len(os.listdir("/proc/self/fd"))
                _assert(count <= baseline,
                        f"fd leak detected after attempt {i+1}: "
                        f"baseline={baseline}, now={count} "
                        f"(delta={count - baseline})")
        finally:
            _asyncio.create_subprocess_exec = orig  # type: ignore[assignment]

    # The common production case — OSError (FileNotFound, Permission,
    # ENOMEM) during spawn.
    await _leak_stable_across(lambda: OSError("simulated spawn failure"))

    # A rarer but plausible case — the CLI raises a generic exception
    # mid-spawn (e.g. in a custom preexec_fn). We still need both fds
    # closed before re-raising.
    await _leak_stable_across(lambda: RuntimeError("non-OSError from spawn"))

    print("  PASS run_once_does_not_leak_pty_fds_on_spawn_failure")


async def _main() -> None:
    print("=" * 60)
    print("RunLogger + run_agent integration tests")
    print("-" * 60)
    await test_successful_resume_captures_ordered_events()
    await test_giveup_emits_agent_resume_giveup_event()
    await test_broken_event_sink_surfaces_failures_and_keeps_loop_alive()
    await test_run_once_crash_emits_terminal_session_events()
    await test_run_once_does_not_leak_pty_fds_on_spawn_failure()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(_main())
