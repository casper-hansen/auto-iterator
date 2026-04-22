"""Orchestration-level test: review-loop.py drives review/fix/outer events.

This test wires the mock review agent (``tests/mock_review_agent.py``)
into the real ``review-loop.main`` entrypoint and inspects the resulting
structured logs. It covers:

* Starting a run creates ``logs/<run_id>/events.jsonl`` +
  ``logs/<run_id>/state.json`` and appends a ``run_started`` line to
  ``logs/index.jsonl``.
* The review → fix → review → fresh-eyes path emits the expected
  orchestration events (``outer_started``, ``inner_started``,
  ``review_started``/``review_finished`` with verdict, ``fix_started``/
  ``fix_finished``) in the right order.
* Agent/tool events ride in the same ``events.jsonl`` as orchestration
  events (proving the layers are not separated into two sinks).
* On run exit, ``index.jsonl`` receives ``run_finished`` and the final
  ``state.json`` carries ``finished=True`` with the terminal snapshot.
* A future consumer can answer "what phase/verdict?" purely from
  ``state.json``.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

# ``review-loop.py`` is not a package; importlib loads it as a module file.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "review_loop", REPO_ROOT / "review-loop.py",
)
assert _spec and _spec.loader, "failed to locate review-loop.py"
review_loop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review_loop)

MOCK_AGENT = REPO_ROOT / "tests" / "mock_review_agent.py"


def _read_events(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_index(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _assert(cond: bool, msg: str) -> None:
    assert cond, msg


class _preserve_stdout:
    """Save/restore fd 1 + ``sys.stdout`` + the broken-flag around a test.

    The production broken-pipe defence swaps fd 1 to ``/dev/null`` on
    the first ``OSError`` observed against stdout (see
    :func:`iterator_loop.logging.mark_stdout_broken`). That swap is
    global and irreversible within a process — which is the whole
    point in production. Tests that deliberately force a broken-pipe
    scenario (banner/summary monkey-patched to raise
    ``BrokenPipeError``) would therefore also silence the test
    runner's own stdout for every subsequent test in the same
    process, so all following ``PASS`` lines vanish even though the
    tests themselves succeed.

    This context manager captures fd 1 + ``sys.stdout`` + the
    ``_stdout_broken`` flag on entry and restores them on exit so
    the runner's output survives. It is only needed in tests that
    intentionally exercise the swap; normal runs should leave fd 1
    alone.
    """

    def __enter__(self) -> "_preserve_stdout":
        from iterator_loop import logging as lg

        self._lg = lg
        self._saved_fd1 = os.dup(1)
        self._saved_stdout = sys.stdout
        self._saved_stderr = sys.stderr
        self._saved_broken = lg._stdout_broken
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            os.dup2(self._saved_fd1, 1)
        finally:
            os.close(self._saved_fd1)
        sys.stdout = self._saved_stdout
        sys.stderr = self._saved_stderr
        self._lg._stdout_broken = self._saved_broken


async def _run_orchestration(
    logs_dir: Path,
    schedule: str,
    *,
    tool_call: bool = False,
    max_outer: int = 3,
    max_inner: int = 3,
    extra_env: dict | None = None,
) -> int:
    """Invoke review_loop.main with the mock agent and capture its logs."""
    os.chmod(MOCK_AGENT, os.stat(MOCK_AGENT).st_mode | stat.S_IEXEC)
    with tempfile.NamedTemporaryFile("w", suffix="_mockstate") as state_file:
        state_file.write("")
        state_file.flush()

        env_backup = os.environ.copy()
        os.environ["AGENT_CMD"] = str(MOCK_AGENT)
        os.environ["MOCK_STATE_FILE"] = state_file.name
        os.environ["MOCK_SCHEDULE"] = schedule
        if tool_call:
            os.environ["MOCK_TOOL_CALL"] = "1"
        else:
            os.environ.pop("MOCK_TOOL_CALL", None)
        if extra_env:
            os.environ.update(extra_env)

        # Force colour off so captured text stays clean, and make the
        # review-loop treat the mock script as the agent binary.
        os.environ["NO_COLOR"] = "1"

        try:
            return await review_loop.main([
                "--prompt", "test-task",
                "--skip-impl",
                "--impl-model", "m-impl",
                "--fix-model", "m-fix",
                "--reviewer", "m-rev",
                "--max-outer", str(max_outer),
                "--max-inner", str(max_inner),
                "--workspace", "/tmp",
                "--logs-dir", str(logs_dir),
                "--run-id", "orchestration-test-run",
            ])
        finally:
            os.environ.clear()
            os.environ.update(env_backup)


async def test_approved_on_first_pass_emits_clean_flow() -> None:
    """Single outer, single review, APPROVED → terminal approved=True."""
    with tempfile.TemporaryDirectory() as tmp:
        logs_dir = Path(tmp)
        rc = await _run_orchestration(
            logs_dir,
            schedule="Review complete.\nVERDICT: APPROVED",
            tool_call=True,
        )
        _assert(rc == 0, f"expected rc=0 on clean approve, got {rc}")

        run_dir = logs_dir / "orchestration-test-run"
        events = _read_events(run_dir / "events.jsonl")
        index = _read_index(logs_dir / "index.jsonl")
        state = _read_state(run_dir / "state.json")

        types = [e["type"] for e in events]

        # Orchestration events present in order
        for required in [
            "run_started",
            "outer_started",
            "inner_started",
            "review_started",
            "review_finished",
            "inner_finished",
            "outer_finished",
            "run_finished",
        ]:
            _assert(required in types, f"missing orchestration event: {required}\n{types}")

        # Agent-side stream events must ride in the same file
        _assert("agent_session_started" in types, "agent_session_started missing")
        _assert("agent_session_finished" in types, "agent_session_finished missing")
        _assert("tool_call_started" in types, "tool_call_started missing")
        _assert("tool_call_completed" in types, "tool_call_completed missing")
        _assert("assistant_message" in types, "assistant_message missing")

        # Review finished event carries the parsed verdict
        rf = next(e for e in events if e["type"] == "review_finished")
        _assert(rf["verdict"] == "APPROVED", f"verdict mismatch: {rf}")

        # Ordering: review_started < review_finished < run_finished
        idx_rs = types.index("review_started")
        idx_rf = types.index("review_finished")
        idx_end = types.index("run_finished")
        _assert(idx_rs < idx_rf < idx_end,
                f"event ordering broken: review_started@{idx_rs} "
                f"review_finished@{idx_rf} run_finished@{idx_end}")

        # Index has both lifecycle markers
        index_types = [e["type"] for e in index]
        _assert(index_types == ["run_started", "run_finished"],
                f"index.jsonl unexpected: {index_types}")
        _assert(index[1]["approved"] is True, "index run_finished.approved should be True")

        # Final state snapshot is trustworthy as a source of truth
        _assert(state["finished"] is True, "state.finished must be True")
        _assert(state["approved"] is True, "state.approved must be True")
        _assert(state["phase"] == "finished", f"state.phase should be 'finished', got {state['phase']}")
        _assert(state["last_verdict"] == "APPROVED", "state.last_verdict should be APPROVED")
        _assert(state["total_reviews"] == 1, f"state.total_reviews should be 1, got {state['total_reviews']}")
        _assert(state["exit_code"] == 0, "state.exit_code should be 0")

        # Seq integrity
        seqs = [e["seq"] for e in events]
        _assert(seqs == list(range(1, len(events) + 1)),
                f"seq should be 1..N without gaps: {seqs}")

    print("  PASS approved_on_first_pass_emits_clean_flow")


async def test_changes_needed_then_fresh_eyes_approves() -> None:
    """CHANGES_NEEDED → fix → APPROVED (inner>1) → fresh-eyes APPROVED on outer 2."""
    with tempfile.TemporaryDirectory() as tmp:
        logs_dir = Path(tmp)
        schedule = "|".join([
            "VERDICT: CHANGES_NEEDED",         # outer1 inner1: review
            "fix_noop_ok",                     # outer1 inner1: fix
            "VERDICT: APPROVED",               # outer1 inner2: review
            "VERDICT: APPROVED",               # outer2 inner1: fresh-eyes review
        ])
        rc = await _run_orchestration(logs_dir, schedule=schedule, max_outer=3)
        _assert(rc == 0, f"expected rc=0 after fresh-eyes approval, got {rc}")

        run_dir = logs_dir / "orchestration-test-run"
        events = _read_events(run_dir / "events.jsonl")
        state = _read_state(run_dir / "state.json")

        types = [e["type"] for e in events]

        # Fix loop must have fired
        _assert(types.count("fix_started") == 1, f"fix_started count: {types.count('fix_started')}")
        _assert(types.count("fix_finished") == 1, f"fix_finished count: {types.count('fix_finished')}")
        # Three reviews: 2 in outer 1 + 1 fresh-eyes in outer 2
        _assert(types.count("review_started") == 3,
                f"expected 3 review_started, got {types.count('review_started')}")
        _assert(types.count("review_finished") == 3,
                f"expected 3 review_finished, got {types.count('review_finished')}")

        verdicts = [e["verdict"] for e in events if e["type"] == "review_finished"]
        _assert(verdicts == ["CHANGES_NEEDED", "APPROVED", "APPROVED"],
                f"review verdict sequence: {verdicts}")

        # Ordering sanity: fix comes between first two reviews
        rf_positions = [i for i, t in enumerate(types) if t == "review_finished"]
        fs_position = types.index("fix_started")
        _assert(rf_positions[0] < fs_position < rf_positions[1],
                f"fix_started must sit between reviews: reviews@{rf_positions}, fix@{fs_position}")

        # Outer iteration count: 2 outer_started, 2 outer_finished
        _assert(types.count("outer_started") == 2,
                f"expected 2 outer_started, got {types.count('outer_started')}")
        _assert(types.count("outer_finished") == 2,
                f"expected 2 outer_finished, got {types.count('outer_finished')}")

        # Final snapshot should reflect approval
        _assert(state["approved"] is True, "state.approved should be True")
        _assert(state["last_verdict"] == "APPROVED",
                f"state.last_verdict should be APPROVED, got {state['last_verdict']}")
        _assert(state["total_reviews"] == 3,
                f"state.total_reviews should be 3, got {state['total_reviews']}")
        _assert(state["finished"] is True, "state.finished should be True")
        _assert(state["phase"] == "finished", f"state.phase should be 'finished'")

    print("  PASS changes_needed_then_fresh_eyes_approves")


async def _run_crash_scenario(
    logs_dir: Path, exc: BaseException,
) -> tuple[type[BaseException], Path]:
    """Run ``review_loop.main`` with ``_run_loop`` swapped for a raiser.

    Returns the actual exception type that propagated plus the per-run
    directory so the caller can inspect ``state.json`` / ``index.jsonl``.
    """
    os.chmod(MOCK_AGENT, os.stat(MOCK_AGENT).st_mode | stat.S_IEXEC)
    with tempfile.NamedTemporaryFile("w", suffix="_mockstate") as state_file:
        state_file.write("")
        state_file.flush()

        env_backup = os.environ.copy()
        os.environ["AGENT_CMD"] = str(MOCK_AGENT)
        os.environ["MOCK_STATE_FILE"] = state_file.name
        os.environ["MOCK_SCHEDULE"] = "VERDICT: APPROVED"
        os.environ["NO_COLOR"] = "1"

        # Swap the loop for a raiser so we hit the except BaseException
        # path in review_loop.main without needing a real crash.
        orig_run_loop = review_loop._run_loop

        async def _raiser(cfg, logger):  # type: ignore[unused-argument]
            raise exc

        review_loop._run_loop = _raiser
        try:
            raised: type[BaseException] | None = None
            try:
                await review_loop.main([
                    "--prompt", "test-task",
                    "--skip-impl",
                    "--max-outer", "1",
                    "--max-inner", "1",
                    "--workspace", "/tmp",
                    "--logs-dir", str(logs_dir),
                    "--run-id", "crash-test-run",
                ])
            except BaseException as propagated:
                raised = type(propagated)
        finally:
            review_loop._run_loop = orig_run_loop
            os.environ.clear()
            os.environ.update(env_backup)

    assert raised is not None, "expected the crash to propagate out of main()"
    return raised, logs_dir / "crash-test-run"


async def test_crash_records_real_exit_code_not_invented_two() -> None:
    """On an unhandled exception, ``state.json`` and ``index.jsonl`` must
    record the exit code Python will actually produce — never a
    hard-coded ``2``.

    Covers every crash class the main entrypoint cares about:

    * Generic ``RuntimeError``  → Python exits ``1``.
    * ``KeyboardInterrupt``    → Python exits ``130`` (128 + SIGINT).
    * ``SystemExit(17)``       → Python honours the explicit code.
    * ``SystemExit(-1)``       → POSIX truncates to ``255``
      (``-1 & 0xFF``). Verified empirically with ``python -c
      "raise SystemExit(-1)"``.
    * ``SystemExit(300)``      → POSIX truncates to ``44``
      (``300 & 0xFF``). Recording the raw ``300`` here would let
      ``state.json`` disagree with the parent shell's ``$?``.
    * ``SystemExit(0.0)``      → non-int code: Python prints it and
      exits ``1`` (not ``0`` — float is not honoured).
    """
    cases = [
        (RuntimeError("boom"), 1, "RuntimeError"),
        (KeyboardInterrupt(), 130, "KeyboardInterrupt"),
        (SystemExit(17), 17, "SystemExit"),
        (SystemExit(-1), 255, "SystemExit"),
        (SystemExit(300), 44, "SystemExit"),
        (SystemExit(0.0), 1, "SystemExit"),
    ]
    for exc, expected_code, expected_type in cases:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            raised, run_dir = await _run_crash_scenario(logs_dir, exc)
            _assert(raised is type(exc),
                    f"expected {type(exc).__name__} to propagate, got {raised}")

            state = _read_state(run_dir / "state.json")
            index = _read_index(logs_dir / "index.jsonl")

            _assert(state["finished"] is True,
                    f"state.finished should be True after crash ({expected_type})")
            _assert(state["approved"] is False,
                    f"crash must leave approved=False ({expected_type})")
            _assert(state["exit_code"] == expected_code,
                    f"state.exit_code must reflect real OS exit "
                    f"(expected {expected_code}, got {state['exit_code']}, "
                    f"exc={expected_type})")
            _assert(state["error"] is not None,
                    f"state.error should be populated on crash ({expected_type})")
            _assert(state["error"]["type"] == expected_type,
                    f"state.error.type should be '{expected_type}', "
                    f"got {state['error']}")

            _assert(index[-1]["type"] == "run_finished",
                    f"index must still record run_finished on crash ({expected_type})")
            _assert(index[-1]["exit_code"] == expected_code,
                    f"index.run_finished.exit_code must match real exit "
                    f"(expected {expected_code}, got {index[-1]['exit_code']}, "
                    f"exc={expected_type})")
            _assert(index[-1]["approved"] is False,
                    f"index.run_finished.approved should be False on crash "
                    f"({expected_type})")
    print("  PASS crash_records_real_exit_code_not_invented_two")


def test_expected_exit_code_for_matches_posix_exit_status() -> None:
    """``_expected_exit_code_for`` must return what the parent shell's
    ``$?`` will show after the exception propagates — not the raw
    source-level code.

    POSIX truncates C-level exit statuses to 8 bits via
    ``WEXITSTATUS``. Every expected value below was reproduced with
    ``python -c "raise ..."; echo $?`` on Linux; see the docstring on
    :func:`review_loop._expected_exit_code_for` for the full table.

    Covered classes:

    * ``None`` → ``0``.
    * Plain ``int`` → ``code & 0xFF``, including negative values and
      values above 255. This is the fix for the finding that raw
      ``SystemExit(-1)`` / ``SystemExit(300)`` were being recorded
      verbatim.
    * ``bool`` (subclass of ``int``) → behaves like the underlying int.
    * Non-int payloads (``str``, ``float``, tuple, …) → ``1``.
    * ``KeyboardInterrupt`` → ``130``.
    * Any other exception → ``1``.
    """
    fn = review_loop._expected_exit_code_for

    cases: list[tuple[BaseException, int, str]] = [
        # SystemExit(None) / SystemExit(0): the no-op clean exit.
        (SystemExit(None), 0, "SystemExit(None)"),
        (SystemExit(0), 0, "SystemExit(0)"),
        # Plain positive ints that don't need truncation.
        (SystemExit(1), 1, "SystemExit(1)"),
        (SystemExit(2), 2, "SystemExit(2)"),
        (SystemExit(17), 17, "SystemExit(17)"),
        (SystemExit(255), 255, "SystemExit(255)"),
        # Negative / out-of-range: POSIX truncates.
        (SystemExit(-1), 255, "SystemExit(-1) [POSIX: -1 & 0xFF = 255]"),
        (SystemExit(-3), 253, "SystemExit(-3) [POSIX: -3 & 0xFF = 253]"),
        (SystemExit(256), 0, "SystemExit(256) [POSIX: 256 & 0xFF = 0]"),
        (SystemExit(-256), 0, "SystemExit(-256)"),
        (SystemExit(300), 44, "SystemExit(300) [POSIX: 300 & 0xFF = 44]"),
        (SystemExit(999999), 63, "SystemExit(999999)"),
        # bool subclass of int.
        (SystemExit(True), 1, "SystemExit(True)"),
        (SystemExit(False), 0, "SystemExit(False)"),
        # Non-int payloads: Python prints + exits 1.
        (SystemExit("oops"), 1, "SystemExit('oops')"),
        (SystemExit(0.0), 1, "SystemExit(0.0)"),
        (SystemExit(2.5), 1, "SystemExit(2.5)"),
        (SystemExit((1, 2)), 1, "SystemExit((1,2))"),
        # Signal + generic paths.
        (KeyboardInterrupt(), 130, "KeyboardInterrupt"),
        (RuntimeError("boom"), 1, "RuntimeError"),
        (ValueError("nope"), 1, "ValueError"),
    ]
    for exc, expected, label in cases:
        got = fn(exc)
        _assert(
            got == expected,
            f"_expected_exit_code_for({label}) must return {expected}, got {got}",
        )
    print("  PASS expected_exit_code_for_matches_posix_exit_status")


async def test_crash_after_progress_preserves_live_counters() -> None:
    """A mid-loop crash must not overwrite already-emitted progress
    counters with stale zeros from ``main()``'s uninitialized locals.

    Reviewer reproducer: swap ``_run_loop()`` for one that emits real
    ``outer_started`` / ``inner_started`` / ``review_started`` /
    ``fix_started`` / … events and *then* raises ``RuntimeError``.
    Before this fix, the ``except`` branch in ``main()`` called
    ``logger.finish(total_reviews=0, outer_loops=0)`` because those
    locals only got assigned on the happy-return path. That overwrote
    the already-correct live values in both ``state.json`` (via the
    ``run_finished`` event handler) and in the ``index.jsonl`` record.

    The fix keeps the structured logs as the authoritative
    terminal-state source: the crash handler no longer passes the
    stale locals, and ``RunLogger.finish()`` falls back to its live
    in-memory state (``state['total_reviews']`` / ``state['outer']``)
    — which was driven by the exact same event stream that produced
    ``events.jsonl``. Result: the final snapshot reflects what the
    loop really did before crashing.
    """
    os.chmod(MOCK_AGENT, os.stat(MOCK_AGENT).st_mode | stat.S_IEXEC)
    with tempfile.TemporaryDirectory() as tmp:
        logs_dir = Path(tmp)

        # Emit a realistic progress trail: outer 1, two inner reviews
        # (first CHANGES_NEEDED + fix, second starts its review), then
        # raise before the second review can finish. This matches the
        # classic "agent died mid-run after real work was done".
        async def _raise_after_progress(cfg, logger):  # type: ignore[unused-argument]
            logger.outer_started(outer=1, max_outer=cfg.max_outer)
            logger.inner_started(
                outer=1, inner=1, max_inner=cfg.max_inner, tag="[O1 I1]",
            )
            logger.review_started(model="m-rev", tag="[O1 I1]")
            logger.review_finished(
                verdict="CHANGES_NEEDED", rc=1, tag="[O1 I1]",
            )
            logger.fix_started(model="m-fix", tag="[O1 I1]")
            logger.fix_finished(rc=0, tag="[O1 I1]")
            logger.inner_finished(
                outer=1, inner=1, verdict="CHANGES_NEEDED",
            )
            logger.inner_started(
                outer=1, inner=2, max_inner=cfg.max_inner, tag="[O1 I2]",
            )
            logger.review_started(model="m-rev", tag="[O1 I2]")
            raise RuntimeError("boom mid-loop after progress")

        env_backup = os.environ.copy()
        os.environ["AGENT_CMD"] = str(MOCK_AGENT)
        # The mock agent isn't actually invoked on this path (our
        # replacement ``_run_loop`` raises before any step runs), but
        # ``review_loop.main`` still validates ``cfg.agent_cmd`` exists.
        os.environ["MOCK_STATE_FILE"] = "/tmp/_mock_state_crash_preserve"
        Path("/tmp/_mock_state_crash_preserve").write_text("")
        os.environ["MOCK_SCHEDULE"] = "VERDICT: APPROVED"
        os.environ["NO_COLOR"] = "1"

        orig_run_loop = review_loop._run_loop
        review_loop._run_loop = _raise_after_progress
        try:
            raised: type[BaseException] | None = None
            try:
                await review_loop.main([
                    "--prompt", "test-task",
                    "--skip-impl",
                    "--max-outer", "3",
                    "--max-inner", "3",
                    "--workspace", "/tmp",
                    "--logs-dir", str(logs_dir),
                    "--run-id", "crash-preserves-progress",
                ])
            except RuntimeError as exc:
                raised = type(exc)
        finally:
            review_loop._run_loop = orig_run_loop
            os.environ.clear()
            os.environ.update(env_backup)
            Path("/tmp/_mock_state_crash_preserve").unlink(missing_ok=True)

        _assert(raised is RuntimeError,
                f"expected RuntimeError to propagate, got {raised}")

        run_dir = logs_dir / "crash-preserves-progress"
        state = _read_state(run_dir / "state.json")
        index = _read_index(logs_dir / "index.jsonl")
        events = _read_events(run_dir / "events.jsonl")

        # Live counters survived the crash: two review_started emits and
        # the first outer loop were already in the event stream before
        # the raise, so both state.json and index.jsonl must reflect
        # them — not the stale zeros the old crash path wrote.
        _assert(state["total_reviews"] == 2,
                f"state.total_reviews must reflect live progress "
                f"(expected 2, got {state['total_reviews']})")
        _assert(state["outer"] == 1,
                f"state.outer must reflect the outer loop in progress "
                f"(expected 1, got {state['outer']})")
        _assert(state["inner"] == 2,
                f"state.inner must reflect the inner loop in progress "
                f"(expected 2, got {state['inner']})")
        _assert(state["last_verdict"] == "CHANGES_NEEDED",
                f"state.last_verdict must reflect the pre-crash review "
                f"(expected CHANGES_NEEDED, got {state['last_verdict']})")
        _assert(state["finished"] is True,
                "state.finished should be True after crash handler runs")
        _assert(state["approved"] is False,
                "state.approved must stay False after crash")
        _assert(state["exit_code"] == 1,
                f"state.exit_code must be 1 for RuntimeError, "
                f"got {state['exit_code']}")
        _assert(state["error"] is not None,
                "state.error must be populated on crash")
        _assert(state["error"]["type"] == "RuntimeError",
                f"state.error.type must be RuntimeError, got {state['error']}")

        # Root index.jsonl must carry the same live counters — the
        # "discoverability without walking per-run dirs" contract
        # breaks if the root summary disagrees with per-run state.
        idx_finish = index[-1]
        _assert(idx_finish["type"] == "run_finished",
                f"last index entry must be run_finished, got {idx_finish['type']}")
        _assert(idx_finish["total_reviews"] == 2,
                f"index.run_finished.total_reviews must reflect live "
                f"progress (expected 2, got {idx_finish['total_reviews']})")
        _assert(idx_finish["outer_loops"] == 1,
                f"index.run_finished.outer_loops must reflect the outer "
                f"loop that actually ran (expected 1, "
                f"got {idx_finish['outer_loops']})")
        _assert(idx_finish["approved"] is False,
                "index.run_finished.approved must be False on crash")
        _assert(idx_finish["exit_code"] == 1,
                f"index.run_finished.exit_code must match OS exit "
                f"(expected 1, got {idx_finish['exit_code']})")

        # The event stream itself must be complete: every progress
        # event we emitted before the raise is present, plus the
        # run_error + run_finished that close out the run.
        types = [e["type"] for e in events]
        for required in [
            "run_started",
            "outer_started",
            "inner_started",
            "review_started",
            "review_finished",
            "fix_started",
            "fix_finished",
            "inner_finished",
            "run_error",
            "run_finished",
        ]:
            _assert(required in types,
                    f"missing event after mid-loop crash: {required}\n{types}")

        # Two review_started emits before the raise.
        _assert(types.count("review_started") == 2,
                f"expected 2 review_started events, got {types.count('review_started')}")

        # run_error must precede run_finished so a consumer tailing
        # events.jsonl sees the crash diagnosis before the terminal
        # summary.
        _assert(
            types.index("run_error") < types.index("run_finished"),
            "run_error must be emitted before run_finished",
        )

    print("  PASS crash_after_progress_preserves_live_counters")


async def test_banner_broken_pipe_still_records_full_lifecycle() -> None:
    """A ``BrokenPipeError`` from ``banner()`` must NOT orphan the
    ``run_started`` record — the structured log must remain the
    authoritative source of run status regardless of whether stdout is
    still writable.

    Reviewer reproducer: forced ``review_loop.banner`` to raise
    ``BrokenPipeError``. With the old ordering, ``logger.start()``
    ran *after* ``banner()``, so the exception propagated before any
    ``run_started`` line could reach ``index.jsonl`` / ``events.jsonl``.
    The run directory existed (from the constructor) but a TUI
    polling the root index saw nothing — the run was invisible to
    downstream consumers.

    The fix moves ``logger.start()`` ahead of any human-facing output
    and wraps the banner block in ``except OSError`` so a broken pipe
    is swallowed. A clean mock run now produces:

    * ``index.jsonl``: ``run_started`` + ``run_finished`` (both
      present and well-formed).
    * ``state.json``: terminal ``finished=True, approved=True,
      exit_code=0``.
    * The whole orchestration event stream in ``events.jsonl``,
      unchanged by the console failure.
    """
    os.chmod(MOCK_AGENT, os.stat(MOCK_AGENT).st_mode | stat.S_IEXEC)
    with tempfile.TemporaryDirectory() as tmp, _preserve_stdout():
        logs_dir = Path(tmp)

        def boom_banner(title, items):  # type: ignore[unused-argument]
            raise BrokenPipeError("stdout closed at banner()")

        orig_banner = review_loop.banner
        review_loop.banner = boom_banner

        with tempfile.NamedTemporaryFile("w", suffix="_mockstate") as state_file:
            state_file.write("")
            state_file.flush()
            env_backup = os.environ.copy()
            os.environ["AGENT_CMD"] = str(MOCK_AGENT)
            os.environ["MOCK_STATE_FILE"] = state_file.name
            os.environ["MOCK_SCHEDULE"] = "VERDICT: APPROVED"
            os.environ["NO_COLOR"] = "1"
            try:
                rc = await review_loop.main([
                    "--prompt", "test-task",
                    "--skip-impl",
                    "--max-outer", "1",
                    "--max-inner", "1",
                    "--workspace", "/tmp",
                    "--logs-dir", str(logs_dir),
                    "--run-id", "banner-broken-pipe",
                ])
            finally:
                review_loop.banner = orig_banner
                os.environ.clear()
                os.environ.update(env_backup)

        _assert(rc == 0, f"clean approval should still return rc=0, got {rc}")

        run_dir = logs_dir / "banner-broken-pipe"
        index = _read_index(logs_dir / "index.jsonl")
        events = _read_events(run_dir / "events.jsonl")
        state = _read_state(run_dir / "state.json")

        index_types = [e["type"] for e in index]
        _assert(index_types == ["run_started", "run_finished"],
                f"index must carry full lifecycle even when banner() "
                f"raises: got {index_types}")
        _assert(index[-1]["exit_code"] == 0,
                f"index.run_finished.exit_code should be 0, got {index[-1]['exit_code']}")

        event_types = [e["type"] for e in events]
        _assert("run_started" in event_types,
                f"run_started missing from events.jsonl after banner "
                f"failure: {event_types}")
        _assert("run_finished" in event_types,
                f"run_finished missing from events.jsonl after banner "
                f"failure: {event_types}")
        _assert("review_started" in event_types and "review_finished" in event_types,
                f"orchestration events missing after banner failure: "
                f"{event_types}")

        _assert(state["finished"] is True,
                "state.finished must be True after run completes despite "
                "banner failure")
        _assert(state["approved"] is True,
                "state.approved must be True after clean approval")
        _assert(state["exit_code"] == 0,
                f"state.exit_code should be 0, got {state['exit_code']}")
        _assert(state["phase"] == "finished",
                f"state.phase should be 'finished', got {state['phase']}")
    print("  PASS banner_broken_pipe_still_records_full_lifecycle")


async def test_summary_broken_pipe_still_records_run_finished() -> None:
    """A ``BrokenPipeError`` from ``summary()`` at shutdown must NOT
    orphan the ``run_finished`` record.

    Reviewer reproducer: forced ``review_loop.summary`` to raise
    ``BrokenPipeError``. With the old ordering, ``logger.finish()``
    ran *after* ``summary()``, so a crashing summary left
    ``index.jsonl`` with only ``run_started`` and ``state.json``
    stuck at ``finished=false`` forever. A future consumer polling
    the index would believe the run was still live.

    The fix records ``run_finished`` *before* the summary block and
    wraps the summary call in ``except OSError`` so the failure is
    swallowed. The run still returns the correct exit code and all
    terminal state is durable in both the per-run snapshot and the
    root index before stdout is touched again.
    """
    os.chmod(MOCK_AGENT, os.stat(MOCK_AGENT).st_mode | stat.S_IEXEC)
    with tempfile.TemporaryDirectory() as tmp, _preserve_stdout():
        logs_dir = Path(tmp)

        def boom_summary(**kwargs):  # type: ignore[unused-argument]
            raise BrokenPipeError("stdout closed at summary()")

        orig_summary = review_loop.summary
        review_loop.summary = boom_summary

        with tempfile.NamedTemporaryFile("w", suffix="_mockstate") as state_file:
            state_file.write("")
            state_file.flush()
            env_backup = os.environ.copy()
            os.environ["AGENT_CMD"] = str(MOCK_AGENT)
            os.environ["MOCK_STATE_FILE"] = state_file.name
            os.environ["MOCK_SCHEDULE"] = "VERDICT: APPROVED"
            os.environ["NO_COLOR"] = "1"
            try:
                rc = await review_loop.main([
                    "--prompt", "test-task",
                    "--skip-impl",
                    "--max-outer", "1",
                    "--max-inner", "1",
                    "--workspace", "/tmp",
                    "--logs-dir", str(logs_dir),
                    "--run-id", "summary-broken-pipe",
                ])
            finally:
                review_loop.summary = orig_summary
                os.environ.clear()
                os.environ.update(env_backup)

        _assert(rc == 0, f"summary failure should not change rc=0, got {rc}")

        run_dir = logs_dir / "summary-broken-pipe"
        index = _read_index(logs_dir / "index.jsonl")
        state = _read_state(run_dir / "state.json")

        index_types = [e["type"] for e in index]
        _assert(index_types == ["run_started", "run_finished"],
                f"index must still record run_finished when summary() "
                f"raises: got {index_types}")
        _assert(index[-1]["approved"] is True,
                "index.run_finished.approved must stay True")
        _assert(index[-1]["exit_code"] == 0,
                "index.run_finished.exit_code must be 0")

        _assert(state["finished"] is True,
                "state.finished must be True despite summary failure")
        _assert(state["approved"] is True,
                "state.approved must be True for a clean approval")
        _assert(state["exit_code"] == 0,
                f"state.exit_code should be 0, got {state['exit_code']}")
        _assert(state["phase"] == "finished",
                f"state.phase should be 'finished', got {state['phase']}")
        _assert(state["last_verdict"] == "APPROVED",
                f"state.last_verdict should be APPROVED, got "
                f"{state['last_verdict']}")
    print("  PASS summary_broken_pipe_still_records_run_finished")


async def test_started_at_consistent_between_index_and_state_end_to_end() -> None:
    """End-to-end verification that ``index.jsonl.run_started.started_at``
    equals ``state.json.started_at`` exactly.

    The unit test in ``test_run_log.py`` covers the ``RunLogger.start``
    contract directly; this higher-level test proves the CLI wiring
    in ``review_loop.main`` (``banner_items().started_at`` →
    ``start_config['started_at']``) doesn't introduce a drift between
    the two surfaces on a live run. Without the fix a ~2ms gap was
    visible on every clean run because the index used the
    constructor-time default while state was overwritten from the
    config.
    """
    os.chmod(MOCK_AGENT, os.stat(MOCK_AGENT).st_mode | stat.S_IEXEC)
    with tempfile.TemporaryDirectory() as tmp:
        logs_dir = Path(tmp)
        rc = await _run_orchestration(
            logs_dir,
            schedule="VERDICT: APPROVED",
        )
        _assert(rc == 0, f"clean run should return rc=0, got {rc}")

        run_dir = logs_dir / "orchestration-test-run"
        state = _read_state(run_dir / "state.json")
        index = _read_index(logs_dir / "index.jsonl")
        run_started = next(e for e in index if e["type"] == "run_started")

        _assert(run_started["started_at"] == state["started_at"],
                f"index.run_started.started_at must match "
                f"state.started_at exactly "
                f"(index={run_started['started_at']!r}, "
                f"state={state['started_at']!r})")
        # And both must match the banner-time value that flowed through
        # ``start_config['started_at']`` — which is what the human sees
        # in the banner line, so TUI and banner agree too.
        _assert(state["config"]["started_at"] == state["started_at"],
                f"state.config.started_at must match state.started_at "
                f"(config={state['config']['started_at']!r}, "
                f"state={state['started_at']!r})")
    print("  PASS started_at_consistent_between_index_and_state_end_to_end")


def test_review_loop_piped_to_head_exits_match_state_exit_code() -> None:
    """End-to-end: ``review-loop.py ... | head -0`` must exit with the
    same status the structured log records.

    Reviewer's Round 11 reproducer: piping to ``head -0`` closed the
    parent shell's read end immediately. Before the fix, two
    things broke:

    1. The first ``log()`` / ``warn()`` / ``section()`` call after
       the pipe closed raised ``BrokenPipeError``, aborting the
       loop before any review step could run. Downstream consumers
       polling the structured log saw only ``run_started`` — no
       ``run_finished``, no verdict, no progress.
    2. Even if we caught that, Python's interpreter-shutdown flush
       of ``sys.stdout`` hit the same broken pipe and exited
       **120**. ``state.json.exit_code`` then said ``1`` while
       the parent shell's ``$?`` said ``120`` — a direct violation
       of the "structured log is the source of truth" contract.

    The fix (routing every print through ``_safe_print``, which
    swaps fd 1 to ``/dev/null`` on the first failure) addresses
    both: the loop completes normally and the chosen exit code
    survives interpreter shutdown. This test locks both in by
    comparing the parent shell's ``$?`` to ``state.exit_code`` for
    both approval and exhaustion paths.

    Implementation note: this runs in a real subprocess because
    the bug only manifested at true process exit — in-process
    tests can't exercise the interpreter-shutdown flush path.
    """
    import subprocess
    os.chmod(MOCK_AGENT, os.stat(MOCK_AGENT).st_mode | stat.S_IEXEC)

    cases = [
        ("VERDICT: APPROVED", 0, "clean approval"),
        ("VERDICT: CHANGES_NEEDED", 1, "exhaustion"),
    ]
    for schedule, expected_rc, label in cases:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            mock_state = Path(tmp) / "mock_state"
            mock_state.write_text("")

            env = {
                **os.environ,
                "AGENT_CMD": str(MOCK_AGENT),
                "MOCK_STATE_FILE": str(mock_state),
                "MOCK_SCHEDULE": schedule,
                "NO_COLOR": "1",
            }
            # ``${PIPESTATUS[0]}`` is the exit code of the first
            # command in the pipeline — i.e. review-loop.py
            # itself — regardless of ``head -0``'s exit. This is
            # what a TUI/CI parent process would read to decide
            # whether the run succeeded.
            cmd = (
                f"{sys.executable} {REPO_ROOT}/review-loop.py "
                f"--prompt test --skip-impl "
                f"--max-outer 1 --max-inner 1 "
                f"--workspace /tmp "
                f"--logs-dir {logs_dir} "
                f"--run-id pipe-head-e2e "
                f"| head -0; exit ${{PIPESTATUS[0]}}"
            )
            result = subprocess.run(
                ["bash", "-c", cmd],
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
            )
            shell_rc = result.returncode

            run_dir = logs_dir / "pipe-head-e2e"
            state = _read_state(run_dir / "state.json")
            index = _read_index(logs_dir / "index.jsonl")
            events = _read_events(run_dir / "events.jsonl")

            # Core contract: structured log == OS reality.
            _assert(
                shell_rc == state["exit_code"],
                f"[{label}] shell rc ({shell_rc}) must match "
                f"state.exit_code ({state['exit_code']}); "
                f"stderr={result.stderr!r}",
            )
            # The specific 120 regression.
            _assert(
                shell_rc != 120,
                f"[{label}] shell rc=120 means the fd-1 swap did not "
                f"happen before interpreter shutdown (the original "
                f"repro); stderr={result.stderr!r}",
            )
            _assert(
                shell_rc == expected_rc,
                f"[{label}] expected rc={expected_rc}, got {shell_rc}; "
                f"stderr={result.stderr!r}",
            )
            # Loop completed despite the broken pipe. Before the
            # ``_safe_print`` fix, the first ``log()`` call aborted
            # the loop and ``state.finished`` stayed False.
            _assert(
                state["finished"] is True,
                f"[{label}] state.finished must be True after loop "
                f"completes; state={state}",
            )
            _assert(
                state["phase"] == "finished",
                f"[{label}] state.phase should be 'finished', got "
                f"{state['phase']!r}",
            )
            # Full lifecycle in the root index — a consumer
            # enumerating runs from ``index.jsonl`` must still see
            # this one as completed.
            index_types = [e["type"] for e in index]
            _assert(
                index_types == ["run_started", "run_finished"],
                f"[{label}] index must have full lifecycle even "
                f"with broken stdout: got {index_types}",
            )
            _assert(
                index[-1]["exit_code"] == expected_rc,
                f"[{label}] index.run_finished.exit_code must be "
                f"{expected_rc}, got {index[-1]['exit_code']}",
            )
            # The whole review was still emitted into
            # ``events.jsonl`` even though console output was
            # going nowhere. Before the fix, the mid-loop
            # ``log()`` call raised before ``review_started``
            # ever reached disk.
            event_types = [e["type"] for e in events]
            _assert(
                "review_started" in event_types and
                "review_finished" in event_types,
                f"[{label}] orchestration events missing from "
                f"events.jsonl under broken stdout: {event_types}",
            )
    print("  PASS review_loop_piped_to_head_exits_match_state_exit_code")


async def test_not_approved_exits_nonzero_and_records_terminal_state() -> None:
    """Every review returns CHANGES_NEEDED → loop exhausts, rc=1, state reflects it."""
    with tempfile.TemporaryDirectory() as tmp:
        logs_dir = Path(tmp)
        # Plenty of CHANGES_NEEDED to keep the inner loop churning
        schedule = "|".join(
            ["VERDICT: CHANGES_NEEDED", "fix_ok"] * 10
            + ["VERDICT: CHANGES_NEEDED"] * 10
        )
        rc = await _run_orchestration(
            logs_dir, schedule=schedule, max_outer=2, max_inner=2,
        )
        _assert(rc == 1, f"expected rc=1 after exhaustion, got {rc}")

        run_dir = logs_dir / "orchestration-test-run"
        state = _read_state(run_dir / "state.json")
        index = _read_index(logs_dir / "index.jsonl")

        _assert(state["finished"] is True, "state.finished should be True after exhaustion")
        _assert(state["approved"] is False, "state.approved should be False after exhaustion")
        _assert(state["exit_code"] == 1, "state.exit_code should be 1")
        _assert(state["last_verdict"] == "CHANGES_NEEDED",
                f"state.last_verdict should be CHANGES_NEEDED, got {state['last_verdict']}")

        _assert(index[-1]["type"] == "run_finished", "index final entry must be run_finished")
        _assert(index[-1]["approved"] is False, "index.run_finished.approved should be False")
        _assert(index[-1]["exit_code"] == 1, "index.run_finished.exit_code should be 1")

    print("  PASS not_approved_exits_nonzero_and_records_terminal_state")


async def _main() -> None:
    print("=" * 60)
    print("Orchestration-level RunLogger tests")
    print("-" * 60)
    test_expected_exit_code_for_matches_posix_exit_status()
    await test_approved_on_first_pass_emits_clean_flow()
    await test_changes_needed_then_fresh_eyes_approves()
    await test_not_approved_exits_nonzero_and_records_terminal_state()
    await test_crash_records_real_exit_code_not_invented_two()
    await test_crash_after_progress_preserves_live_counters()
    await test_banner_broken_pipe_still_records_full_lifecycle()
    await test_summary_broken_pipe_still_records_run_finished()
    await test_started_at_consistent_between_index_and_state_end_to_end()
    test_review_loop_piped_to_head_exits_match_state_exit_code()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(_main())
