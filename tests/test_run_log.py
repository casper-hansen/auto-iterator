"""Focused unit tests for the :mod:`iterator_loop.run_log` writer.

Covers the invariants reviewers care about most:

* ``events.jsonl`` is append-only, ordered by ``seq``, and one JSON object
  per line (grep-able, tail-able).
* ``state.json`` is rewritten atomically on every event (no ``.tmp``
  leftover, always valid JSON), reflects the latest snapshot, and carries
  the declared schema.
* ``logs/index.jsonl`` gets both ``run_started`` and ``run_finished`` so
  an operator can enumerate runs without walking each per-run directory.
* The rolling ``tail`` caps at ``tail_size`` lines.
* Stream events from an agent sink (``tool_call_started`` /
  ``_completed``, ``assistant_partial``, ``agent_result``) round-trip
  into both ``events.jsonl`` and ``state.json``.
* Unknown event types are tolerated without breaking the schema.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iterator_loop.run_log import (  # noqa: E402
    EVENTS_FILENAME,
    INDEX_FILENAME,
    PHASE_FINISHED,
    PHASE_INIT,
    PHASE_REVIEW,
    SCHEMA_VERSION,
    STATE_FILENAME,
    InvalidRunIdError,
    RunLogger,
    new_run_id,
)


def _read_events(path: Path) -> list[dict]:
    """Return ``events.jsonl`` as a list of dicts, one per line."""
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def _read_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_index(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def _assert(cond: bool, msg: str) -> None:
    assert cond, msg


def test_new_run_id_is_sortable_and_unique() -> None:
    a = new_run_id()
    b = new_run_id()
    _assert(a != b, "run ids should be unique per call")
    _assert("T" in a and a.endswith(a.split("-")[-1]), "unexpected run id format")
    _assert(len(a) >= 18, f"run id too short: {a!r}")
    print("  PASS new_run_id_is_sortable_and_unique")


def test_logger_creates_initial_files() -> None:
    """Constructor alone must leave a readable events.jsonl + state.json
    so a concurrent TUI/reader cannot race the first emit."""
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)
        _assert(logger.events_path.exists(), "events.jsonl not created on __init__")
        _assert(logger.state_path.exists(), "state.json not created on __init__")
        state = _read_state(logger.state_path)
        _assert(state["schema_version"] == SCHEMA_VERSION, "schema_version missing/bad")
        _assert(state["run_id"] == logger.run_id, "run_id mismatch in initial state")
        _assert(state["phase"] == PHASE_INIT, "initial phase should be 'init'")
        _assert(state["finished"] is False, "fresh state must not be finished")
        _assert(state["tail"] == [], "tail should start empty")
        # Running + last-sample + observation counter must all start as
        # "not yet observed" so a TUI can distinguish a fresh run from one
        # that observed a literal zero.
        usage = state["usage"]
        _assert(usage["tokens"] is None, "usage.tokens must be unknown, not fabricated")
        _assert(usage["cost"] is None, "usage.cost must be unknown, not fabricated")
        _assert(usage["last_tokens"] is None, "usage.last_tokens must start unknown")
        _assert(usage["last_cost"] is None, "usage.last_cost must start unknown")
        _assert(usage["results_observed"] == 0,
                f"no results observed yet, got {usage['results_observed']}")
        _assert(state["error"] is None, "error slot must start null, not fabricated")
    print("  PASS logger_creates_initial_files")


def test_emit_appends_ordered_events_and_rewrites_state() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)
        logger.emit("phase_a", {"foo": 1})
        logger.emit("phase_b", {"bar": 2})
        logger.emit("phase_c", {})

        events = _read_events(logger.events_path)
        seqs = [e["seq"] for e in events]
        types = [e["type"] for e in events]
        _assert(seqs == [1, 2, 3], f"seq must be monotonic 1..N, got {seqs}")
        _assert(types == ["phase_a", "phase_b", "phase_c"], f"types out of order: {types}")
        _assert(events[0]["foo"] == 1 and events[1]["bar"] == 2,
                "payload fields missing from event record")

        state = _read_state(logger.state_path)
        _assert(state["last_event"]["seq"] == 3, "state.last_event.seq should track latest")
        _assert(state["last_event"]["type"] == "phase_c",
                "state.last_event.type should track latest")

        # No temp file should linger after an atomic replace.
        tmp_files = list(logger.run_dir.glob("*.tmp"))
        _assert(tmp_files == [], f"stale .tmp file(s) found: {tmp_files}")
    print("  PASS emit_appends_ordered_events_and_rewrites_state")


def test_start_and_finish_update_index_and_state() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)
        logger.start(config={
            "prompt": "do the thing",
            "context": "extra",
            "impl_model": "m-impl",
            "fix_model": "m-fix",
            "reviewer_model": "m-rev",
            "max_outer": 3,
            "max_inner": 2,
            "workspace": "/tmp/ws",
            "skip_impl": False,
            "extra_flags": ["--foo"],
            "agent_cmd": "agent",
        })
        logger.emit("review_started", {"tag": "[Outer 1, Inner 1]", "model": "m-rev"})
        logger.emit("review_finished", {"verdict": "APPROVED", "rc": 0, "tag": "[Outer 1, Inner 1]"})
        logger.finish(approved=True, exit_code=0, total_reviews=1, outer_loops=1)

        index = _read_index(Path(tmp) / INDEX_FILENAME)
        types = [e["type"] for e in index]
        _assert(types == ["run_started", "run_finished"],
                f"index.jsonl must have run_started then run_finished, got {types}")
        _assert(index[0]["run_id"] == logger.run_id, "index run_started.run_id mismatch")
        _assert(index[0]["run_dir"] == logger.run_id,
                "index run_dir should be the per-run dir name, not an absolute path")
        _assert("prompt_preview" in index[0], "index run_started should carry prompt_preview")
        _assert(index[1]["approved"] is True and index[1]["exit_code"] == 0,
                "index run_finished must mirror terminal outcome")

        state = _read_state(logger.state_path)
        _assert(state["phase"] == PHASE_FINISHED, "phase must be 'finished' after finish()")
        _assert(state["approved"] is True, "approved flag not propagated")
        _assert(state["exit_code"] == 0, "exit_code not propagated")
        _assert(state["last_verdict"] == "APPROVED", "last_verdict not captured")
        _assert(state["total_reviews"] == 1, "total_reviews should track review_started count")
        _assert(state["config"]["impl_model"] == "m-impl", "config not persisted in state")
        _assert(state["config"]["extra_flags"] == ["--foo"], "extra_flags not stringified correctly")
        _assert(state["max_outer"] == 3 and state["max_inner"] == 2,
                "max_outer/max_inner not exposed at top level")
        _assert(state["flags"]["skip_impl"] is False, "skip_impl flag not exposed")

        events = _read_events(logger.events_path)
        event_types = [e["type"] for e in events]
        # Ordered: run_started, review_started, review_finished, run_finished
        _assert(event_types == ["run_started", "review_started", "review_finished", "run_finished"],
                f"unexpected event order in events.jsonl: {event_types}")
    print("  PASS start_and_finish_update_index_and_state")


def test_tail_ring_buffer_caps_at_tail_size() -> None:
    """Once complete lines land in the ring buffer, the oldest ones must
    fall out when ``tail_size`` is reached. Each chunk here ends with
    ``\\n`` so every line is committed immediately, which is what
    realistic streaming looks like once a newline arrives."""
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp, tail_size=3)
        logger.emit("assistant_partial", {"text": "line-1\nline-2\nline-3\n"})
        logger.emit("assistant_partial", {"text": "line-4\n"})
        logger.emit("assistant_partial", {"text": "line-5\nline-6\n"})

        state = _read_state(logger.state_path)
        _assert(state["tail"] == ["line-4", "line-5", "line-6"],
                f"tail ring buffer not capped correctly: {state['tail']}")

        # Events are still complete, not clipped
        events = _read_events(logger.events_path)
        _assert(len(events) == 3, f"expected 3 events in jsonl, got {len(events)}")
    print("  PASS tail_ring_buffer_caps_at_tail_size")


def test_tool_call_events_update_pending_tools_and_last_tool() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)
        logger.emit("agent_session_started", {"model": "m1", "attempt": 0, "max_resume_attempts": 3})
        state = _read_state(logger.state_path)
        _assert(state["agent"]["session_active"] is True, "session_active flag not set")
        _assert(state["agent"]["model"] == "m1", "agent.model not captured")

        logger.emit("tool_call_started", {"name": "shell", "summary": "shell: ls"})
        state = _read_state(logger.state_path)
        _assert(state["agent"]["pending_tools"] == 1, "pending_tools not incremented")
        _assert(state["agent"]["last_tool"]["phase"] == "started", "last_tool.phase wrong")
        _assert(state["agent"]["last_tool"]["summary"] == "shell: ls",
                "last_tool.summary missing")

        logger.emit("tool_call_completed", {"name": "shell", "summary": "shell ✓ exit 0"})
        state = _read_state(logger.state_path)
        _assert(state["agent"]["pending_tools"] == 0, "pending_tools not decremented")
        _assert(state["agent"]["last_tool"]["phase"] == "completed", "last_tool.phase wrong")

        logger.emit("agent_result", {"text": "final",
                                     "usage": {"input_tokens": 10, "output_tokens": 5},
                                     "cost": 0.001})
        state = _read_state(logger.state_path)
        _assert(state["agent"]["saw_result"] is True, "saw_result not flipped on agent_result")
        _assert(state["usage"]["tokens"] == {"input_tokens": 10, "output_tokens": 5},
                "usage.tokens not captured from agent_result")
        _assert(state["usage"]["cost"] == 0.001, "usage.cost not captured from agent_result")

        logger.emit("agent_session_finished", {
            "model": "m1", "attempt": 0, "rc": 0, "rc_display": "0",
            "clean_exit": True, "saw_result": True, "pending_tools": 0,
            "stream_closed": False,
        })
        state = _read_state(logger.state_path)
        _assert(state["agent"]["session_active"] is False, "session_active not cleared")
        _assert(state["agent"]["last_rc"] == 0, "agent.last_rc not captured")
    print("  PASS tool_call_events_update_pending_tools_and_last_tool")


def test_atomic_state_write_never_leaves_tmp() -> None:
    """Drive many rapid emits; confirm state.json is always valid JSON and
    no ``.tmp`` sibling lingers."""
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)
        for i in range(100):
            logger.emit("probe", {"i": i})
            # Any observer seeing the file now must get valid JSON.
            data = _read_state(logger.state_path)
            _assert(data["last_event"]["type"] == "probe",
                    f"state.json torn at iter {i}")
            _assert(data["last_event"]["seq"] == i + 1,
                    f"seq out of sync at iter {i}: {data['last_event']['seq']}")
            tmp_files = list(logger.run_dir.glob("*.tmp"))
            _assert(tmp_files == [], f"stale .tmp after iter {i}: {tmp_files}")
    print("  PASS atomic_state_write_never_leaves_tmp")


def test_unknown_event_types_are_accepted() -> None:
    """New event types (e.g. future operator interception) must not
    break the schema."""
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)
        logger.emit("some_future_event", {"custom": "value"})

        state = _read_state(logger.state_path)
        _assert(state["last_event"]["type"] == "some_future_event",
                "last_event.type not updated for unknown event")

        events = _read_events(logger.events_path)
        _assert(events[-1]["custom"] == "value",
                "unknown event payload fields must still round-trip")
    print("  PASS unknown_event_types_are_accepted")


def test_guidance_received_sets_pending_guidance() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)
        logger.guidance_received(text="Focus on tests", source="operator")

        state = _read_state(logger.state_path)
        _assert(state["pending_guidance"] is not None, "pending_guidance not set")
        _assert(state["pending_guidance"]["text"] == "Focus on tests",
                "pending_guidance.text mismatch")
        _assert(state["pending_guidance"]["source"] == "operator",
                "pending_guidance.source mismatch")
    print("  PASS guidance_received_sets_pending_guidance")


def test_tail_handles_streamed_partial_deltas() -> None:
    """Byte-level partial deltas must assemble into complete lines in the
    tail — not fragments — and the final ``assistant_message`` must not
    duplicate content that the partials already covered.

    Reproduces the real Cursor CLI streaming path: a visible line like
    ``"hello\\nworld"`` arrives as a sequence of deltas (``"hel"``,
    ``"lo\\nwor"``, ``"ld"``) and is terminated by a single non-partial
    ``assistant_message`` with the whole assembled text. Before this
    was fixed, the tail contained
    ``['hel', 'lo', 'wor', 'ld', 'hello', 'world']``; now it must hold
    exactly the two visible lines with the in-flight partial shown
    live while it streams.
    """
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)

        logger.emit("assistant_partial", {"text": "hel"})
        state = _read_state(logger.state_path)
        _assert(state["tail"] == ["hel"],
                f"in-flight partial should be visible as a single streaming "
                f"line, got {state['tail']}")

        logger.emit("assistant_partial", {"text": "lo\nwor"})
        state = _read_state(logger.state_path)
        _assert(state["tail"] == ["hello", "wor"],
                f"partials should merge into complete lines; got {state['tail']}")

        logger.emit("assistant_partial", {"text": "ld"})
        state = _read_state(logger.state_path)
        _assert(state["tail"] == ["hello", "world"],
                f"trailing partial should reflect the latest streaming line, "
                f"got {state['tail']}")

        logger.emit("assistant_message", {"text": "hello\nworld"})
        state = _read_state(logger.state_path)
        _assert(state["tail"] == ["hello", "world"],
                f"final assistant_message must not duplicate streamed "
                f"content, got {state['tail']}")

        # And the raw events.jsonl is untouched — every partial + final
        # message is preserved verbatim so a consumer can replay if
        # needed.
        types = [e["type"] for e in _read_events(logger.events_path)]
        _assert(types.count("assistant_partial") == 3,
                f"all 3 partial events must be kept in events.jsonl: {types}")
        _assert(types.count("assistant_message") == 1,
                f"assistant_message must be kept in events.jsonl: {types}")
    print("  PASS tail_handles_streamed_partial_deltas")


def test_tail_handles_non_streaming_assistant_message() -> None:
    """Agents that don't stream partials (just a single
    ``assistant_message``) must still populate the tail."""
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)
        logger.emit("assistant_message", {"text": "alpha\nbeta\ngamma"})
        state = _read_state(logger.state_path)
        _assert(state["tail"] == ["alpha", "beta", "gamma"],
                f"non-streaming assistant_message must land in tail: "
                f"{state['tail']}")
    print("  PASS tail_handles_non_streaming_assistant_message")


def test_tail_does_not_duplicate_agent_result_echo() -> None:
    """The ``agent_result`` event sometimes echoes the final assistant
    text. When we've already seen ``assistant_message`` for the
    session, the result echo must not pad the tail with duplicates."""
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)
        logger.emit("agent_session_started", {"model": "m", "attempt": 0})
        logger.emit("assistant_message", {"text": "one\ntwo"})
        logger.emit("agent_result", {"text": "one\ntwo",
                                      "usage": {"input_tokens": 1},
                                      "cost": 0.0})
        state = _read_state(logger.state_path)
        _assert(state["tail"] == ["one", "two"],
                f"agent_result echo must not duplicate tail: {state['tail']}")
    print("  PASS tail_does_not_duplicate_agent_result_echo")


def test_tail_session_boundary_flushes_dangling_partial() -> None:
    """If a partial never terminated before a session dies and resumes,
    the next session must not splice its first delta onto the previous
    session's dangling buffer."""
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)
        logger.emit("agent_session_started", {"model": "m1", "attempt": 0})
        logger.emit("assistant_partial", {"text": "incomplete-first"})
        # Session dies mid-line — no closing '\n', no assistant_message.
        logger.emit("agent_session_finished", {"rc": 1, "attempt": 0})
        logger.emit("agent_session_started", {"model": "m1", "attempt": 1})
        logger.emit("assistant_partial", {"text": "next-session"})
        state = _read_state(logger.state_path)
        # The first session's dangling partial must have been committed
        # as its own line at the session boundary; the resume's delta
        # must appear as a separate streaming line.
        _assert(state["tail"] == ["incomplete-first", "next-session"],
                f"dangling partial must be flushed at session boundary, "
                f"got {state['tail']}")
    print("  PASS tail_session_boundary_flushes_dangling_partial")


def test_usage_accumulates_across_agent_results() -> None:
    """Multiple ``agent_result`` events should accumulate a *running*
    total for tokens and cost instead of discarding earlier samples.

    This matters because every implementation/review/fix session emits
    its own ``result`` event. A consumer that asks ``state.json``
    "how much did this run cost so far?" must get the sum, not just
    the last session's bill.
    """
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)
        logger.emit("agent_result", {
            "text": "",
            "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 2},
            "cost": 0.001,
        })
        logger.emit("agent_result", {
            "text": "",
            "usage": {"input_tokens": 7, "output_tokens": 3, "cache_read_tokens": 1},
            "cost": 0.0007,
        })
        logger.emit("agent_result", {
            "text": "",
            "usage": {"input_tokens": 4, "output_tokens": 2, "cache_read_tokens": 0},
            "cost": 0.0002,
        })
        state = _read_state(logger.state_path)
        u = state["usage"]
        _assert(u["tokens"] == {
            "input_tokens": 21,
            "output_tokens": 10,
            "cache_read_tokens": 3,
        }, f"usage.tokens must sum numeric counters, got {u['tokens']}")
        _assert(abs(u["cost"] - 0.0019) < 1e-12,
                f"usage.cost must sum to 0.0019, got {u['cost']}")
        _assert(u["last_tokens"] == {
            "input_tokens": 4, "output_tokens": 2, "cache_read_tokens": 0,
        }, f"usage.last_tokens should be the most recent sample, got {u['last_tokens']}")
        _assert(abs(u["last_cost"] - 0.0002) < 1e-12,
                f"usage.last_cost should be the most recent cost, got {u['last_cost']}")
        _assert(u["results_observed"] == 3,
                f"results_observed should be 3, got {u['results_observed']}")
    print("  PASS usage_accumulates_across_agent_results")


def test_usage_tolerates_missing_or_partial_fields() -> None:
    """Agents that don't report usage or cost must leave the running
    totals untouched (never fabricated)."""
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)
        # First: only usage, no cost.
        logger.emit("agent_result", {
            "text": "", "usage": {"input_tokens": 3}, "cost": None,
        })
        state = _read_state(logger.state_path)
        _assert(state["usage"]["tokens"] == {"input_tokens": 3},
                f"tokens captured on first sample, got {state['usage']['tokens']}")
        _assert(state["usage"]["cost"] is None,
                f"cost must stay None when not reported, got {state['usage']['cost']}")

        # Second: cost only, no usage.
        logger.emit("agent_result", {"text": "", "cost": 0.5})
        state = _read_state(logger.state_path)
        _assert(state["usage"]["tokens"] == {"input_tokens": 3},
                f"tokens must stay at previous total, got {state['usage']['tokens']}")
        _assert(state["usage"]["cost"] == 0.5,
                f"cost initialized from first observation, got {state['usage']['cost']}")

        # Third: nothing at all — results_observed must not tick.
        logger.emit("agent_result", {"text": "plain"})
        state = _read_state(logger.state_path)
        _assert(state["usage"]["results_observed"] == 2,
                f"results_observed should only count events with usage/cost, "
                f"got {state['usage']['results_observed']}")
    print("  PASS usage_tolerates_missing_or_partial_fields")


def test_run_error_event_populates_state_error_slot() -> None:
    """``run_error`` must surface in ``state.json['error']`` so a
    consumer can distinguish a crashed run from a clean finish without
    parsing tracebacks out of logs."""
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)
        logger.emit("run_error", {
            "error_type": "RuntimeError",
            "message": "boom",
            "traceback": "Traceback (most recent call last):\n  ...\nRuntimeError: boom",
        })
        # finish() with the real exit code the OS will observe.
        logger.finish(approved=False, exit_code=1, total_reviews=0, outer_loops=0)

        state = _read_state(logger.state_path)
        _assert(state["error"] is not None, "state.error should be populated")
        _assert(state["error"]["type"] == "RuntimeError",
                f"error.type mismatch: {state['error']}")
        _assert(state["error"]["message"] == "boom",
                f"error.message mismatch: {state['error']}")
        _assert("RuntimeError: boom" in state["error"]["traceback"],
                "error.traceback must be preserved verbatim")
        _assert(state["exit_code"] == 1,
                f"exit_code should match the real OS exit, got {state['exit_code']}")
        _assert(state["finished"] is True, "state must be terminal after finish")
    print("  PASS run_error_event_populates_state_error_slot")


def test_run_id_rejects_path_traversal_and_escapes_root() -> None:
    """An explicit ``run_id`` must never be allowed to break out of the
    configured logs root.

    Reproduces two escape paths that the old constructor accepted:

    * ``"../escape"`` — ``Path(root) / "../escape"`` resolves *outside*
      ``root`` (e.g. ``/tmp/abc/../escape`` → ``/tmp/escape``), so a
      future consumer indexing ``root/`` would never see the run.
    * ``"/abs/path"`` — Python's ``Path(root) / "/abs/path"`` is just
      ``/abs/path`` because the absolute side wins. The old
      constructor happily ``mkdir``-ed it.

    Both must now raise :class:`InvalidRunIdError` *before* any
    filesystem side effect, leaving the root untouched.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pre = set(root.iterdir())

        for bad in ["../escape", "foo/bar", "foo\\bar", "/abs/path"]:
            try:
                RunLogger(root_dir=root, run_id=bad)
            except InvalidRunIdError:
                continue
            raise AssertionError(f"run_id={bad!r} should have been rejected")

        # No stray directories / files were created outside the root by
        # any of the rejected attempts.
        post = set(root.iterdir())
        _assert(post == pre,
                f"rejected run_ids must not leave artifacts; "
                f"new entries: {post - pre}")
    print("  PASS run_id_rejects_path_traversal_and_escapes_root")


def test_run_id_rejects_reserved_and_unsafe_values() -> None:
    """Beyond path traversal, the allowlist must also reject values that
    break portability or common CLI/shell assumptions.

    * Empty string and ``.``/``..`` are ambiguous path names.
    * Null byte would break most POSIX calls.
    * Leading ``-`` collides with argparse-style flag parsing when the
      id is later pasted into another command.
    * Leading ``.`` would hide the run directory from default ``ls``.
    * Absurdly long values should be rejected eagerly.
    """
    bad_cases = [
        "",                  # empty
        ".", "..",          # reserved
        "hello\x00world",    # null byte
        "-dash-leading",     # CLI-flag ambiguity
        ".hidden-leading",   # hides the directory
        "a" * 500,          # length bound
        "bad chars!",        # whitespace + punctuation
        "em🍕oji",          # non-ASCII disallowed
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for bad in bad_cases:
            try:
                RunLogger(root_dir=tmp, run_id=bad)
            except InvalidRunIdError:
                continue
            raise AssertionError(f"run_id={bad!r} should have been rejected")
    print("  PASS run_id_rejects_reserved_and_unsafe_values")


def test_run_id_accepts_portable_alphanumeric_forms() -> None:
    """The sanitizer must still accept normal operator-chosen ids and
    the auto-generated timestamp+uuid form."""
    good_cases = [
        "run-1",
        "orchestration-test-run",
        "exp.2026-04-21",
        "abc_123",
        "Z9",
        new_run_id(),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for good in good_cases:
            logger = RunLogger(root_dir=tmp, run_id=good)
            _assert(logger.run_id == good,
                    f"run_id should round-trip, got {logger.run_id!r}")
            _assert(logger.run_dir.is_dir(),
                    f"run_dir not created for {good!r}: {logger.run_dir}")
    print("  PASS run_id_accepts_portable_alphanumeric_forms")


def test_duplicate_run_id_refuses_to_merge_into_existing_run_dir() -> None:
    """Two runs with the same explicit ``run_id`` must not merge.

    Without this check a second ``RunLogger(...run_id='dup')`` appends
    to the existing ``events.jsonl``, producing duplicate ``seq``
    values (reproducer: ``[1, 2, 1, 2]``) and making the ordering
    contract meaningless. After the fix the second construction
    raises :class:`FileExistsError` *before* touching either file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lg1 = RunLogger(root_dir=root, run_id="dup")
        lg1.emit("phase_a", {})
        lg1.emit("phase_b", {})

        # Capture the first run's artifacts so we can prove the second
        # construction didn't mutate them.
        events_before = (root / "dup" / EVENTS_FILENAME).read_text(encoding="utf-8")
        state_before = (root / "dup" / STATE_FILENAME).read_text(encoding="utf-8")

        try:
            RunLogger(root_dir=root, run_id="dup")
        except FileExistsError:
            pass
        else:
            raise AssertionError(
                "second RunLogger with a duplicate run_id must raise "
                "FileExistsError instead of silently merging"
            )

        events_after = (root / "dup" / EVENTS_FILENAME).read_text(encoding="utf-8")
        state_after = (root / "dup" / STATE_FILENAME).read_text(encoding="utf-8")
        _assert(events_before == events_after,
                "first run's events.jsonl must not be touched by the "
                "rejected second construction")
        _assert(state_before == state_after,
                "first run's state.json must not be touched by the "
                "rejected second construction")

        seqs = [json.loads(line)["seq"] for line in events_after.splitlines() if line.strip()]
        _assert(seqs == [1, 2],
                f"events.jsonl must not carry duplicate seq values; got {seqs}")
    print("  PASS duplicate_run_id_refuses_to_merge_into_existing_run_dir")


def test_run_id_rejects_symlink_pointing_outside_root() -> None:
    """A pre-existing symlink at ``root/<run_id>`` must not be followed.

    Reproduces the reviewer's exact scenario: ``ln -s /outside logs/linkrun``
    then ``RunLogger(root_dir='logs', run_id='linkrun')``. Before this
    was fixed, the constructor happily wrote ``events.jsonl`` /
    ``state.json`` into the symlink target because it only checked
    ``run_dir.parent.resolve()`` — which still points at the real logs
    root since the symlink is *at* the run-id component, not above it.

    The guard must now reject this outright with no filesystem writes
    at either the root-level symlink path or the target path.
    """
    with tempfile.TemporaryDirectory() as root_tmp:
        with tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(root_tmp)
            outside = Path(outside_tmp)
            link_path = root / "linkrun"
            os.symlink(outside, link_path)

            outside_pre = set(outside.iterdir())

            try:
                RunLogger(root_dir=root, run_id="linkrun")
            except InvalidRunIdError:
                pass
            else:
                raise AssertionError(
                    "symlink at run_dir pointing outside the logs root "
                    "must be rejected; constructor returned normally"
                )

            # No writes leaked into the symlink target.
            outside_post = set(outside.iterdir())
            _assert(outside_post == outside_pre,
                    f"rejected symlink run_id must not leak files into "
                    f"target; new entries: {outside_post - outside_pre}")
            # And since the write would have gone *through* the symlink,
            # the symlink itself should also be empty of log artifacts.
            # (``Path.iterdir`` follows symlinks, so this is the same set.)
            _assert(not (outside / EVENTS_FILENAME).exists(),
                    "events.jsonl must not exist at symlink target")
            _assert(not (outside / STATE_FILENAME).exists(),
                    "state.json must not exist at symlink target")
    print("  PASS run_id_rejects_symlink_pointing_outside_root")


def test_run_id_rejects_dangling_symlink_at_run_dir() -> None:
    """A *broken* symlink at ``run_dir`` must still be rejected.

    Belt-and-suspenders: ``Path.is_symlink`` returns True for dangling
    links too, so even a target that doesn't exist cannot be used to
    side-step the per-run-directory contract (nor cause a confusing
    ``FileNotFoundError`` deeper in the writer).
    """
    with tempfile.TemporaryDirectory() as root_tmp:
        root = Path(root_tmp)
        link_path = root / "danglerun"
        os.symlink(root / "this-does-not-exist", link_path)

        try:
            RunLogger(root_dir=root, run_id="danglerun")
        except InvalidRunIdError:
            pass
        else:
            raise AssertionError(
                "dangling symlink at run_dir must be rejected"
            )

        _assert(link_path.is_symlink(),
                "the dangling link itself should be untouched by the "
                "rejected construction")
    print("  PASS run_id_rejects_dangling_symlink_at_run_dir")


def test_run_id_allows_symlinked_root_when_run_dir_lands_under_it() -> None:
    """A symlinked ``root_dir`` is fine as long as ``run_dir`` is a
    direct child of the canonical root. Operators sometimes expose
    ``logs/`` via a symlink into a data volume; that must keep working.
    The guard only fires when ``run_dir.resolve()`` differs from
    ``root.resolve() / run_id`` — which it doesn't in this case."""
    with tempfile.TemporaryDirectory() as real_tmp:
        with tempfile.TemporaryDirectory() as link_parent:
            real_root = Path(real_tmp)
            link_root = Path(link_parent) / "logs-via-link"
            os.symlink(real_root, link_root)

            logger = RunLogger(root_dir=link_root, run_id="via-link")
            logger.emit("probe", {"ok": True})

            # Writes land in the real root, but the "logs/<run_id>/"
            # contract is still met because the resolved run_dir is a
            # direct child of the resolved root.
            _assert((real_root / "via-link" / EVENTS_FILENAME).exists(),
                    "events.jsonl should exist in the resolved root")
            _assert((real_root / "via-link" / STATE_FILENAME).exists(),
                    "state.json should exist in the resolved root")
    print("  PASS run_id_allows_symlinked_root_when_run_dir_lands_under_it")


def test_empty_preexisting_dir_is_tolerated_for_new_run() -> None:
    """An empty sibling directory (operator pre-created, stale mount, …)
    must not block a fresh run that claims it by ``run_id``. Only a
    directory already holding ``events.jsonl`` / ``state.json`` counts
    as a collision.
    """
    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "empty-preexisting"
        pre.mkdir(parents=True)
        logger = RunLogger(root_dir=tmp, run_id="empty-preexisting")
        _assert(logger.events_path.exists() and logger.state_path.exists(),
                "empty pre-existing dir must be usable for a new run")
        _assert(logger.run_dir == pre, "run_dir path mismatch")
    print("  PASS empty_preexisting_dir_is_tolerated_for_new_run")


def test_finish_falls_back_to_live_state_when_counters_omitted() -> None:
    """``finish()`` must default ``total_reviews`` and ``outer_loops`` to
    the live in-memory state when the caller omits them.

    This is the core invariant the mid-loop crash path relies on. If
    the loop emits real progress events (``outer_started``,
    ``review_started``, …) and then raises before the caller learns
    the final counters, the caller's locals are still stale zeros.
    Passing those stale zeros would overwrite the already-correct live
    values in both ``state.json`` and ``index.jsonl``. The fallback
    makes the structured logs trustworthy even when the caller can't
    be.
    """
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)
        logger.start(config={"prompt": "p", "max_outer": 3, "max_inner": 3})

        # Drive real progress events — these are the same helpers the
        # orchestration loop uses, so state updates exactly match what
        # a live run would produce.
        logger.outer_started(outer=1, max_outer=3)
        logger.inner_started(outer=1, inner=1, max_inner=3, tag="[O1 I1]")
        logger.review_started(model="m", tag="[O1 I1]")
        logger.review_finished(verdict="CHANGES_NEEDED", rc=1, tag="[O1 I1]")
        logger.fix_started(model="m", tag="[O1 I1]")
        logger.fix_finished(rc=0, tag="[O1 I1]")
        logger.inner_finished(outer=1, inner=1, verdict="CHANGES_NEEDED")
        logger.inner_started(outer=1, inner=2, max_inner=3, tag="[O1 I2]")
        logger.review_started(model="m", tag="[O1 I2]")

        # Live state after the above event stream.
        pre_finish = _read_state(logger.state_path)
        _assert(pre_finish["total_reviews"] == 2,
                f"precondition: state.total_reviews must be 2 from "
                f"review_started events, got {pre_finish['total_reviews']}")
        _assert(pre_finish["outer"] == 1,
                f"precondition: state.outer must be 1, got {pre_finish['outer']}")

        # Call finish() WITHOUT counters — simulates the crash path.
        logger.finish(approved=False, exit_code=1)

        state = _read_state(logger.state_path)
        index = _read_index(Path(tmp) / INDEX_FILENAME)

        # State's total_reviews must be preserved (or, equivalently,
        # re-derived from the same live value; either way, never zero).
        _assert(state["total_reviews"] == 2,
                f"finish() with omitted total_reviews must fall back "
                f"to live state, got {state['total_reviews']}")

        idx_finish = index[-1]
        _assert(idx_finish["type"] == "run_finished",
                f"last index entry must be run_finished, got {idx_finish}")
        _assert(idx_finish["total_reviews"] == 2,
                f"index.run_finished.total_reviews must match live "
                f"state, got {idx_finish['total_reviews']}")
        _assert(idx_finish["outer_loops"] == 1,
                f"index.run_finished.outer_loops must match live "
                f"state.outer, got {idx_finish['outer_loops']}")
    print("  PASS finish_falls_back_to_live_state_when_counters_omitted")


def test_finish_preserves_explicit_counters_over_live_state() -> None:
    """Callers that *do* have authoritative counts must still win.

    The fallback to live state only kicks in when the caller passes
    ``None`` (or omits the argument). An explicit non-None value must
    always be honoured — otherwise the happy-path call site (where
    the locals from ``_run_loop()``'s return value are the source of
    truth) would silently race with whatever order the logger
    observed the updates in.
    """
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)
        logger.start(config={"prompt": "p", "max_outer": 2, "max_inner": 2})

        # Push live state up to (total_reviews=2, outer=1).
        logger.outer_started(outer=1, max_outer=2)
        logger.inner_started(outer=1, inner=1, max_inner=2, tag="[O1 I1]")
        logger.review_started(model="m", tag="[O1 I1]")
        logger.review_started(model="m", tag="[O1 I1]")

        # Caller passes explicit values that differ from live state.
        logger.finish(
            approved=True,
            exit_code=0,
            total_reviews=42,
            outer_loops=7,
        )

        state = _read_state(logger.state_path)
        index = _read_index(Path(tmp) / INDEX_FILENAME)

        _assert(state["total_reviews"] == 42,
                f"explicit total_reviews must win over live state "
                f"(expected 42, got {state['total_reviews']})")
        idx_finish = index[-1]
        _assert(idx_finish["total_reviews"] == 42,
                f"index.total_reviews must reflect explicit value, "
                f"got {idx_finish['total_reviews']}")
        _assert(idx_finish["outer_loops"] == 7,
                f"index.outer_loops must reflect explicit value, "
                f"got {idx_finish['outer_loops']}")
    print("  PASS finish_preserves_explicit_counters_over_live_state")


def test_finish_before_any_progress_reports_zeros() -> None:
    """If ``finish()`` is called before any progress event landed —
    e.g. a crash very early in ``main()``, before ``_run_loop`` even
    started — the fallback must report ``0`` counters, not ``None``
    or a stale sentinel. A consumer reading ``index.jsonl`` must not
    need to know that ``state.outer`` was ``None`` mid-run; the
    finish record is a flat, well-typed summary.
    """
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)
        logger.start(config={"prompt": "p", "max_outer": 1, "max_inner": 1})

        # Precondition: initial state has no progress yet.
        initial = _read_state(logger.state_path)
        _assert(initial["outer"] is None,
                f"precondition: state.outer starts None, got {initial['outer']}")
        _assert(initial["total_reviews"] == 0,
                f"precondition: state.total_reviews starts 0, got "
                f"{initial['total_reviews']}")

        # Omit both counters — crash-path semantics.
        logger.finish(approved=False, exit_code=1)

        state = _read_state(logger.state_path)
        index = _read_index(Path(tmp) / INDEX_FILENAME)
        idx_finish = index[-1]

        # Both counters must be the int ``0`` — never ``None``, since
        # ``index.jsonl`` is a stable public contract that downstream
        # tools parse as ints.
        _assert(state["total_reviews"] == 0,
                f"state.total_reviews must be 0 when no reviews ran, "
                f"got {state['total_reviews']!r}")
        _assert(idx_finish["outer_loops"] == 0 and isinstance(idx_finish["outer_loops"], int),
                f"index.outer_loops must be int 0 on no-progress crash, "
                f"got {idx_finish['outer_loops']!r}")
        _assert(idx_finish["total_reviews"] == 0
                and isinstance(idx_finish["total_reviews"], int),
                f"index.total_reviews must be int 0 on no-progress "
                f"crash, got {idx_finish['total_reviews']!r}")
    print("  PASS finish_before_any_progress_reports_zeros")


def test_started_at_matches_in_index_and_state_after_start() -> None:
    """``index.jsonl.run_started.started_at`` must equal
    ``state.json.started_at`` exactly.

    Reviewer reproducer: on a clean run, the index entry was written
    using the constructor-time ``_state["started_at"]`` while the
    subsequent ``run_started`` event handler then overwrote
    ``state.started_at`` from the config payload. Result: a small but
    real drift (reviewer saw ~2ms) between the two machine-readable
    surfaces for the same run — an avoidable inconsistency in the
    public contract.

    With the fix, ``start()`` promotes ``config.started_at`` to the
    canonical value *before* writing the index entry, so both files
    agree on one timestamp. When ``config.started_at`` is omitted
    (legacy / test callers), the constructor-time default is used
    uniformly.
    """
    fixed_ts = "2026-01-02T03:04:05.678901+00:00"
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)
        # Capture the constructor-time default so we can prove the
        # config value actually took precedence.
        ctor_started_at = _read_state(logger.state_path)["started_at"]
        _assert(ctor_started_at != fixed_ts,
                "precondition: constructor time must differ from the "
                "fixed config value; otherwise the test proves nothing")

        logger.start(config={
            "prompt": "p",
            "max_outer": 1,
            "max_inner": 1,
            "started_at": fixed_ts,
        })

        index = _read_index(Path(tmp) / INDEX_FILENAME)
        state = _read_state(logger.state_path)

        run_started = next(e for e in index if e["type"] == "run_started")
        _assert(run_started["started_at"] == fixed_ts,
                f"index.run_started.started_at must honour "
                f"config.started_at (expected {fixed_ts!r}, got "
                f"{run_started['started_at']!r})")
        _assert(state["started_at"] == fixed_ts,
                f"state.started_at must honour config.started_at "
                f"(expected {fixed_ts!r}, got {state['started_at']!r})")
        _assert(run_started["started_at"] == state["started_at"],
                f"index and state must agree on started_at "
                f"(index={run_started['started_at']!r}, "
                f"state={state['started_at']!r})")
    print("  PASS started_at_matches_in_index_and_state_after_start")


def test_started_at_falls_back_to_ctor_time_when_config_omits_it() -> None:
    """When ``config.started_at`` is absent, the constructor-time value
    is used as the canonical ``started_at`` in *both* files.

    Callers that don't supply ``started_at`` in their config (older
    tests, non-CLI entrypoints) still get a consistent timestamp
    across index and state — never one source of truth from the
    constructor and a different one from the event handler.
    """
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(root_dir=tmp)
        ctor_started_at = _read_state(logger.state_path)["started_at"]

        logger.start(config={
            "prompt": "p",
            "max_outer": 1,
            "max_inner": 1,
            # no started_at here
        })

        index = _read_index(Path(tmp) / INDEX_FILENAME)
        state = _read_state(logger.state_path)
        run_started = next(e for e in index if e["type"] == "run_started")

        _assert(run_started["started_at"] == ctor_started_at,
                f"index.run_started.started_at should fall back to "
                f"constructor time when config omits it "
                f"(expected {ctor_started_at!r}, got "
                f"{run_started['started_at']!r})")
        _assert(state["started_at"] == ctor_started_at,
                f"state.started_at should fall back to constructor "
                f"time when config omits it "
                f"(expected {ctor_started_at!r}, got "
                f"{state['started_at']!r})")
    print("  PASS started_at_falls_back_to_ctor_time_when_config_omits_it")


def test_coexists_with_legacy_flat_log_files() -> None:
    """index.jsonl and per-run dirs must coexist with older flat ``.log``
    artifacts that live in the same root without polluting the schema."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Pre-existing flat log from an older run (what logs/ already has today).
        (root / "legacy-run.log").write_text("old style log\n", encoding="utf-8")

        logger = RunLogger(root_dir=tmp)
        logger.start(config={"prompt": "p", "max_outer": 1, "max_inner": 1})
        logger.finish(approved=False, exit_code=1, total_reviews=0, outer_loops=0)

        _assert((root / "legacy-run.log").exists(),
                "legacy .log file should remain untouched")
        _assert((root / INDEX_FILENAME).exists(), "index.jsonl not created")
        _assert((root / logger.run_id / EVENTS_FILENAME).exists(),
                "per-run events.jsonl missing")
        _assert((root / logger.run_id / STATE_FILENAME).exists(),
                "per-run state.json missing")
    print("  PASS coexists_with_legacy_flat_log_files")


# ── Broken-stdout defence (Round 11) ────────────────────────────────────────
# These tests pin the contract of the ``_safe_print`` / ``mark_stdout_broken``
# helpers that keep ``review-loop.py`` alive when its stdout pipe closes
# (SSH drop, pager quit, ``review-loop.py | head -0``, …). Without them the
# loop aborted mid-work on the first ``log()`` call after the pipe closed,
# and — worse — the interpreter-shutdown flush of ``sys.stdout`` hit the
# broken pipe and exited **120**, making the OS exit code disagree with
# ``state.json.exit_code``. Full end-to-end coverage (shell ``$?`` vs
# ``state.exit_code`` after ``| head -0``) lives in
# ``tests/test_run_log_orchestration.py``; these tests nail down the
# helper-level invariants so a future refactor can't silently regress.


def test_safe_print_swallows_oserror_regardless_of_stream() -> None:
    """``_safe_print`` must never raise ``OSError``.

    Reviewer's repro (``review-loop.py | head -0``): a mid-run
    ``log()`` call after the pipe closed propagated ``BrokenPipeError``
    up through the orchestration loop and aborted the run before
    any review/fix step could execute. The fix routes every print
    through ``_safe_print`` which catches ``OSError`` and silently
    drops. This test pins that contract at the helper level.
    """
    from iterator_loop import logging as lg

    class BrokenStream:
        def write(self, *a, **kw):
            raise BrokenPipeError(32, "simulated broken pipe")

        def flush(self):
            raise BrokenPipeError(32, "simulated broken pipe")

    # Explicit broken file — must NOT propagate.
    lg._safe_print("hello", file=BrokenStream(), flush=True)
    lg._safe_print("world", file=BrokenStream())
    # Also must survive non-pipe OSError (e.g. ENOSPC on a broken tty).
    class EnospcStream:
        def write(self, *a, **kw):
            raise OSError(28, "No space left on device")

        def flush(self):
            raise OSError(28, "No space left on device")

    lg._safe_print("disk full", file=EnospcStream())
    print("  PASS safe_print_swallows_oserror_regardless_of_stream")


def test_safe_print_marks_stdout_broken_only_on_stdout_failure() -> None:
    """``_safe_print`` only triggers the global ``fd 1`` swap when the
    failing stream is ``sys.stdout`` (or ``file=None`` — the default).

    Why it matters: callers that write to stderr or custom streams
    may hit transient ``OSError`` conditions that have nothing to
    do with stdout. Silencing stdout for the rest of the run on
    those would lose human-facing output the terminal can still
    show. The defence is scoped to its actual target: stdout
    itself dying.
    """
    from iterator_loop import logging as lg

    class BrokenStream:
        def write(self, *a, **kw):
            raise BrokenPipeError(32, "broken")

        def flush(self):
            raise BrokenPipeError(32, "broken")

    # Stub ``mark_stdout_broken`` so the test process's real fd 1
    # is left alone; we only care whether the helper decided to
    # call it.
    calls: list[int] = []
    orig_mark = lg.mark_stdout_broken

    def stub_mark() -> bool:
        calls.append(1)
        return True

    lg.mark_stdout_broken = stub_mark  # type: ignore[assignment]

    orig_stdout = sys.stdout
    try:
        # 1. Non-stdout stream failure → must NOT mark broken.
        lg._safe_print("x", file=BrokenStream())
        _assert(calls == [],
                f"non-stdout failure must not mark stdout broken, got "
                f"{len(calls)} call(s)")

        # 2. Implicit stdout (``file=None``) failure → MUST mark broken
        #    exactly once.
        sys.stdout = BrokenStream()  # type: ignore[assignment]
        lg._safe_print("y")
        _assert(calls == [1],
                f"implicit stdout failure must mark broken exactly "
                f"once, got {calls}")

        # 3. Explicit ``file=sys.stdout`` failure → also marks broken.
        lg._safe_print("z", file=sys.stdout)
        _assert(calls == [1, 1],
                f"explicit file=sys.stdout must also mark broken, got "
                f"{calls}")
    finally:
        sys.stdout = orig_stdout
        lg.mark_stdout_broken = orig_mark  # type: ignore[assignment]
    print("  PASS safe_print_marks_stdout_broken_only_on_stdout_failure")


def test_mark_stdout_broken_is_idempotent_in_subprocess() -> None:
    """End-to-end subprocess test for ``mark_stdout_broken``.

    The helper is global and irreversible (it swaps fd 1 to
    ``/dev/null`` with :func:`os.dup2`), so we exercise it in a
    subprocess to keep the main test process's fd 1 intact.

    Contract:

    * First call returns ``True`` and actually redirects fd 1 to
      ``/dev/null``.
    * :func:`is_stdout_broken` flips to ``True`` after the first
      call.
    * Second call returns ``False`` (idempotent) and is a no-op.
    * ``sys.stdout`` writes after the swap succeed silently — the
      subprocess's stdout capture stays empty.
    * ``stderr`` is untouched.
    * A subsequent ``sys.exit(42)`` produces exit code ``42`` in
      the parent, not ``120``. This is the direct defence against
      the reviewer's "shell sees 120, state says 1" regression:
      the interpreter-shutdown flush of ``sys.stdout`` now lands
      in ``/dev/null`` and the chosen exit code survives.
    """
    import subprocess
    script = r"""
import os, stat, sys
sys.path.insert(0, "src")
from iterator_loop.logging import mark_stdout_broken, is_stdout_broken

assert is_stdout_broken() is False, "flag should start False"
assert mark_stdout_broken() is True, "first call should return True"
assert is_stdout_broken() is True, "flag should be True after first call"
assert mark_stdout_broken() is False, "second call should return False"

# fd 1 must now point at /dev/null (a character device on Linux).
st = os.fstat(1)
assert stat.S_ISCHR(st.st_mode), f"fd 1 is not a char device: mode={st.st_mode:o}"

# Post-swap stdout writes succeed silently and must not reach the
# parent's captured pipe. sys.stderr is untouched so we can still
# communicate.
print("this should vanish into /dev/null")
print("along with this one")
sys.stderr.write("OK-FROM-STDERR\n")

# Critical: interpreter shutdown flush after sys.exit(42) must see
# /dev/null, not a broken pipe, so the parent observes 42 — not 120.
sys.exit(42)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
    )
    _assert(result.returncode == 42,
            f"subprocess should exit 42, got {result.returncode}; "
            f"stderr={result.stderr!r}")
    _assert(result.stdout == "",
            f"post-swap stdout writes must not reach the parent, "
            f"got {result.stdout!r}")
    _assert("OK-FROM-STDERR" in result.stderr,
            f"stderr should still work after the stdout swap, "
            f"got {result.stderr!r}")
    print("  PASS mark_stdout_broken_is_idempotent_in_subprocess")


TESTS = [
    test_new_run_id_is_sortable_and_unique,
    test_logger_creates_initial_files,
    test_emit_appends_ordered_events_and_rewrites_state,
    test_start_and_finish_update_index_and_state,
    test_tail_ring_buffer_caps_at_tail_size,
    test_tool_call_events_update_pending_tools_and_last_tool,
    test_atomic_state_write_never_leaves_tmp,
    test_unknown_event_types_are_accepted,
    test_guidance_received_sets_pending_guidance,
    test_tail_handles_streamed_partial_deltas,
    test_tail_handles_non_streaming_assistant_message,
    test_tail_does_not_duplicate_agent_result_echo,
    test_tail_session_boundary_flushes_dangling_partial,
    test_usage_accumulates_across_agent_results,
    test_usage_tolerates_missing_or_partial_fields,
    test_run_error_event_populates_state_error_slot,
    test_run_id_rejects_path_traversal_and_escapes_root,
    test_run_id_rejects_reserved_and_unsafe_values,
    test_run_id_accepts_portable_alphanumeric_forms,
    test_duplicate_run_id_refuses_to_merge_into_existing_run_dir,
    test_run_id_rejects_symlink_pointing_outside_root,
    test_run_id_rejects_dangling_symlink_at_run_dir,
    test_run_id_allows_symlinked_root_when_run_dir_lands_under_it,
    test_empty_preexisting_dir_is_tolerated_for_new_run,
    test_finish_falls_back_to_live_state_when_counters_omitted,
    test_finish_preserves_explicit_counters_over_live_state,
    test_finish_before_any_progress_reports_zeros,
    test_started_at_matches_in_index_and_state_after_start,
    test_started_at_falls_back_to_ctor_time_when_config_omits_it,
    test_coexists_with_legacy_flat_log_files,
    test_safe_print_swallows_oserror_regardless_of_stream,
    test_safe_print_marks_stdout_broken_only_on_stdout_failure,
    test_mark_stdout_broken_is_idempotent_in_subprocess,
]


def main() -> None:
    print("=" * 60)
    print("RunLogger unit tests")
    print("-" * 60)
    for fn in TESTS:
        fn()
    print("=" * 60)
    print(f"ALL {len(TESTS)} TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
