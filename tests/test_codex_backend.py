"""Codex backend — command building and event-stream compatibility.

These tests pin the backend's compatibility surface for both supported
codex-cli versions:

* Initial / continue command shapes use a ``model_reasoning_effort``
  value that's accepted by codex-cli 0.36 (whose ``ReasoningEffort`` enum
  tops out at ``high``) as well as by 0.124+ (which adds ``xhigh`` but
  still accepts ``high``).
* ``build_continue_cmd`` does not invoke ``codex exec resume`` — that
  subcommand has version-dependent flag-parsing quirks (PR
  openai/codex#8440 made global flags work after ``resume`` only in
  0.124+, and issue openai/codex#6717 makes ``--last`` reject any
  positional prompt). The retry path therefore re-runs ``codex exec``
  from scratch with the original prompt prepended by a resume hint.
* ``handle_event`` decodes both the modern ``thread.started`` /
  ``item.completed`` shape and the legacy ``{id, msg: {type, …}}`` shape
  emitted by codex-cli 0.36.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_iterator.backends.codex import (  # noqa: E402
    _RESUME_PROMPT,
    CodexBackend,
)


# ── Fakes ────────────────────────────────────────────────────────────────────


class _FakeFmt:
    """Captures the ``feed`` / ``feed_tool`` calls dispatched by the backend.

    The real ``OutputFormatter`` only cares about printing — for these
    tests we just need to inspect what got routed where so we can assert
    on agent text vs tool breadcrumbs without having to drive a PTY.
    """

    def __init__(self) -> None:
        self.text_lines: list[str] = []
        self.tool_lines: list[str] = []

    def feed(self, line: str) -> None:
        self.text_lines.append(line)

    def feed_tool(self, line: str) -> None:
        self.tool_lines.append(line)


class _FakeReader:
    """Minimal stand-in for ``_StreamReader`` for ``handle_event`` tests."""

    def __init__(self) -> None:
        self._fmt = _FakeFmt()
        self._text_buf: list[str] = []
        self._full_text: list[str] = []
        self.pending_tools = 0
        self.saw_result = False

    def _flush_text(self) -> None:
        if not self._text_buf:
            return
        for line in "".join(self._text_buf).split("\n"):
            self._fmt.feed(line)
        self._text_buf.clear()


# ── Command-building tests ───────────────────────────────────────────────────


def test_build_initial_cmd_uses_high_reasoning_effort() -> None:
    """``high`` is the topmost variant accepted by codex-cli 0.36's enum.

    Earlier revisions of this backend pinned ``xhigh`` (a variant only
    added in 0.124+), which made the initial run abort on 0.36 with
    ``unknown variant 'xhigh' …expected one of minimal/low/medium/high``.
    """
    cmd = CodexBackend().build_initial_cmd(
        agent_cmd="codex",
        model="gpt-5.5",
        prompt="task",
        workspace="/tmp",
        extra_flags=[],
    )
    assert 'model_reasoning_effort="high"' in cmd
    assert 'model_reasoning_effort="xhigh"' not in cmd


def test_build_initial_cmd_layout() -> None:
    """The base flag block precedes ``--model`` / ``-C`` / prompt."""
    cmd = CodexBackend().build_initial_cmd(
        agent_cmd="codex",
        model="gpt-5.5",
        prompt="do the thing",
        workspace="/some/ws",
        extra_flags=["--extra"],
    )
    assert cmd[0] == "codex"
    assert cmd[1] == "exec"
    # Global stream-config flags must come before the subcommand-modifying
    # flags so codex-cli 0.36 (which only accepts them as global) parses
    # them correctly.
    assert "--json" in cmd
    json_idx = cmd.index("--json")
    model_idx = cmd.index("--model")
    assert json_idx < model_idx
    # Workspace + prompt are the trailing arguments.
    assert cmd[-3:] == ["-C", "/some/ws", "--extra", "do the thing"][-3:] or (
        cmd.index("-C") < cmd.index("--extra") < len(cmd) - 1
    )
    assert cmd[-1] == "do the thing"


def test_build_continue_cmd_does_not_use_resume_subcommand() -> None:
    """Retries must not invoke ``codex exec resume``.

    The resume subcommand has two version-dependent quirks (flag-parsing
    rules that differ between 0.36 and 0.124+, plus a clap conflict
    between ``--last`` and any positional prompt). We sidestep both by
    re-running ``codex exec`` with the original prompt prepended by the
    resume hint.
    """
    cmd = CodexBackend().build_continue_cmd(
        agent_cmd="codex",
        model="gpt-5.5",
        prompt="original task",
        workspace="/tmp",
        extra_flags=[],
    )
    assert "resume" not in cmd
    assert "--last" not in cmd
    assert cmd[0] == "codex"
    assert cmd[1] == "exec"
    # The prompt is always last and prefixed with the resume hint so the
    # agent knows it's a recovery attempt rather than a first try.
    assert cmd[-1].startswith(_RESUME_PROMPT)
    assert "original task" in cmd[-1]


def test_build_continue_cmd_uses_high_reasoning_effort() -> None:
    """Same enum constraint as the initial command — never emit xhigh."""
    cmd = CodexBackend().build_continue_cmd(
        agent_cmd="codex",
        model="gpt-5.5",
        prompt="t",
        workspace="/tmp",
        extra_flags=[],
    )
    assert 'model_reasoning_effort="high"' in cmd
    assert 'model_reasoning_effort="xhigh"' not in cmd


# ── Modern (>=0.124) event format ────────────────────────────────────────────


def test_handle_event_modern_agent_message() -> None:
    """``item.completed`` with ``agent_message`` populates ``_full_text``."""
    reader = _FakeReader()
    CodexBackend().handle_event(
        {"type": "item.completed",
         "item": {"id": "x", "type": "agent_message", "text": "hello"}},
        reader,
    )
    assert reader._full_text == ["hello"]
    assert "hello" in reader._fmt.text_lines


def test_handle_event_modern_tool_pairs_pending() -> None:
    """``item.started`` + ``item.completed`` increment / decrement pending."""
    be = CodexBackend()
    reader = _FakeReader()
    be.handle_event(
        {"type": "item.started",
         "item": {"id": "1", "type": "command_execution",
                  "command": "ls", "status": "in_progress"}},
        reader,
    )
    assert reader.pending_tools == 1
    be.handle_event(
        {"type": "item.completed",
         "item": {"id": "1", "type": "command_execution",
                  "command": "ls", "exit_code": 0,
                  "aggregated_output": "x", "status": "completed"}},
        reader,
    )
    assert reader.pending_tools == 0


def test_handle_event_modern_turn_completed_marks_result() -> None:
    reader = _FakeReader()
    CodexBackend().handle_event(
        {"type": "turn.completed",
         "usage": {"input_tokens": 1, "cached_input_tokens": 0,
                   "output_tokens": 1}},
        reader,
    )
    assert reader.saw_result is True


# ── Legacy (codex 0.36) event format ─────────────────────────────────────────


def test_handle_event_legacy_agent_message() -> None:
    """Legacy ``msg.type=agent_message`` with ``message`` field is captured."""
    reader = _FakeReader()
    CodexBackend().handle_event(
        {"id": "0", "msg": {"type": "agent_message", "message": "hi there"}},
        reader,
    )
    assert reader._full_text == ["hi there"]


def test_handle_event_legacy_tool_pairs_pending() -> None:
    """``exec_command_begin`` / ``exec_command_end`` track pending tools."""
    be = CodexBackend()
    reader = _FakeReader()
    be.handle_event(
        {"id": "0",
         "msg": {"type": "exec_command_begin",
                 "call_id": "c1",
                 "command": ["bash", "-lc", "ls"],
                 "cwd": "/tmp"}},
        reader,
    )
    assert reader.pending_tools == 1
    be.handle_event(
        {"id": "0",
         "msg": {"type": "exec_command_end",
                 "call_id": "c1",
                 "stdout": "file.txt\n",
                 "stderr": "",
                 "exit_code": 0}},
        reader,
    )
    assert reader.pending_tools == 0


def test_handle_event_legacy_patch_apply_tracks_pending() -> None:
    be = CodexBackend()
    reader = _FakeReader()
    be.handle_event(
        {"id": "0",
         "msg": {"type": "patch_apply_begin",
                 "call_id": "p1",
                 "auto_approved": True,
                 "changes": {"a.txt": {"kind": "update"}}}},
        reader,
    )
    assert reader.pending_tools == 1
    be.handle_event(
        {"id": "0",
         "msg": {"type": "patch_apply_end",
                 "call_id": "p1",
                 "success": True}},
        reader,
    )
    assert reader.pending_tools == 0


def test_handle_event_legacy_task_complete_marks_result() -> None:
    """``task_complete`` is the legacy ``saw_result`` signal."""
    reader = _FakeReader()
    CodexBackend().handle_event(
        {"id": "0",
         "msg": {"type": "task_complete",
                 "last_agent_message": "all done"}},
        reader,
    )
    assert reader.saw_result is True


def test_handle_event_legacy_task_complete_fills_text_when_missing() -> None:
    """If ``agent_message`` was missed, ``last_agent_message`` is the fallback."""
    reader = _FakeReader()
    CodexBackend().handle_event(
        {"id": "0",
         "msg": {"type": "task_complete",
                 "last_agent_message": "fallback text"}},
        reader,
    )
    assert reader._full_text == ["fallback text"]


def test_handle_event_legacy_task_complete_preserves_prior_text() -> None:
    """If ``_full_text`` already has content, don't overwrite with fallback."""
    reader = _FakeReader()
    reader._full_text.append("already captured")
    CodexBackend().handle_event(
        {"id": "0",
         "msg": {"type": "task_complete",
                 "last_agent_message": "fallback"}},
        reader,
    )
    assert reader._full_text == ["already captured"]


def test_handle_event_unknown_event_is_ignored() -> None:
    """Unrecognised events must not crash the dispatcher.

    Both codex-cli versions emit a long tail of events we don't model
    (token counts, reasoning summaries, MCP list responses, …); the loop
    is happy as long as nothing raises.
    """
    reader = _FakeReader()
    CodexBackend().handle_event(
        {"type": "thread.started", "thread_id": "abc"}, reader,
    )
    CodexBackend().handle_event(
        {"id": "0", "msg": {"type": "session_configured",
                            "session_id": "abc", "model": "gpt-5.5"}},
        reader,
    )
    CodexBackend().handle_event({}, reader)
    CodexBackend().handle_event({"id": "0"}, reader)


def test_handle_event_dispatch_picks_modern_when_both_keys_present() -> None:
    """If a top-level ``type`` is present, dispatch to the modern path.

    Belt-and-braces: in case some future codex revision emits both
    shapes during a transition window, the modern shape wins.
    """
    reader = _FakeReader()
    CodexBackend().handle_event(
        {"type": "item.completed",
         "item": {"id": "x", "type": "agent_message", "text": "modern"},
         "msg": {"type": "agent_message", "message": "legacy"}},
        reader,
    )
    assert reader._full_text == ["modern"]


# ── Continue-cmd plumbing through agent.run_agent ────────────────────────────


def test_run_agent_passes_prompt_to_build_continue_cmd(monkeypatch: pytest.MonkeyPatch) -> None:
    """``agent.run_agent`` forwards the original prompt on retry attempts.

    Pre-fix, ``build_continue_cmd`` had no access to the prompt, which
    forced the codex backend to fabricate a context-free ``--last`` resume
    command. With the prompt threaded through, codex can re-run from
    scratch with the real task on retry.
    """
    import asyncio
    from auto_iterator import agent as agent_mod

    captured: dict[str, object] = {}

    class _StubBackend:
        def build_initial_cmd(self, *args, **kwargs):
            return ["/bin/false"]

        def build_continue_cmd(
            self, agent_cmd, model, prompt, workspace, extra_flags,
        ):
            captured["prompt"] = prompt
            return ["/bin/true"]

        def handle_event(self, evt, reader):
            pass

    async def _fake_run_once(cmd, tag, backend, *, cwd=None):
        # Force one abnormal exit so the loop dips into ``build_continue_cmd``.
        if cmd == ["/bin/false"]:
            return agent_mod._RunOutcome(
                rc=1, signal_name="", text="",
                saw_result=False, pending_tools=0, stream_closed_msg=False,
            )
        return agent_mod._RunOutcome(
            rc=0, signal_name="", text="", saw_result=True,
            pending_tools=0, stream_closed_msg=False,
        )

    monkeypatch.setattr(agent_mod, "_run_once", _fake_run_once)
    monkeypatch.setattr(agent_mod, "get_backend", lambda name: _StubBackend())

    rc, _ = asyncio.run(agent_mod.run_agent(
        model="m",
        prompt="my-original-task",
        tag="[t]",
        workspace="/tmp",
        agent_cmd="x",
        extra_flags=[],
        backend="codex",
        max_resume_attempts=1,
    ))
    assert rc == 0
    assert captured["prompt"] == "my-original-task"
