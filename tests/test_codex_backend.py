"""Codex backend — command building and event-stream compatibility.

These tests pin the backend's compatibility surface for the modern
codex-cli (0.124+, post openai/codex#4525) wire format:

* Initial / continue command shapes pin ``model_reasoning_effort=xhigh``
  and route stream-config flags as globals before subcommand-modifying
  flags.
* ``build_continue_cmd`` does not invoke ``codex exec resume`` — issue
  openai/codex#6717 makes ``--last`` reject any positional prompt, so
  the retry path re-runs ``codex exec`` from scratch with the original
  prompt prepended by a resume hint.
* ``handle_event`` decodes the ``thread.started`` / ``item.started`` /
  ``item.completed`` / ``turn.completed`` shape and ignores unknown
  events.
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


def test_build_initial_cmd_uses_xhigh_reasoning_effort() -> None:
    """``xhigh`` is the reasoning-effort value we pin for codex runs.

    ``xhigh`` is the topmost variant in codex-cli 0.124+'s
    ``ReasoningEffort`` enum; we always want max effort for the loop.
    """
    cmd = CodexBackend().build_initial_cmd(
        agent_cmd="codex",
        model="gpt-5.5",
        prompt="task",
        workspace="/tmp",
        extra_flags=[],
    )
    assert 'model_reasoning_effort="xhigh"' in cmd


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
    # Global stream-config flags must come before any subcommand-modifying
    # flags so codex-cli parses them at the global level.
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

    Issue openai/codex#6717 makes ``--last`` reject any positional prompt.
    We sidestep it by re-running ``codex exec`` with the original prompt
    prepended by the resume hint.
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


def test_build_continue_cmd_uses_xhigh_reasoning_effort() -> None:
    """Same reasoning-effort pin as the initial command."""
    cmd = CodexBackend().build_continue_cmd(
        agent_cmd="codex",
        model="gpt-5.5",
        prompt="t",
        workspace="/tmp",
        extra_flags=[],
    )
    assert 'model_reasoning_effort="xhigh"' in cmd


# ── Modern (>=0.124) event format ────────────────────────────────────────────


def test_handle_event_agent_message() -> None:
    """``item.completed`` with ``agent_message`` populates ``_full_text``."""
    reader = _FakeReader()
    CodexBackend().handle_event(
        {"type": "item.completed",
         "item": {"id": "x", "type": "agent_message", "text": "hello"}},
        reader,
    )
    assert reader._full_text == ["hello"]
    assert "hello" in reader._fmt.text_lines


def test_handle_event_tool_pairs_pending() -> None:
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


def test_handle_event_turn_completed_marks_result() -> None:
    reader = _FakeReader()
    CodexBackend().handle_event(
        {"type": "turn.completed",
         "usage": {"input_tokens": 1, "cached_input_tokens": 0,
                   "output_tokens": 1}},
        reader,
    )
    assert reader.saw_result is True


def test_handle_event_unknown_event_is_ignored() -> None:
    """Unrecognised events must not crash the dispatcher.

    codex-cli emits a long tail of events we don't model (token counts,
    reasoning summaries, MCP list responses, …); the loop is happy as
    long as nothing raises.
    """
    reader = _FakeReader()
    CodexBackend().handle_event(
        {"type": "thread.started", "thread_id": "abc"}, reader,
    )
    CodexBackend().handle_event({}, reader)
    CodexBackend().handle_event({"type": "turn.started"}, reader)


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
