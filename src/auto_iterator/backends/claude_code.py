"""Claude Code CLI backend.

Drives ``claude -p --output-format stream-json --include-partial-messages``
and translates the Anthropic-shaped stream back into ``_StreamReader`` state.

Event shapes consumed here (observed against claude 2.1.x):

* ``{"type": "system", ...}`` — init/status metadata, ignored.
* ``{"type": "stream_event", "event": {...}}`` — raw API event:
    - ``content_block_delta`` with ``delta.type == "text_delta"`` → partial text.
    - Other inner types (thinking deltas, block starts/stops) are ignored;
      we rely on the subsequent ``assistant`` event for the authoritative text.
* ``{"type": "assistant", "message": {"content": [...]}}`` — emitted once per
  content block *stop*. Each block is ``{type: "text" | "thinking" | "tool_use", ...}``.
  ``tool_use`` blocks mark the start of a tool call (pending++).
* ``{"type": "user", "message": {"content": [{"type": "tool_result", ...}]}}`` —
  a tool result arriving back from the runtime (pending--).
* ``{"type": "result", "subtype": "success" | "error_*", "result": "..."}`` —
  the final event; used as a fallback ``full_text`` source if no assistant
  text was captured (e.g. the agent only produced tool calls).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..tool_formatter import _shorten_path, _truncate

_RESUME_PROMPT = (
    "Your previous session ended unexpectedly before completing the task. "
    "Please continue where you left off."
)

# Review skill adapted from Claude Code's `/ultrareview` for local-branch
# review (no GitHub PR). Resolved relative to the repo root so edits to
# the markdown take effect on the next review call without a reinstall.
_SKILL_PATH = (
    Path(__file__).resolve().parents[3] / "skills" / "claude-review.md"
)

# Match the default reviewer's window — see `feature/prompts.py`.
_HISTORY_ROUNDS = 2
_HISTORY_ENTRIES = _HISTORY_ROUNDS * 2


def _tool_use_summary(block: dict) -> str:
    """One-line label for an assistant ``tool_use`` content block."""
    name = block.get("name", "?")
    inp = block.get("input", {}) or {}

    if name == "Bash":
        label = inp.get("description") or inp.get("command", "")
        return f"Bash: {_truncate(label, 90)}"
    if name == "Read":
        return f"Read: {_shorten_path(inp.get('file_path', ''))}"
    if name in ("Edit", "Write"):
        return f"{name}: {_shorten_path(inp.get('file_path', ''))}"
    if name == "NotebookEdit":
        return f"NotebookEdit: {_shorten_path(inp.get('notebook_path', ''))}"
    if name == "Glob":
        pat = inp.get("pattern", "")
        path = _shorten_path(inp.get("path", "")) if inp.get("path") else ""
        return f"Glob: {pat}" + (f" in {path}" if path else "")
    if name == "Grep":
        pat = _truncate(inp.get("pattern", ""), 40)
        path = _shorten_path(inp.get("path", "")) if inp.get("path") else ""
        return f'Grep: "{pat}"' + (f" in {path}" if path else "")
    if name == "WebFetch":
        return f"WebFetch: {_truncate(inp.get('url', ''), 80)}"
    if name == "WebSearch":
        return f'WebSearch: "{_truncate(inp.get("query", ""), 60)}"'
    if name == "Task":
        desc = inp.get("description", "")
        sub = inp.get("subagent_type", "")
        return f"Task: {desc}" + (f" [{sub}]" if sub else "")
    if name == "TodoWrite":
        n = len(inp.get("todos", []))
        return f"TodoWrite: {n} item(s)"
    if name == "Skill":
        return f"Skill: {inp.get('skill', '?')}"
    if name in ("TaskGet", "TaskStop", "TaskUpdate", "TaskOutput"):
        tid = inp.get("task_id", inp.get("shell_id", ""))
        return f"{name}: {tid}"

    s = json.dumps(inp, separators=(",", ":"))
    return f"{name}({_truncate(s, 160)})" if s != "{}" else name


def _render_operator_extras(guidance: list[str]) -> str:
    """Render operator-supplied guidance as a markdown block. Returns an
    empty string when there's no guidance so the skill template collapses
    cleanly."""
    if not guidance:
        return ""
    bullets = "\n".join(f"- {g}" for g in guidance)
    return (
        "## Operator guidance (apply on top of the task above)\n\n"
        f"{bullets}"
    )


def _render_history_block(history: list[dict[str, str]]) -> str:
    if not history:
        return ""
    recent = history[-_HISTORY_ENTRIES:]
    parts: list[str] = []
    for i, entry in enumerate(recent, 1):
        label = "Review" if entry["role"] == "reviewer" else "Fix"
        parts.append(f"### Round {i} — {label}\n\n{entry['content']}")
    body = "\n\n".join(parts)
    return (
        "## Previous review cycle history (oldest first)\n\n"
        "Verify prior concerns are fixed; also check for regressions or "
        "new issues anywhere in the diff.\n\n"
        f"{body}\n"
    )


def _tool_result_summary(block: dict) -> str:
    status = "✗" if block.get("is_error") else "✓"
    content = block.get("content")
    # ``content`` is either a string or a list of {type: "text", text: "..."}.
    if isinstance(content, list):
        parts = [c.get("text", "") for c in content if c.get("type") == "text"]
        snippet = " ".join(p.strip() for p in parts if p).split("\n", 1)[0]
    elif isinstance(content, str):
        snippet = content.split("\n", 1)[0]
    else:
        snippet = ""
    return f"tool_result {status}" + (f": {_truncate(snippet, 80)}" if snippet else "")


class ClaudeCodeBackend:
    name = "claude-code"
    default_cmd = "claude"
    display_name = "Claude Code CLI"
    install_hint = (
        "Install it with: curl -fsSL https://claude.ai/install.sh | bash"
    )

    # Claude Code only runs Claude models; ``opus`` resolves to the latest
    # Opus so defaults follow the model family forward without edits here.
    # ``--effort max`` (set in ``_BASE_FLAGS``) pairs with these to run the
    # biggest model at its highest reasoning effort. Overridable per-call
    # via ``extra_flags`` (later flags win).
    default_impl_model = "opus"
    default_fix_model = "opus"
    default_reviewer_model = "opus"
    default_experimenter_model = "opus"
    default_adjuster_model = "opus"
    default_analyst_model = "opus"

    # Workspace access comes from ``cwd=workspace`` on the subprocess —
    # ``--add-dir`` is not used here because it's variadic (``<directories...>``)
    # and would greedily swallow the trailing prompt positional.
    _BASE_FLAGS = (
        "-p",
        "--output-format", "stream-json",
        "--include-partial-messages",
        "--verbose",
        "--dangerously-skip-permissions",
        "--effort", "max",
    )

    def _base_cmd(
        self,
        agent_cmd: str,
        model: str,
    ) -> list[str]:
        return [
            agent_cmd,
            *self._BASE_FLAGS,
            "--model", model,
        ]

    def build_initial_cmd(
        self,
        agent_cmd: str,
        model: str,
        prompt: str,
        workspace: str,
        extra_flags: list[str],
    ) -> list[str]:
        cmd = self._base_cmd(agent_cmd, model)
        cmd.extend(extra_flags)
        cmd.append(prompt)
        return cmd

    def build_continue_cmd(
        self,
        agent_cmd: str,
        model: str,
        workspace: str,
        extra_flags: list[str],
    ) -> list[str]:
        cmd = self._base_cmd(agent_cmd, model)
        cmd.append("--continue")
        cmd.extend(extra_flags)
        cmd.append(_RESUME_PROMPT)
        return cmd

    def build_review_prompt(
        self,
        task: str,
        history: list[dict[str, str]],
        *,
        guidance: list[str] | None = None,
    ) -> str:
        """Review prompt driven by ``skills/claude-review.md``.

        Dispatches an ``/ultrareview``-style multi-agent review against
        the local branch diff, substituting the task, prior review
        history, and any operator guidance into the skill template.
        Guidance lands *before* the terminal "VERDICT on its own line,
        nothing after it" instruction so it doesn't defeat the skill's
        output contract."""
        skill = _SKILL_PATH.read_text(encoding="utf-8")
        return (
            skill
            .replace("{{TASK}}", task.strip())
            .replace("{{HISTORY_BLOCK}}", _render_history_block(history))
            .replace(
                "{{OPERATOR_EXTRAS_BLOCK}}",
                _render_operator_extras(guidance or []),
            )
        )

    def handle_event(self, evt: dict, reader) -> None:
        etype = evt.get("type", "")

        if etype == "stream_event":
            inner = evt.get("event") or {}
            if inner.get("type") == "content_block_delta":
                delta = inner.get("delta") or {}
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        reader._text_buf.append(text)
                        reader._flush_complete_lines()
            return

        if etype == "assistant":
            reader._flush_text()
            content = (evt.get("message") or {}).get("content") or []
            parts: list[str] = []
            for c in content:
                ctype = c.get("type")
                if ctype == "text":
                    parts.append(c.get("text", ""))
                elif ctype == "tool_use":
                    reader.pending_tools += 1
                    reader._fmt.feed_tool(f"→ {_tool_use_summary(c)}")
            assembled = "".join(parts)
            if assembled:
                reader._full_text.append(assembled)
            return

        if etype == "user":
            reader._flush_text()
            content = (evt.get("message") or {}).get("content") or []
            for c in content:
                if c.get("type") == "tool_result":
                    reader.pending_tools = max(0, reader.pending_tools - 1)
                    reader._fmt.feed_tool(f"← {_tool_result_summary(c)}")
            return

        if etype == "result":
            reader._flush_text()
            reader.saw_result = True
            if evt.get("is_error"):
                return
            result_text = evt.get("result", "")
            if result_text and not reader._full_text:
                reader._full_text.append(result_text)
