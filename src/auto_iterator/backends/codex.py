"""OpenAI Codex CLI backend.

Drives ``codex exec --json`` and translates the Codex event stream into
``_StreamReader`` state the loop inspects.

Two event-stream shapes are supported because the ``codex exec --json``
format changed between codex-cli minor versions:

* **Modern** (codex-cli 0.124+, after openai/codex#4525): top-level events
  ``thread.started`` / ``turn.started`` / ``turn.completed`` / ``turn.failed``
  and item events ``item.started`` / ``item.updated`` / ``item.completed``
  with item types ``agent_message``, ``reasoning``, ``command_execution``,
  ``file_change``, ``mcp_tool_call``, ``web_search``, ``todo_list``.
* **Legacy** (codex-cli 0.36-era, pre-#4525): wire format is
  ``{"id": "...", "msg": {"type": "<EventMsg variant>", ...}}``
  with snake_case variants like ``task_started``, ``agent_message``,
  ``exec_command_begin`` / ``exec_command_end``, ``patch_apply_begin`` /
  ``patch_apply_end``, ``mcp_tool_call_begin`` / ``mcp_tool_call_end``,
  ``web_search_begin`` / ``web_search_end``, and ``task_complete``.

``handle_event`` dispatches on whichever shape arrives so the same backend
works against both ``codex --version`` ranges without operator config.
"""

from __future__ import annotations

import json

from ..tool_formatter import _shorten_path, _truncate

_RESUME_PROMPT = (
    "Your previous session ended unexpectedly before completing the task. "
    "Please continue where you left off."
)


# ── Modern (>=0.124) event helpers ───────────────────────────────────────────


def _item_start_summary(item: dict) -> str:
    """One-line label for the start of a Codex tool-like item."""
    itype = item.get("type", "?")

    if itype == "command_execution":
        cmd = item.get("command", "")
        return f"shell: {_truncate(cmd, 90)}"

    if itype == "file_change":
        changes = item.get("changes", []) or []
        if not changes:
            return "file_change"
        if len(changes) == 1:
            c = changes[0]
            kind = c.get("kind", "change")
            return f"{kind}: {_shorten_path(c.get('path', ''))}"
        first = _shorten_path(changes[0].get("path", ""))
        return f"file_change: {first} (+{len(changes) - 1} more)"

    if itype == "todo_list":
        n = len(item.get("items", []) or [])
        return f"todo_list: {n} item(s)"

    if itype == "web_search":
        q = _truncate(item.get("query", ""), 60)
        return f'web_search: "{q}"' if q else "web_search"

    if itype == "mcp_tool_call":
        server = item.get("server", "")
        tool = item.get("tool", item.get("name", ""))
        return f"mcp: {server}/{tool}" if server else f"mcp: {tool}"

    extras = {k: v for k, v in item.items() if k not in ("id", "type", "status")}
    s = json.dumps(extras, separators=(",", ":"))
    return f"{itype}({_truncate(s, 160)})" if s != "{}" else itype


def _item_complete_summary(item: dict) -> str:
    """One-line label for the completion of a Codex tool-like item."""
    itype = item.get("type", "?")
    sym = "✗" if item.get("status") == "failed" else "✓"

    if itype == "command_execution":
        ec = item.get("exit_code")
        ec_disp = "?" if ec is None else ec
        out = (item.get("aggregated_output") or "").strip()
        first = out.split("\n", 1)[0][:80] if out else ""
        head = f"shell {sym} exit {ec_disp}"
        return f"{head}: {first}" if first else head

    if itype == "file_change":
        changes = item.get("changes", []) or []
        if len(changes) == 1:
            c = changes[0]
            return f"{c.get('kind', 'change')} {sym} {_shorten_path(c.get('path', ''))}"
        return f"file_change {sym} {len(changes)} file(s)"

    return f"{itype} {sym}"


def _is_tool_item(itype: str) -> bool:
    """True if the item represents a tool call (not an assistant message)."""
    return itype not in ("agent_message", "reasoning")


# ── Legacy (codex 0.36-era) event helpers ────────────────────────────────────

# In the legacy stream, each event is ``{"id": "...", "msg": {"type": ...}}``
# with EventMsg variants serialized in snake_case. Begin/end variants pair
# up via ``call_id``; we only care about counting, not pairing, so the id
# matters less here than in the modern stream.

_LEGACY_BEGIN = ("exec_command_begin", "patch_apply_begin",
                 "mcp_tool_call_begin", "web_search_begin")
_LEGACY_END = ("exec_command_end", "patch_apply_end",
               "mcp_tool_call_end", "web_search_end")


def _legacy_begin_summary(mtype: str, msg: dict) -> str:
    """One-line label for the start of a legacy tool-like event."""
    if mtype == "exec_command_begin":
        cmd = msg.get("command") or []
        if isinstance(cmd, list):
            label = " ".join(str(c) for c in cmd)
        else:
            label = str(cmd)
        return f"shell: {_truncate(label, 90)}"
    if mtype == "patch_apply_begin":
        changes = msg.get("changes") or {}
        # ``changes`` is a {path: FileChange} map; show the first key.
        if isinstance(changes, dict) and changes:
            paths = list(changes.keys())
            first = _shorten_path(str(paths[0]))
            extra = f" (+{len(paths) - 1} more)" if len(paths) > 1 else ""
            return f"patch: {first}{extra}"
        return "patch"
    if mtype == "mcp_tool_call_begin":
        # The invocation lives under ``invocation`` in the legacy event.
        inv = msg.get("invocation") or {}
        server = inv.get("server", "")
        tool = inv.get("tool", "")
        return f"mcp: {server}/{tool}" if server else f"mcp: {tool}"
    if mtype == "web_search_begin":
        q = _truncate(msg.get("query", ""), 60)
        return f'web_search: "{q}"' if q else "web_search"
    return mtype


def _legacy_end_summary(mtype: str, msg: dict) -> str:
    """One-line label for the completion of a legacy tool-like event."""
    if mtype == "exec_command_end":
        ec = msg.get("exit_code")
        ec_disp = "?" if ec is None else ec
        sym = "✓" if ec == 0 else "✗"
        out = (msg.get("stdout") or "").strip()
        first = out.split("\n", 1)[0][:80] if out else ""
        head = f"shell {sym} exit {ec_disp}"
        return f"{head}: {first}" if first else head
    if mtype == "patch_apply_end":
        sym = "✓" if msg.get("success", True) else "✗"
        return f"patch {sym}"
    if mtype == "mcp_tool_call_end":
        sym = "✗" if msg.get("is_error") else "✓"
        return f"mcp {sym}"
    if mtype == "web_search_end":
        return "web_search ✓"
    return f"{mtype} ✓"


class CodexBackend:
    name = "codex"
    default_cmd = "codex"
    display_name = "OpenAI Codex CLI"
    install_hint = (
        "Install it from https://github.com/openai/codex (ships as `codex`). "
        "Tested against codex-cli 0.36 and 0.124+; both event-stream formats "
        "are supported."
    )

    # Codex only runs OpenAI-family models; ``gpt-5.5`` is the current
    # flagship. Effort is pinned to ``high`` via ``_BASE_FLAGS`` below —
    # the Codex CLI doesn't expose a dedicated ``--effort`` flag, so we
    # route it through ``-c model_reasoning_effort="high"`` to mirror what
    # Cursor's ``-thinking-max`` suffix and Claude Code's ``--effort max``
    # do. ``high`` is the topmost variant in codex-cli 0.36's
    # ``ReasoningEffort`` enum (``minimal``/``low``/``medium``/``high``);
    # later codex builds added ``xhigh`` but we keep ``high`` so a single
    # backend works against both. Overridable per-call via
    # ``extra_flags`` / ``--impl-model`` / ``--reviewer`` on the CLI
    # frontends (later flags win).
    default_impl_model = "gpt-5.5"
    default_fix_model = "gpt-5.5"
    default_reviewer_model = "gpt-5.5"
    default_experimenter_model = "gpt-5.5"
    default_adjuster_model = "gpt-5.5"
    default_analyst_model = "gpt-5.5"

    # ``--skip-git-repo-check`` lets Codex run in workspaces that aren't git
    # repositories (matches the looser contract of the other backends).
    # ``--dangerously-bypass-approvals-and-sandbox`` is the Codex equivalent
    # of Cursor's ``--trust`` / Claude Code's ``--dangerously-skip-permissions``
    # — required for unattended loop runs.
    # ``-c model_reasoning_effort="high"`` pins the model to its highest
    # reasoning effort that's accepted across all supported codex-cli
    # versions; quoting the value keeps it as a TOML string.
    _BASE_FLAGS = (
        "--json",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "-c", 'model_reasoning_effort="high"',
    )

    def build_initial_cmd(
        self,
        agent_cmd: str,
        model: str,
        prompt: str,
        workspace: str,
        extra_flags: list[str],
    ) -> list[str]:
        # ``-C`` is accepted by ``codex exec`` but not ``codex exec resume``;
        # pass it only on the initial invocation. The subprocess also sets
        # ``cwd=workspace`` so resume inherits the same directory, which is
        # how ``resume --last`` selects the correct session.
        cmd = [
            agent_cmd, "exec",
            *self._BASE_FLAGS,
            "--model", model,
            "-C", workspace,
        ]
        cmd.extend(extra_flags)
        cmd.append(prompt)
        return cmd

    def build_continue_cmd(
        self,
        agent_cmd: str,
        model: str,
        prompt: str,
        workspace: str,
        extra_flags: list[str],
    ) -> list[str]:
        # codex-cli's ``exec resume`` subcommand has two version-dependent
        # quirks that make it unreliable for unattended retries:
        #
        #   * 0.36-era builds reject global flags (``--json``, ``--model``,
        #     ``--skip-git-repo-check``, ``--dangerously-bypass-…``) when
        #     they appear *after* ``resume`` (PR openai/codex#8440 made
        #     these flags global, but landed in 0.124+).
        #   * Both 0.36 and 0.63 hit a clap-arg conflict whenever a positional
        #     prompt accompanies ``--last`` (issue openai/codex#6717): clap
        #     fills ``[SESSION_ID]`` with the prompt and rejects it.
        #
        # The retry path is only entered after the *previous* attempt died
        # abnormally, so we sidestep the resume subcommand entirely and
        # re-run ``codex exec`` from scratch with the original prompt. We
        # lose mid-task continuity, but the outer review loop already
        # rebuilds context across calls — what matters is that the retry
        # actually runs to completion instead of erroring on flag parsing.
        cmd = [
            agent_cmd, "exec",
            *self._BASE_FLAGS,
            "--model", model,
            "-C", workspace,
        ]
        cmd.extend(extra_flags)
        # Pre-pend the resume hint to the original task so the agent knows
        # this is a recovery attempt, not a fresh first try. Concatenating
        # in this direction keeps any task-specific instructions intact.
        cmd.append(f"{_RESUME_PROMPT}\n\n{prompt}")
        return cmd

    def handle_event(self, evt: dict, reader) -> None:
        # Modern stream events have a top-level ``type`` field; legacy
        # events nest the variant under ``msg.type``. Dispatch on whichever
        # shape arrived so we can ride out the codex-cli version range
        # without operator config.
        if evt.get("type"):
            self._handle_modern_event(evt, reader)
            return
        if isinstance(evt.get("msg"), dict):
            self._handle_legacy_event(evt, reader)

    def _handle_modern_event(self, evt: dict, reader) -> None:
        etype = evt.get("type", "")

        if etype == "item.started":
            item = evt.get("item") or {}
            if _is_tool_item(item.get("type", "")):
                reader._flush_text()
                reader.pending_tools += 1
                reader._fmt.feed_tool(f"→ {_item_start_summary(item)}")
            return

        if etype == "item.completed":
            item = evt.get("item") or {}
            itype = item.get("type", "")
            if itype == "agent_message":
                reader._flush_text()
                text = item.get("text", "")
                if text:
                    reader._full_text.append(text)
                    # Codex doesn't emit incremental text deltas, so the
                    # whole assistant message arrives here. Echo it to the
                    # console so the user sees it during the run.
                    for line in text.split("\n"):
                        reader._fmt.feed(line)
                return
            if _is_tool_item(itype):
                reader._flush_text()
                reader.pending_tools = max(0, reader.pending_tools - 1)
                reader._fmt.feed_tool(f"← {_item_complete_summary(item)}")
            return

        if etype == "turn.completed":
            reader._flush_text()
            reader.saw_result = True
            return

    def _handle_legacy_event(self, evt: dict, reader) -> None:
        msg = evt.get("msg") or {}
        mtype = msg.get("type", "")

        if mtype == "agent_message":
            # Codex 0.36 suppresses agent_message_delta in JSON mode and
            # emits the whole assistant message in one shot here, so we
            # treat it as authoritative final output (mirrors how the
            # modern stream's ``item.completed.agent_message`` is handled).
            reader._flush_text()
            text = msg.get("message", "")
            if text:
                reader._full_text.append(text)
                for line in text.split("\n"):
                    reader._fmt.feed(line)
            return

        if mtype in _LEGACY_BEGIN:
            reader._flush_text()
            reader.pending_tools += 1
            reader._fmt.feed_tool(f"→ {_legacy_begin_summary(mtype, msg)}")
            return

        if mtype in _LEGACY_END:
            reader._flush_text()
            reader.pending_tools = max(0, reader.pending_tools - 1)
            reader._fmt.feed_tool(f"← {_legacy_end_summary(mtype, msg)}")
            return

        if mtype == "task_complete":
            # Legacy ``saw_result`` signal. ``last_agent_message`` is a
            # fallback in case the agent_message event was missed (e.g.
            # the stream was truncated mid-message).
            reader._flush_text()
            reader.saw_result = True
            last = msg.get("last_agent_message") or ""
            if last and not reader._full_text:
                reader._full_text.append(last)
            return
