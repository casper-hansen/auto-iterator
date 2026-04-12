#!/usr/bin/env python3
"""review-loop.py — Automated implement → review → fix loop using the Cursor CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pty
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Colors ───────────────────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ

BOLD = "\033[1m" if _USE_COLOR else ""
DIM = "\033[2m" if _USE_COLOR else ""
RED = "\033[0;31m" if _USE_COLOR else ""
GREEN = "\033[0;32m" if _USE_COLOR else ""
YELLOW = "\033[0;33m" if _USE_COLOR else ""
BLUE = "\033[0;34m" if _USE_COLOR else ""
CYAN = "\033[0;36m" if _USE_COLOR else ""
MAGENTA = "\033[0;35m" if _USE_COLOR else ""
NC = "\033[0m" if _USE_COLOR else ""

# Defaults aligned with currently available Cursor models.
DEFAULT_IMPL_MODEL = "claude-4.6-opus-high"
DEFAULT_FIX_MODEL = "claude-4.6-opus-high"
DEFAULT_REVIEWER_MODEL = "gpt-5.4-xhigh"

# ── Timestamped logging ──────────────────────────────────────────────────────


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str, tag: str = "") -> None:
    prefix = f"{DIM}{_ts()}{NC} {tag} " if tag else f"{DIM}{_ts()}{NC} "
    print(f"{prefix}{BLUE}▸{NC} {msg}", flush=True)


def ok(msg: str, tag: str = "") -> None:
    prefix = f"{DIM}{_ts()}{NC} {tag} " if tag else f"{DIM}{_ts()}{NC} "
    print(f"{prefix}{GREEN}✓{NC} {msg}", flush=True)


def warn(msg: str, tag: str = "") -> None:
    prefix = f"{DIM}{_ts()}{NC} {tag} " if tag else f"{DIM}{_ts()}{NC} "
    print(f"{prefix}{YELLOW}⚠{NC} {msg}", flush=True)


def err(msg: str) -> None:
    print(f"{DIM}{_ts()}{NC} {RED}✗{NC} {msg}", file=sys.stderr, flush=True)


def hr() -> None:
    print(f"{DIM}{'─' * 72}{NC}", flush=True)


def make_tag(outer: int, inner: int) -> str:
    return f"{BOLD}[Outer {outer}, Inner {inner}]{NC}"


def parse_verdict(text: str) -> str:
    matches = re.findall(r"VERDICT:\s*(APPROVED|CHANGES_NEEDED)", text)
    return matches[-1] if matches else "UNKNOWN"


# ── Output formatter ─────────────────────────────────────────────────────────

_THINKING_OPEN = re.compile(r"<(?:antml:)?thinking>")
_THINKING_CLOSE = re.compile(r"</(?:antml:)?thinking>")
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\].*?\x07")


class OutputFormatter:
    """Streams agent output, dimming reasoning and printing messages plainly."""

    THINKING = "thinking"
    ASSISTANT = "assistant"

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self._state = self.ASSISTANT

    def feed(self, line: str) -> None:
        ts = f"{DIM}{_ts()}{NC}"

        if _THINKING_OPEN.search(line):
            self._state = self.THINKING
            after = _THINKING_OPEN.sub("", line).strip()
            if after:
                print(f"{ts} {self.tag}   {DIM}{after}{NC}", flush=True)
            return

        if _THINKING_CLOSE.search(line):
            before = _THINKING_CLOSE.sub("", line).strip()
            if before:
                print(f"{ts} {self.tag}   {DIM}{before}{NC}", flush=True)
            self._state = self.ASSISTANT
            return

        if self._state == self.THINKING:
            print(f"{ts} {self.tag}   {DIM}{line}{NC}", flush=True)
        else:
            print(f"{ts} {self.tag}   {line}", flush=True)

    def feed_tool(self, line: str) -> None:
        """Print a tool-call line, always dimmed to reduce visual noise."""
        ts = f"{DIM}{_ts()}{NC}"
        print(f"{ts} {self.tag}     {DIM}{line}{NC}", flush=True)

    def flush(self) -> None:
        self._state = self.ASSISTANT


# ── Agent runner ─────────────────────────────────────────────────────────────


async def run_agent(
    *,
    mode: str,
    model: str,
    prompt: str,
    tag: str,
    workspace: str,
    agent_cmd: str,
    extra_flags: list[str],
) -> tuple[int, str]:
    """Launch the Cursor agent CLI with stream-json output.

    Streams tool-call and assistant-text events to the console as they
    arrive, so the log file is populated in real time.

    Returns (exit_code, captured_full_text).
    """
    cmd: list[str] = [
        agent_cmd, "-p",
        "--output-format", "stream-json",
        "--stream-partial-output",
        "--model", model,
        "--workspace", workspace,
        "--trust",
    ]
    if mode == "write":
        cmd.append("--force")
    cmd.extend(extra_flags)
    cmd.append(prompt)

    master_fd, slave_fd = pty.openpty()

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
    )
    os.close(slave_fd)

    full_text: list[str] = []
    fmt = OutputFormatter(tag)
    text_buf: list[str] = []

    def _flush_text() -> None:
        """Flush any accumulated partial text to the formatter."""
        if not text_buf:
            return
        assembled = "".join(text_buf)
        text_buf.clear()
        for line in assembled.split("\n"):
            fmt.feed(line)
        sys.stdout.flush()

    def _flush_complete_lines() -> None:
        """Flush only complete lines (up to last newline) from text_buf."""
        combined = "".join(text_buf)
        if "\n" not in combined:
            return
        parts = combined.split("\n")
        for l in parts[:-1]:
            fmt.feed(l)
        text_buf.clear()
        if parts[-1]:
            text_buf.append(parts[-1])
        sys.stdout.flush()

    def _read_pty() -> None:
        """Blocking reader in a thread — parses stream-json events."""
        buf = b""
        while True:
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                raw_line, buf = buf.split(b"\n", 1)
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r")
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    cleaned = _ANSI_RE.sub("", line).strip()
                    if cleaned:
                        fmt.feed(cleaned)
                        sys.stdout.flush()
                    continue
                _handle_event(evt)
        if buf:
            leftover = buf.decode("utf-8", errors="replace").rstrip("\r")
            if leftover:
                try:
                    evt = json.loads(leftover)
                    _handle_event(evt)
                except json.JSONDecodeError:
                    cleaned = _ANSI_RE.sub("", leftover).strip()
                    if cleaned:
                        fmt.feed(cleaned)
                        sys.stdout.flush()
        try:
            os.close(master_fd)
        except OSError:
            pass

    def _handle_event(evt: dict) -> None:
        etype = evt.get("type", "")

        if etype == "assistant":
            content_parts = evt.get("message", {}).get("content", [])
            ts_ms = evt.get("timestamp_ms")
            has_model_call_id = "model_call_id" in evt
            is_partial = ts_ms is not None and not has_model_call_id

            if is_partial:
                for part in content_parts:
                    if part.get("type") == "text" and part["text"]:
                        text_buf.append(part["text"])
                _flush_complete_lines()
            else:
                _flush_text()
                assembled = "".join(
                    p.get("text", "")
                    for p in content_parts
                    if p.get("type") == "text"
                )
                if assembled:
                    full_text.append(assembled)

        elif etype == "tool_call":
            _flush_text()
            sub = evt.get("subtype", "")
            tc = evt.get("tool_call", {})
            if sub == "started":
                fmt.feed_tool(f"→ {_tool_summary(tc)}")
            elif sub == "completed":
                fmt.feed_tool(f"← {_tool_summary(tc, completed=True)}")

        elif etype == "result":
            _flush_text()
            result_text = evt.get("result", "")
            if result_text and not full_text:
                full_text.append(result_text)

    loop = asyncio.get_running_loop()
    read_task = loop.run_in_executor(None, _read_pty)
    await proc.wait()
    await read_task

    fmt.flush()
    return proc.returncode or 0, "\n".join(full_text)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _shorten_path(path: str, max_parts: int = 4) -> str:
    parts = path.rstrip("/").split("/")
    if len(parts) <= max_parts:
        return path
    return "…/" + "/".join(parts[-max_parts:])


def _tool_summary(tc: dict, *, completed: bool = False) -> str:
    """Return a compact one-line summary of a stream-json tool_call payload."""
    for key in tc:
        if not key.endswith("ToolCall"):
            continue
        name = key.replace("ToolCall", "")
        inner = tc[key]
        args = inner.get("args", {})
        if completed and "result" in inner:
            return _fmt_result(name, args, inner["result"])
        return _fmt_call(name, args)
    return _truncate(json.dumps(tc, separators=(",", ":")), 200)


# ── Tool call formatters (start) ─────────────────────────────────────────────

def _fmt_call(name: str, args: dict) -> str:  # noqa: C901
    """Human-readable one-liner for a tool invocation (all 20 tools)."""

    # 1. Shell
    if name == "shell":
        cmd = args.get("command", "")
        desc = args.get("description", "")
        label = desc or cmd
        return f"shell: {_truncate(label, 90)}"

    # 2. Glob
    if name == "glob":
        pat = args.get("globPattern", args.get("glob_pattern", ""))
        d = _shorten_path(args.get("targetDirectory", args.get("target_directory", "")))
        return f"glob: {pat}" + (f" in {d}" if d else "")

    # 3. Grep
    if name == "grep":
        pat = _truncate(args.get("pattern", ""), 40)
        p = _shorten_path(args.get("path", ""))
        gl = args.get("glob", "")
        suffix = f" [{gl}]" if gl else ""
        return f'grep: "{pat}"' + (f" in {p}" if p else "") + suffix

    # 4. Read
    if name == "read":
        p = _shorten_path(args.get("path", ""))
        extras = []
        if args.get("offset"):
            extras.append(f"L{args['offset']}")
        if args.get("limit"):
            extras.append(f"+{args['limit']}")
        return f"read: {p}" + (f" ({', '.join(extras)})" if extras else "")

    # 5. Delete
    if name == "delete":
        return f"delete: {_shorten_path(args.get('path', ''))}"

    # 6. StrReplace / edit
    if name in ("strReplace", "edit"):
        p = _shorten_path(args.get("path", args.get("filePath", "")))
        ra = " (all)" if args.get("replace_all") or args.get("replaceAll") else ""
        return f"edit: {p}{ra}"

    # 7. Write / createFile
    if name in ("write", "createFile"):
        p = _shorten_path(args.get("path", ""))
        size = len(args.get("contents", args.get("content", "")))
        return f"write: {p} ({size} chars)"

    # 8. EditNotebook
    if name == "editNotebook":
        nb = _shorten_path(args.get("targetNotebook", args.get("target_notebook", "")))
        idx = args.get("cellIdx", args.get("cell_idx", "?"))
        new = "new " if args.get("isNewCell", args.get("is_new_cell")) else ""
        return f"editNotebook: {nb} {new}cell {idx}"

    # 9. TodoWrite / updateTodos
    if name in ("todoWrite", "updateTodos"):
        n = len(args.get("todos", []))
        merge = args.get("merge", False)
        return f"todos: {'merge' if merge else 'replace'} {n} item(s)"

    # 10. ReadLints
    if name == "readLints":
        paths = args.get("paths", [])
        if paths:
            shown = ", ".join(_shorten_path(p) for p in paths[:3])
            extra = f" +{len(paths) - 3}" if len(paths) > 3 else ""
            return f"lints: {shown}{extra}"
        return "lints: (workspace)"

    # 11. SemanticSearch / codebaseSearch
    if name in ("semanticSearch", "codebaseSearch"):
        q = _truncate(args.get("query", ""), 60)
        dirs = args.get("targetDirectories", args.get("target_directories", []))
        where = ", ".join(_shorten_path(d) for d in dirs[:2]) if dirs else "all"
        return f'search: "{q}" in {where}'

    # 12. WebSearch
    if name == "webSearch":
        term = _truncate(args.get("searchTerm", args.get("search_term", "")), 60)
        return f'webSearch: "{term}"'

    # 13. WebFetch / urlFetch
    if name in ("webFetch", "urlFetch"):
        url = _truncate(args.get("url", ""), 80)
        return f"fetch: {url}"

    # 14. GenerateImage
    if name == "generateImage":
        desc = _truncate(args.get("description", ""), 60)
        return f'image: "{desc}"'

    # 15. AskQuestion
    if name == "askQuestion":
        qs = args.get("questions", [])
        title = args.get("title", "")
        label = title or f"{len(qs)} question(s)"
        return f"ask: {label}"

    # 16. Task
    if name == "task":
        desc = args.get("description", "?")
        model = args.get("model", "")
        sub = args.get("subagentType", args.get("subagent_type", ""))
        parts = [desc]
        if sub:
            parts.append(f"[{sub}]")
        if model:
            parts.append(f"({model})")
        return f"task: {' '.join(parts)}"

    # 17. Await
    if name == "await":
        tid = args.get("taskId", args.get("task_id", ""))
        ms = args.get("blockUntilMs", args.get("block_until_ms", ""))
        pat = args.get("pattern", "")
        parts = []
        if tid:
            parts.append(f"id={tid}")
        if ms:
            parts.append(f"{ms}ms")
        if pat:
            parts.append(f"/{_truncate(pat, 30)}/")
        return f"await: {' '.join(parts)}" if parts else "await"

    # 18. FetchMcpResource
    if name == "fetchMcpResource":
        srv = args.get("server", "")
        uri = _truncate(args.get("uri", ""), 60)
        return f"mcpResource: {srv} {uri}"

    # 19. CallMcpTool / mcpTool
    if name in ("callMcpTool", "mcpTool"):
        srv = args.get("server", "")
        tn = args.get("toolName", "")
        return f"mcp: {srv}/{tn}"

    # 20. SwitchMode
    if name == "switchMode":
        mode = args.get("targetModeId", args.get("target_mode_id", ""))
        expl = args.get("explanation", "")
        return f"switchMode → {mode}" + (f" ({_truncate(expl, 40)})" if expl else "")

    # MCP helpers
    if name == "listMcpResources":
        return "listMcpResources"

    # Unknown — show raw args truncated
    s = json.dumps(args, separators=(",", ":"))
    return f"{name}({_truncate(s, 200)})" if s != "{}" else name


# ── Tool result formatters (completed) ───────────────────────────────────────

def _fmt_result(name: str, args: dict, result: object) -> str:  # noqa: C901
    """Human-readable one-liner for a tool result (all 20 tools)."""
    if isinstance(result, dict):
        # Shell rejection (special top-level key)
        if "rejected" in result:
            reason = result["rejected"].get("reason", "")
            return f"shell ✗ rejected" + (f" ({reason})" if reason else "")

        s = result.get("success")
        if isinstance(s, dict):

            # 1. Shell
            if name == "shell":
                ec = s.get("exitCode", "?")
                sym = "✓" if ec == 0 else "✗"
                out = (s.get("stdout") or "").strip()
                first = out.split("\n")[0][:80] if out else ""
                return f"shell {sym} exit {ec}" + (f": {first}" if first else "")

            # 2. Glob
            if name == "glob":
                n = s.get("totalFiles", 0)
                files = s.get("files", [])
                shown = ", ".join(files[:3])
                extra = f" +{n - 3}" if n > 3 else ""
                return f"glob ✓ {n} file(s)" + (f": {shown}{extra}" if shown else "")

            # 3. Grep
            if name == "grep":
                pat = _truncate(args.get("pattern", ""), 30)
                total = s.get("totalMatchedLines", s.get("totalLines", None))
                if total is None:
                    for ws in (s.get("workspaceResults") or {}).values():
                        c = ws.get("content", {})
                        total = c.get("totalMatchedLines", c.get("totalLines"))
                        if total is not None:
                            break
                return f'grep ✓ {total if total is not None else "?"} match(es) for "{pat}"'

            # 4. Read
            if name == "read":
                p = _shorten_path(s.get("path", args.get("path", "")))
                if s.get("isEmpty"):
                    return f"read ✓ {p} (empty)"
                return f"read ✓ {p} ({s.get('totalLines', '?')} lines)"

            # 5. Delete
            if name == "delete":
                p = _shorten_path(args.get("path", ""))
                return f"delete ✓ {p}"

            # 6. StrReplace / edit
            if name in ("strReplace", "edit"):
                p = _shorten_path(s.get("path", args.get("path", "")))
                added = s.get("linesAdded", 0)
                removed = s.get("linesRemoved", 0)
                return f"edit ✓ {p} (+{added} −{removed})"

            # 7. Write / createFile
            if name in ("write", "createFile"):
                p = _shorten_path(s.get("path", args.get("path", "")))
                lines = s.get("totalLines", "?")
                return f"write ✓ {p} ({lines} lines)"

            # 8. EditNotebook
            if name == "editNotebook":
                nb = _shorten_path(args.get("targetNotebook", args.get("target_notebook", "")))
                return f"editNotebook ✓ {nb}"

            # 9. TodoWrite / updateTodos
            if name in ("todoWrite", "updateTodos"):
                return "todos ✓ updated"

            # 10. ReadLints
            if name == "readLints":
                n = len(s.get("diagnostics", s.get("lints", [])))
                return f"lints ✓ {n} diagnostic(s)"

            # 11. SemanticSearch / codebaseSearch
            if name in ("semanticSearch", "codebaseSearch"):
                n = len(s.get("results", s.get("chunks", [])))
                return f"search ✓ {n} result(s)"

            # 12. WebSearch
            if name == "webSearch":
                return "webSearch ✓"

            # 13. WebFetch / urlFetch
            if name in ("webFetch", "urlFetch"):
                return "fetch ✓"

            # 14. GenerateImage
            if name == "generateImage":
                return "image ✓"

            # 15. AskQuestion
            if name == "askQuestion":
                return "ask ✓ answered"

            # 16. Task
            if name == "task":
                return "task ✓ completed"

            # 17. Await
            if name == "await":
                return "await ✓"

            # 18. FetchMcpResource
            if name == "fetchMcpResource":
                return "mcpResource ✓"

            # 19. CallMcpTool / mcpTool
            if name in ("callMcpTool", "mcpTool"):
                tn = args.get("toolName", "mcp")
                return f"mcp ✓ {tn}"

            # 20. SwitchMode
            if name == "switchMode":
                mode = args.get("targetModeId", args.get("target_mode_id", ""))
                return f"switchMode ✓ → {mode}"

            # MCP helpers
            if name == "listMcpResources":
                return f"listMcpResources ✓ {len(s.get('resources', []))} resource(s)"

    # Unknown — show raw result truncated
    short = json.dumps(result, separators=(",", ":"))
    return f"{name} ⇒ {_truncate(short, 200)}"


# ── Prompt builders ──────────────────────────────────────────────────────────


def task_description(prompt: str, context: str) -> str:
    desc = prompt
    if context:
        desc += f"\n\nAdditional context:\n{context}"
    return desc


def _format_history(history: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for i, entry in enumerate(history, 1):
        label = "Review" if entry["role"] == "reviewer" else "Fix"
        parts.append(f"### Round {i} — {label}\n\n{entry['content']}")
    return "\n\n".join(parts)


def build_review_prompt(
    prompt: str, context: str, history: list[dict[str, str]]
) -> str:
    preamble = (
        "Inspect the git diff on our branch to main branch. "
        "Review if we have made an excellent implementation of the following:\n\n"
        f"{task_description(prompt, context)}"
    )
    if history:
        preamble += (
            "\n\n--- Review cycle history (oldest first) ---\n\n"
            f"{_format_history(history)}\n\n"
            "--- End of history ---\n\n"
            "Verify that previous concerns are fixed, but also check for "
            "regressions or any new issues in the overall diff."
        )
    return (
        f"{preamble}\n\n"
        "End your response with exactly one of:\n"
        "VERDICT: APPROVED\n"
        "VERDICT: CHANGES_NEEDED"
    )


def build_fix_prompt(
    prompt: str, context: str, history: list[dict[str, str]]
) -> str:
    return (
        "Fix these findings\n\n"
        "# Task\n\n"
        f"{task_description(prompt, context)}\n\n"
        "# Review cycle history\n\n"
        f"{_format_history(history)}\n\n"
        "Address the issues identified in the latest review above."
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="review-loop.py",
        description="Automated implement → review → fix loop using the Cursor CLI",
    )
    prompt_group = p.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Feature / task description")
    prompt_group.add_argument(
        "--prompt-file",
        help="Path to a UTF-8 text file containing the feature / task description",
    )
    context_group = p.add_mutually_exclusive_group()
    context_group.add_argument("--context", default="", help="Extra context for reviewers")
    context_group.add_argument(
        "--context-file",
        help="Path to a UTF-8 text file containing extra reviewer context",
    )
    p.add_argument("--impl-model", default=DEFAULT_IMPL_MODEL)
    p.add_argument("--fix-model", default=DEFAULT_FIX_MODEL)
    p.add_argument("--reviewer", default=DEFAULT_REVIEWER_MODEL, dest="reviewer_model")
    p.add_argument("--max-outer", type=int, default=10)
    p.add_argument("--max-inner", type=int, default=10)
    p.add_argument("--workspace", default=".")
    p.add_argument("--skip-impl", action="store_true")
    p.add_argument("--extra-flags", action="append", default=[])
    return p


async def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    agent_cmd = os.environ.get("AGENT_CMD", "agent")
    try:
        prompt = _load_text_arg(args.prompt, args.prompt_file, label="prompt")
        context = _load_text_arg(args.context, args.context_file, label="context")
    except OSError as exc:
        err(str(exc))
        return 1
    impl_model: str = args.impl_model
    fix_model: str = args.fix_model or impl_model
    reviewer_model: str = args.reviewer_model
    max_outer: int = args.max_outer
    max_inner: int = args.max_inner
    workspace: str = str(Path(args.workspace).resolve())
    skip_impl: bool = args.skip_impl
    extra_flags: list[str] = args.extra_flags

    if max_outer < 1:
        err(f"--max-outer must be a positive integer (got '{max_outer}')")
        return 1
    if max_inner < 1:
        err(f"--max-inner must be a positive integer (got '{max_inner}')")
        return 1

    if not await _command_exists(agent_cmd):
        err(f"Cursor agent CLI not found ('{agent_cmd}').")
        print("Install it with: curl https://cursor.com/install -fsSL | bash")
        return 1

    agent_kw = dict(
        workspace=workspace,
        agent_cmd=agent_cmd,
        extra_flags=extra_flags,
    )

    # ── Banner ───────────────────────────────────────────────────────────
    print()
    print(f"{DIM}{_ts()}{NC} {BOLD}Review Loop{NC}")
    hr()
    config = {
        "prompt": prompt,
        "context": context,
        "impl_model": impl_model,
        "fix_model": fix_model,
        "reviewer_model": reviewer_model,
        "max_outer": max_outer,
        "max_inner": max_inner,
        "workspace": workspace,
        "skip_impl": skip_impl,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    for k, v in config.items():
        label = k.replace("_", " ").ljust(16)
        print(f"{DIM}{_ts()}{NC}   {label}: {CYAN}{v}{NC}")
    hr()
    print()

    # ── Step 1: Implementation ───────────────────────────────────────────
    if not skip_impl:
        impl_tag = f"{BOLD}[Impl]{NC}"
        log(f"Implementing feature with {CYAN}{impl_model}{NC}", impl_tag)
        log(f"Prompt: {prompt[:120]}...", impl_tag)
        print()

        rc, _ = await run_agent(
            mode="write", model=impl_model, prompt=prompt,
            tag=impl_tag,
            **agent_kw,
        )
        if rc == 0:
            ok("Implementation complete", impl_tag)
        else:
            warn("Implementation agent exited with non-zero status", impl_tag)
        hr()
        print()
    else:
        log("Skipping implementation (--skip-impl)")
        print()

    # ── Step 2: Outer loop ───────────────────────────────────────────────
    approved = False
    total_reviews = 0
    verdict = ""
    outer = 0

    for outer in range(1, max_outer + 1):
        print(f"{DIM}{_ts()}{NC} {BOLD}Outer Loop {outer}/{max_outer} — fresh context{NC}")
        hr()

        history: list[dict[str, str]] = []
        inner = 0
        for inner in range(1, max_inner + 1):
            total_reviews += 1
            tag = make_tag(outer, inner)

            # ── Review ───────────────────────────────────────────────
            log(f"Review — {CYAN}{reviewer_model}{NC}", tag)

            review_prompt = build_review_prompt(prompt, context, history)
            rc, review_text = await run_agent(
                mode="readonly", model=reviewer_model, prompt=review_prompt,
                tag=tag,
                **agent_kw,
            )
            history.append({"role": "reviewer", "content": review_text})

            if rc == 0:
                verdict = parse_verdict(review_text)
                if verdict == "APPROVED":
                    ok(f"{GREEN}APPROVED{NC}", tag)
                elif verdict == "CHANGES_NEEDED":
                    warn(f"{YELLOW}CHANGES_NEEDED{NC}", tag)
                else:
                    warn("Could not parse verdict — treating as CHANGES_NEEDED", tag)
                    verdict = "CHANGES_NEEDED"
            else:
                warn("Reviewer agent exited with non-zero status", tag)
                verdict = "CHANGES_NEEDED"

            print()

            if verdict == "APPROVED":
                ok("Reviewer approved", tag)
                break

            if inner == max_inner:
                warn(f"Inner loop exhausted ({max_inner} iterations)", tag)
                break

            # ── Fix ──────────────────────────────────────────────────
            log(f"Fixing issues — {CYAN}{fix_model}{NC}", tag)

            fix_prompt = build_fix_prompt(prompt, context, history)
            rc, fix_text = await run_agent(
                mode="write", model=fix_model, prompt=fix_prompt,
                tag=tag,
                **agent_kw,
            )
            history.append({"role": "fixer", "content": fix_text})

            if rc == 0:
                ok("Fixes applied", tag)
            else:
                warn("Fix agent exited with non-zero status", tag)
            print()

        if verdict != "APPROVED":
            warn(
                f"Inner loop did not reach approval after {max_inner} "
                "iteration(s) — outer loop will retry with fresh context",
            )
            hr()
            print()
            continue

        if inner == 1:
            approved = True
            if outer == 1:
                ok("Approved on first pass")
            else:
                ok(f"Fresh-eyes review approved on outer loop {outer}")
            break

        ok(
            f"Inner loop converged after {inner} iteration(s) — "
            "starting fresh-eyes validation in next outer loop",
        )
        hr()
        print()

    if not approved:
        if verdict == "APPROVED":
            warn(
                f"Inner loop converged but MAX_OUTER ({max_outer}) exhausted "
                "without a clean fresh-eyes pass"
            )
        else:
            warn(
                f"Exhausted {max_outer} outer loop(s) without reaching approval"
            )

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    hr()
    print(f"{DIM}{_ts()}{NC} {BOLD}Summary{NC}")
    hr()
    summary = {
        "approved": approved,
        "total_reviews": total_reviews,
        "outer_loops": outer,
        "max_outer": max_outer,
        "max_inner": max_inner,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if approved:
        print(f"{DIM}{_ts()}{NC}   {GREEN}{BOLD}APPROVED{NC} — "
              f"{total_reviews} review(s), {outer} outer loop(s)")
    else:
        print(f"{DIM}{_ts()}{NC}   {YELLOW}{BOLD}NOT FULLY APPROVED{NC} — "
              f"{total_reviews} review(s), {outer} outer loop(s)")
    print(f"{DIM}{_ts()}{NC}   {DIM}{json.dumps(summary, indent=2)}{NC}")
    print()

    return 0 if approved else 1


async def _command_exists(cmd: str) -> bool:
    try:
        proc = await asyncio.create_subprocess_exec(
            "which", cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return proc.returncode == 0
    except OSError:
        return False


def _load_text_arg(
    inline_value: str | None,
    file_value: str | None,
    *,
    label: str,
) -> str:
    if file_value:
        path = Path(file_value).expanduser()
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise OSError(f"Could not read --{label}-file '{path}': {exc}") from exc
    return (inline_value or "").strip()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
