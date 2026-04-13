"""Async runner for the Cursor agent CLI with stream-json output."""

from __future__ import annotations

import asyncio
import json
import os
import pty
import sys

from .logging import warn
from .output_formatter import OutputFormatter, _ANSI_RE
from .tool_formatter import tool_summary

_STREAM_CLOSED = "WritableIterable is closed"

_RESUME_PROMPT = (
    "Your previous session was interrupted by a connection timeout. "
    "Please continue where you left off."
)


class _StreamReader:
    """Consumes newline-delimited JSON from a PTY fd, routing events to an
    OutputFormatter while accumulating the full assistant text."""

    def __init__(self, fmt: OutputFormatter) -> None:
        self._fmt = fmt
        self._text_buf: list[str] = []
        self._full_text: list[str] = []
        self.stream_closed = False

    @property
    def full_text(self) -> str:
        return "\n".join(self._full_text)

    # ── Partial-text buffering ────────────────────────────────────────────

    def _flush_text(self) -> None:
        if not self._text_buf:
            return
        assembled = "".join(self._text_buf)
        self._text_buf.clear()
        for line in assembled.split("\n"):
            self._fmt.feed(line)
        sys.stdout.flush()

    def _flush_complete_lines(self) -> None:
        combined = "".join(self._text_buf)
        if "\n" not in combined:
            return
        parts = combined.split("\n")
        for line in parts[:-1]:
            self._fmt.feed(line)
        self._text_buf.clear()
        if parts[-1]:
            self._text_buf.append(parts[-1])
        sys.stdout.flush()

    # ── Event dispatch ────────────────────────────────────────────────────

    def _handle_event(self, evt: dict) -> None:
        etype = evt.get("type", "")

        if etype == "assistant":
            content_parts = evt.get("message", {}).get("content", [])
            ts_ms = evt.get("timestamp_ms")
            is_partial = ts_ms is not None and "model_call_id" not in evt

            if is_partial:
                for part in content_parts:
                    if part.get("type") == "text" and part["text"]:
                        self._text_buf.append(part["text"])
                self._flush_complete_lines()
            else:
                self._flush_text()
                assembled = "".join(
                    p.get("text", "")
                    for p in content_parts
                    if p.get("type") == "text"
                )
                if assembled:
                    self._full_text.append(assembled)

        elif etype == "tool_call":
            self._flush_text()
            sub = evt.get("subtype", "")
            tc = evt.get("tool_call", {})
            if sub == "started":
                self._fmt.feed_tool(f"→ {tool_summary(tc)}")
            elif sub == "completed":
                self._fmt.feed_tool(f"← {tool_summary(tc, completed=True)}")

        elif etype == "result":
            self._flush_text()
            result_text = evt.get("result", "")
            if result_text and not self._full_text:
                self._full_text.append(result_text)

    # ── PTY reader (runs in a thread via run_in_executor) ─────────────────

    def read_pty(self, master_fd: int) -> None:
        """Blocking loop: read from *master_fd*, parse stream-json events."""
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
                        if _STREAM_CLOSED in cleaned:
                            self.stream_closed = True
                        self._fmt.feed(cleaned)
                        sys.stdout.flush()
                    continue
                self._handle_event(evt)

        if buf:
            leftover = buf.decode("utf-8", errors="replace").rstrip("\r")
            if leftover:
                try:
                    self._handle_event(json.loads(leftover))
                except json.JSONDecodeError:
                    cleaned = _ANSI_RE.sub("", leftover).strip()
                    if cleaned:
                        self._fmt.feed(cleaned)
                        sys.stdout.flush()
        try:
            os.close(master_fd)
        except OSError:
            pass


# ── Public API ────────────────────────────────────────────────────────────────


def _build_cmd(
    agent_cmd: str,
    model: str,
    prompt: str,
    workspace: str,
    extra_flags: list[str],
) -> list[str]:
    cmd = [
        agent_cmd, "-p",
        "--output-format", "stream-json",
        "--stream-partial-output",
        "--model", model,
        "--workspace", workspace,
        "--trust",
        "--force",
    ]
    cmd.extend(extra_flags)
    cmd.append(prompt)
    return cmd


def _build_cmd_continue(
    agent_cmd: str,
    model: str,
    workspace: str,
    extra_flags: list[str],
) -> list[str]:
    """Build a CLI invocation that resumes the most recent session."""
    cmd = [
        agent_cmd, "-p",
        "--output-format", "stream-json",
        "--stream-partial-output",
        "--model", model,
        "--workspace", workspace,
        "--trust",
        "--force",
        "--continue",
    ]
    cmd.extend(extra_flags)
    cmd.append(_RESUME_PROMPT)
    return cmd


async def _run_once(
    cmd: list[str],
    tag: str,
) -> tuple[int, str, bool]:
    """Run a single agent session. Returns *(exit_code, text, stream_closed)*."""
    master_fd, slave_fd = pty.openpty()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
    )
    os.close(slave_fd)

    reader = _StreamReader(OutputFormatter(tag))

    loop = asyncio.get_running_loop()
    read_task = loop.run_in_executor(None, reader.read_pty, master_fd)
    await proc.wait()
    await read_task

    reader._fmt.flush()
    return proc.returncode or 0, reader.full_text, reader.stream_closed


async def run_agent(
    *,
    model: str,
    prompt: str,
    tag: str,
    workspace: str,
    agent_cmd: str,
    extra_flags: list[str],
) -> tuple[int, str]:
    """Launch the Cursor agent CLI and stream its output.

    If the server-side gRPC stream closes (``WritableIterable is closed``),
    the session is automatically resumed via ``--continue`` until the agent
    finishes naturally.

    Returns *(exit_code, captured_full_text)*.
    """
    all_text: list[str] = []
    attempt = 0

    while True:
        if attempt == 0:
            cmd = _build_cmd(agent_cmd, model, prompt, workspace, extra_flags)
        else:
            cmd = _build_cmd_continue(agent_cmd, model, workspace, extra_flags)

        rc, text, stream_closed = await _run_once(cmd, tag)

        if text:
            all_text.append(text)

        if stream_closed:
            attempt += 1
            warn(
                f"Stream closed, auto-resuming (attempt {attempt + 1})…",
                tag,
            )
            continue

        break

    return rc, "\n".join(all_text)
