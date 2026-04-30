"""Async runner for a pluggable CLI agent backend with stream-json output."""

from __future__ import annotations

import asyncio
import json
import os
import pty
import signal as signal_mod
import sys
from dataclasses import dataclass

from .backends import get_backend
from .logging import warn
from .output_formatter import OutputFormatter, _ANSI_RE

_STREAM_CLOSED = "WritableIterable is closed"

_DEFAULT_MAX_RESUME_ATTEMPTS = 3

_DEFAULT_BACKEND = "cursor"


@dataclass
class _RunOutcome:
    """Result of a single ``_run_once`` invocation.

    Captures every scrap of diagnostic data we can reasonably collect so
    the caller can decide whether to resume and can surface a real error
    message instead of a vague 'non-zero status'.
    """

    rc: int                  # Popen-style returncode; negative values = -signal
    signal_name: str         # e.g. "SIGKILL", empty when not killed by a signal
    text: str                # Accumulated assistant text across the session
    saw_result: bool         # Did the CLI emit a stream-json `result` event?
    pending_tools: int       # Tool calls still in-flight at exit
    stream_closed_msg: bool  # Did we see an explicit "WritableIterable is closed"?

    @property
    def clean_exit(self) -> bool:
        return self.rc == 0

    @property
    def rc_display(self) -> str:
        """Human-readable exit status, e.g. ``'137 (SIGKILL)'`` or ``'1'``."""
        if self.rc < 0:
            name = self.signal_name or f"signal {-self.rc}"
            # Match shell convention (128 + signal number) when reporting.
            return f"{128 + (-self.rc)} ({name})"
        return str(self.rc)

    @property
    def should_resume(self) -> bool:
        """True when the session died abnormally and resuming is worth trying."""
        if self.clean_exit:
            return False
        # Classic server-side kill signals previously handled explicitly.
        if self.stream_closed_msg or self.pending_tools > 0:
            return True
        # Killed by a signal (OOM, segfault, SIGTERM, …).
        if self.rc < 0:
            return True
        # Non-zero exit with no final `result` event — the agent ended dirty.
        return not self.saw_result


class _StreamReader:
    """Consumes newline-delimited JSON from a PTY fd, routing events via the
    backend while accumulating the full assistant text."""

    def __init__(self, fmt: OutputFormatter, backend) -> None:
        self._fmt = fmt
        self._backend = backend
        self._text_buf: list[str] = []
        self._full_text: list[str] = []
        self.stream_closed = False
        self.pending_tools = 0
        self.saw_result = False

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
                self._backend.handle_event(evt, self)

        if buf:
            leftover = buf.decode("utf-8", errors="replace").rstrip("\r")
            if leftover:
                try:
                    self._backend.handle_event(json.loads(leftover), self)
                except json.JSONDecodeError:
                    cleaned = _ANSI_RE.sub("", leftover).strip()
                    if cleaned:
                        if _STREAM_CLOSED in cleaned:
                            self.stream_closed = True
                        self._fmt.feed(cleaned)
                        sys.stdout.flush()

        # Surface any partial streaming text left over from a mid-line crash —
        # the CLI's last words, if it had any, live here and would otherwise be
        # silently discarded.
        self._flush_text()

        try:
            os.close(master_fd)
        except OSError:
            pass


# ── Public API ────────────────────────────────────────────────────────────────


def _signal_name_from_rc(rc: int) -> str:
    """Map a negative Popen-style returncode to its signal name."""
    if rc >= 0:
        return ""
    try:
        return signal_mod.Signals(-rc).name
    except (ValueError, AttributeError):
        return f"signal {-rc}"


async def _run_once(
    cmd: list[str],
    tag: str,
    backend,
    *,
    cwd: str | None = None,
) -> _RunOutcome:
    """Run a single agent session and collect everything the child emits.

    stdout and stderr are merged into a PTY so crash messages written to
    stderr still reach the reader.  Even when the child dies hard (signal,
    OOM, …) the returned outcome carries the exact exit status and the
    buffered output captured up to the moment of death.

    ``start_new_session=True`` places the child in a fresh session with no
    controlling terminal, so SIGHUP from a closed SSH / terminal can't
    reach it even if the wrapping ``nohup`` is incomplete (nohup only
    protects itself; child binaries that reinstall their own SIGHUP
    handler lose the inherited ``SIG_IGN``).
    """
    master_fd, slave_fd = pty.openpty()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
        start_new_session=True,
        cwd=cwd,
    )
    os.close(slave_fd)

    reader = _StreamReader(OutputFormatter(tag), backend)

    loop = asyncio.get_running_loop()
    read_task = loop.run_in_executor(None, reader.read_pty, master_fd)
    await proc.wait()
    await read_task

    reader._fmt.flush()
    rc = proc.returncode if proc.returncode is not None else 0
    return _RunOutcome(
        rc=rc,
        signal_name=_signal_name_from_rc(rc),
        text=reader.full_text,
        saw_result=reader.saw_result,
        pending_tools=reader.pending_tools,
        stream_closed_msg=reader.stream_closed,
    )


async def run_agent(
    *,
    model: str,
    prompt: str,
    tag: str,
    workspace: str,
    agent_cmd: str,
    extra_flags: list[str],
    backend: str = _DEFAULT_BACKEND,
    max_resume_attempts: int = _DEFAULT_MAX_RESUME_ATTEMPTS,
) -> tuple[int, str]:
    """Launch the configured agent CLI and stream its output.

    ``backend`` selects the CLI adapter ("cursor" or "claude-code").  The
    backend translates ``run_agent``'s generic kwargs into CLI flags and
    decodes the CLI's stream-json events into the ``_StreamReader`` state
    the loop inspects (text, tool counts, ``saw_result``).

    If the session ends abnormally — server-side kill, signal-induced death
    (SIGKILL / SIGSEGV / …), or any non-zero exit that did not produce a
    stream-json ``result`` event — the session is automatically resumed
    via the backend's continue command up to ``max_resume_attempts`` times.
    Every failure is logged with its exact exit code (and signal name, when
    applicable) so a broken shutdown is always visible instead of silently
    ignored.

    Returns ``(exit_code, captured_full_text)``.  ``exit_code`` is the rc
    of the final attempt — caller-visible and easy to include in warnings.
    """
    be = get_backend(backend)
    all_text: list[str] = []
    attempt = 0
    last_rc = 0

    while True:
        if attempt == 0:
            cmd = be.build_initial_cmd(agent_cmd, model, prompt, workspace, extra_flags)
        else:
            # ``prompt`` is threaded through so backends that can't reliably
            # use a CLI-side ``--continue`` (notably the codex backend, whose
            # ``exec resume`` subcommand has version-dependent flag-parsing
            # quirks) can fall back to re-running with the original task.
            cmd = be.build_continue_cmd(
                agent_cmd, model, prompt, workspace, extra_flags,
            )

        outcome = await _run_once(cmd, tag, be, cwd=workspace)
        last_rc = outcome.rc

        if outcome.text:
            all_text.append(outcome.text)

        if outcome.clean_exit:
            break

        # Always surface the real exit status up front — never silently continue.
        warn(
            f"Agent exited rc={outcome.rc_display} "
            f"(saw_result={outcome.saw_result}, "
            f"pending_tools={outcome.pending_tools}, "
            f"stream_closed={outcome.stream_closed_msg})",
            tag,
        )

        if outcome.should_resume and attempt < max_resume_attempts:
            attempt += 1
            warn(
                f"Session ended abnormally — auto-resuming "
                f"(attempt {attempt}/{max_resume_attempts})…",
                tag,
            )
            continue

        if outcome.should_resume:
            warn(
                f"Giving up after {max_resume_attempts} resume attempt(s); "
                f"agent is still exiting abnormally (last rc={outcome.rc_display}).",
                tag,
            )
        break

    return last_rc, "\n".join(all_text)
