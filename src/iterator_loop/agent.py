"""Async runner for the Cursor agent CLI with stream-json output."""

from __future__ import annotations

import asyncio
import json
import os
import pty
import signal as signal_mod
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .logging import warn
from .output_formatter import OutputFormatter, _ANSI_RE
from .tool_formatter import tool_summary

#: Signature of an optional structured-event sink. Receivers get
#: ``(event_type, payload)`` and must not raise; any exception is
#: swallowed at the call site so a broken sink cannot kill the agent.
EventSink = Callable[[str, dict], None]

_STREAM_CLOSED = "WritableIterable is closed"

_DEFAULT_MAX_RESUME_ATTEMPTS = 3

_RESUME_PROMPT = (
    "Your previous session ended unexpectedly before completing the task. "
    "Please continue where you left off."
)


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
    # Count + first-error string for structured-sink failures observed
    # while the PTY reader was running. ``0`` / ``None`` means every
    # stream event reached the sink; any non-zero count means the
    # machine-readable log is known to be incomplete for this session.
    sink_errors: int = 0
    sink_first_error: Optional[str] = None

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
    """Consumes newline-delimited JSON from a PTY fd, routing events to an
    OutputFormatter while accumulating the full assistant text.

    When an *event_sink* is provided it is invoked for every observable
    stream event (``assistant_partial``, ``assistant_message``,
    ``tool_call_started``, ``tool_call_completed``, ``agent_result``,
    ``stream_closed``). The sink is called from this thread, so the
    receiver must be thread-safe.

    Broken sinks (``ENOSPC``, permission errors, bugs in the consumer)
    must not crash the PTY reader — the caller still needs a final
    outcome to decide whether to auto-resume — but they must also not
    silently corrupt the structured log. ``_emit`` therefore *counts*
    failures, remembers the first error string, and emits a single
    ``warn()`` per session so operators see that ``events.jsonl`` /
    ``state.json`` are degraded instead of finding out later that
    entries silently disappeared.
    """

    def __init__(
        self,
        fmt: OutputFormatter,
        event_sink: Optional[EventSink] = None,
        tag: str = "",
    ) -> None:
        self._fmt = fmt
        self._sink = event_sink
        self._tag = tag
        self._text_buf: list[str] = []
        self._full_text: list[str] = []
        self.stream_closed = False
        self.pending_tools = 0
        self.saw_result = False
        # Broken-sink accounting. ``sink_errors`` is the total number of
        # stream events we failed to deliver to the configured sink
        # during this session; ``sink_first_error`` captures the first
        # ``"<ExcType>: <message>"`` string so the outcome / caller can
        # report the exact cause without having to re-trigger the bug.
        self.sink_errors = 0
        self.sink_first_error: Optional[str] = None
        self._sink_warned = False

    # ── Sink dispatch ─────────────────────────────────────────────────────

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        """Forward *event_type* to the sink; surface failures visibly.

        Runs inside the PTY reader thread, so we must never let a broken
        sink abort the read loop — the caller still needs the final
        outcome to decide whether to auto-resume. But we do *not*
        silently drop the failure: ``self.sink_errors`` / ``self.
        sink_first_error`` are exposed via :class:`_RunOutcome` so a
        downstream caller can assert "the machine-readable log covered
        the whole session", and the first failure emits a ``warn()``
        so a human running the loop interactively sees the degradation.
        """
        sink = self._sink
        if sink is None:
            return
        try:
            sink(event_type, payload)
        except Exception as exc:
            self.sink_errors += 1
            if self.sink_first_error is None:
                self.sink_first_error = f"{type(exc).__name__}: {exc}"
            if not self._sink_warned:
                self._sink_warned = True
                warn(
                    f"run log sink failed on {event_type!r}: "
                    f"{self.sink_first_error} — structured log is "
                    "degraded (further sink failures suppressed in "
                    "this session)",
                    self._tag,
                )

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
                delta_parts: list[str] = []
                for part in content_parts:
                    if part.get("type") == "text" and part["text"]:
                        self._text_buf.append(part["text"])
                        delta_parts.append(part["text"])
                self._flush_complete_lines()
                if delta_parts:
                    self._emit(
                        "assistant_partial",
                        {"text": "".join(delta_parts)},
                    )
            else:
                self._flush_text()
                assembled = "".join(
                    p.get("text", "")
                    for p in content_parts
                    if p.get("type") == "text"
                )
                if assembled:
                    self._full_text.append(assembled)
                    self._emit("assistant_message", {"text": assembled})

        elif etype == "tool_call":
            self._flush_text()
            sub = evt.get("subtype", "")
            tc = evt.get("tool_call", {})
            tool_name = _tool_name(tc)
            if sub == "started":
                self.pending_tools += 1
                # The human-facing terminal renderer formats the tool
                # call inline, but ``events.jsonl`` gets the raw
                # ``tool_call`` payload so consumers (state.json
                # renderer, future TUI, post-mortem analysis) can
                # re-render it however they like instead of being
                # locked into the one-line summary format.
                self._fmt.feed_tool(f"→ {tool_summary(tc)}")
                self._emit(
                    "tool_call_started",
                    {"name": tool_name, "tool_call": tc},
                )
            elif sub == "completed":
                self.pending_tools = max(0, self.pending_tools - 1)
                self._fmt.feed_tool(f"← {tool_summary(tc, completed=True)}")
                self._emit(
                    "tool_call_completed",
                    {"name": tool_name, "tool_call": tc},
                )

        elif etype == "result":
            self._flush_text()
            self.saw_result = True
            result_text = evt.get("result", "")
            if result_text and not self._full_text:
                self._full_text.append(result_text)
            self._emit(
                "agent_result",
                {
                    "text": result_text,
                    "usage": evt.get("usage"),
                    "cost": evt.get("cost"),
                },
            )

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
                            self._emit("stream_closed", {"text": cleaned})
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
                        if _STREAM_CLOSED in cleaned:
                            self.stream_closed = True
                            self._emit("stream_closed", {"text": cleaned})
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


def _tool_name(tc: dict) -> str:
    """Best-effort extraction of a tool name from a stream-json tool_call."""
    for key in tc:
        if key.endswith("ToolCall"):
            return key.replace("ToolCall", "")
    name = tc.get("name")
    if isinstance(name, str):
        return name
    return "unknown"


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
    *,
    event_sink: Optional[EventSink] = None,
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
    # ``pty.openpty()`` returns two fds we own until the child inherits
    # them. If ``create_subprocess_exec`` raises (bad command path,
    # ENOENT, ENOMEM, …) *neither* fd is handed off, so both must be
    # closed here or they leak. Reviewer reproducer on Linux:
    # patching ``create_subprocess_exec`` to raise ``OSError`` grew
    # ``/proc/self/fd`` by exactly 2 per failed call (confirmed 9→11→
    # 13→15 after three attempts), meaning a long-running worker
    # that hits repeated spawn failures will eventually exhaust its
    # fd table before later runs even start. The ``try/except`` +
    # belt-and-suspenders individual closes below ensure neither fd
    # survives a failed spawn.
    master_fd, slave_fd = pty.openpty()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            start_new_session=True,
        )
    except BaseException:
        # Close each fd independently so a double-close / already-
        # closed error on one doesn't skip the other. We swallow
        # ``OSError`` on each because the only way a close can fail
        # here is EBADF (fd already invalid), and there is nothing
        # useful for us to do about that — the goal is simply to
        # avoid leaking still-valid fds. The original exception is
        # re-raised unchanged so the caller sees the real cause.
        try:
            os.close(master_fd)
        except OSError:
            pass
        try:
            os.close(slave_fd)
        except OSError:
            pass
        raise
    os.close(slave_fd)

    reader = _StreamReader(OutputFormatter(tag), event_sink=event_sink, tag=tag)

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
        sink_errors=reader.sink_errors,
        sink_first_error=reader.sink_first_error,
    )


class _SinkGuard:
    """Stateful wrapper that forwards to an :data:`EventSink` and keeps a
    count of failures without ever raising.

    Both the PTY-level ``_StreamReader._emit`` and the asyncio-level
    session-lifecycle events need the same contract:

    * Never let a broken sink crash the surrounding control flow.
    * Never silently drop events either — count them, remember the
      first failure, and surface a single ``warn()`` so the operator
      sees that ``events.jsonl`` / ``state.json`` are incomplete.

    We share the logic here so session-lifecycle events
    (``agent_session_started``, ``agent_resume_started``, …) and stream
    events (``assistant_*``, ``tool_call_*``, ``agent_result``) both
    benefit from the same "warn once, count the rest" behavior.
    """

    def __init__(self, sink: Optional[EventSink], tag: str) -> None:
        self._sink = sink
        self._tag = tag
        self.errors = 0
        self.first_error: Optional[str] = None
        self._warned = False

    def emit(self, event_type: str, payload: dict) -> None:
        if self._sink is None:
            return
        try:
            self._sink(event_type, payload)
        except Exception as exc:
            self.errors += 1
            if self.first_error is None:
                self.first_error = f"{type(exc).__name__}: {exc}"
            if not self._warned:
                self._warned = True
                warn(
                    f"run log sink failed on {event_type!r}: "
                    f"{self.first_error} — structured log is degraded "
                    "(further sink failures suppressed for this run)",
                    self._tag,
                )


async def run_agent(
    *,
    model: str,
    prompt: str,
    tag: str,
    workspace: str,
    agent_cmd: str,
    extra_flags: list[str],
    max_resume_attempts: int = _DEFAULT_MAX_RESUME_ATTEMPTS,
    event_sink: Optional[EventSink] = None,
) -> tuple[int, str]:
    """Launch the Cursor agent CLI and stream its output.

    If the session ends abnormally — server-side kill, signal-induced death
    (SIGKILL / SIGSEGV / …), or any non-zero exit that did not produce a
    stream-json ``result`` event — the session is automatically resumed
    via ``--continue`` up to ``max_resume_attempts`` times.  Every failure
    is logged with its exact exit code (and signal name, when applicable)
    so a broken shutdown is always visible instead of silently ignored.

    When *event_sink* is provided it receives structured events for every
    session lifecycle transition (``agent_session_started``,
    ``agent_session_finished``, ``agent_resume_started``,
    ``agent_exit_abnormal``, ``agent_resume_giveup``) as well as every
    stream event observed by :class:`_StreamReader`. This is the hook
    :class:`~iterator_loop.run_log.RunLogger` uses to produce a structured
    ``events.jsonl`` that is a superset of the PTY stream.

    Returns ``(exit_code, captured_full_text)``.  ``exit_code`` is the rc
    of the final attempt — caller-visible and easy to include in warnings.
    """
    all_text: list[str] = []
    attempt = 0
    last_rc = 0

    prompt_preview = prompt[:200]

    # One guard per ``run_agent`` invocation covers *both* the asyncio
    # lifecycle events emitted below and the PTY-side stream events
    # routed through ``_StreamReader`` (each session gets its own
    # reader with its own counters; we aggregate them into this guard
    # for a single caller-visible signal).
    lifecycle_guard = _SinkGuard(event_sink, tag)
    stream_sink_errors = 0
    stream_first_sink_error: Optional[str] = None

    while True:
        if attempt == 0:
            cmd = _build_cmd(agent_cmd, model, prompt, workspace, extra_flags)
        else:
            cmd = _build_cmd_continue(agent_cmd, model, workspace, extra_flags)

        lifecycle_guard.emit("agent_session_started", {
            "model": model,
            "tag": tag,
            "attempt": attempt,
            "max_resume_attempts": max_resume_attempts,
            "prompt_preview": prompt_preview if attempt == 0 else _RESUME_PROMPT,
            "resumed": attempt > 0,
        })

        # Pair every ``agent_session_started`` with a terminal
        # ``agent_session_finished`` event, even when ``_run_once``
        # itself raises before producing an outcome. Without this,
        # a crash inside ``_run_once`` (bad command path, PTY setup
        # failure, spawn error, …) would leave ``state.agent.
        # session_active`` stuck at ``True`` forever, contradicting
        # ``state.finished=True`` once the caller writes
        # ``run_finished``. The reviewer's reproducer was a
        # monkey-patched ``_run_once`` that raised ``RuntimeError``
        # after the first session_started — the final
        # ``events.jsonl`` showed ``[run_started,
        # agent_session_started, run_finished]`` and the snapshot
        # reported ``agent.session_active=True`` next to
        # ``finished=True``.
        #
        # On the crash path we also emit an ``agent_exit_abnormal``
        # event carrying the exception type/message so a consumer
        # tailing ``state.json`` sees both the clean ``session_active
        # =False`` flip *and* the real error context via
        # ``agent.last_error`` — matching the pattern the non-clean
        # rc path already follows.
        try:
            outcome = await _run_once(cmd, tag, event_sink=event_sink)
        except BaseException as exc:  # noqa: BLE001 - we re-raise below
            lifecycle_guard.emit("agent_session_finished", {
                "model": model,
                "tag": tag,
                "attempt": attempt,
                "rc": None,
                "rc_display": "aborted",
                "clean_exit": False,
                "saw_result": False,
                "pending_tools": 0,
                "stream_closed": False,
                "aborted": True,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            abnormal_msg = (
                f"Agent session aborted before producing an outcome: "
                f"{type(exc).__name__}: {exc}"
            )
            warn(abnormal_msg, tag)
            lifecycle_guard.emit("agent_exit_abnormal", {
                "model": model,
                "tag": tag,
                "attempt": attempt,
                "rc": None,
                "rc_display": "aborted",
                "saw_result": False,
                "pending_tools": 0,
                "stream_closed": False,
                "message": abnormal_msg,
                "error_type": type(exc).__name__,
            })
            raise
        last_rc = outcome.rc

        if outcome.text:
            all_text.append(outcome.text)

        stream_sink_errors += outcome.sink_errors
        if stream_first_sink_error is None and outcome.sink_first_error:
            stream_first_sink_error = outcome.sink_first_error

        lifecycle_guard.emit("agent_session_finished", {
            "model": model,
            "tag": tag,
            "attempt": attempt,
            "rc": outcome.rc,
            "rc_display": outcome.rc_display,
            "clean_exit": outcome.clean_exit,
            "saw_result": outcome.saw_result,
            "pending_tools": outcome.pending_tools,
            "stream_closed": outcome.stream_closed_msg,
        })

        if outcome.clean_exit:
            break

        # Always surface the real exit status up front — never silently continue.
        abnormal_msg = (
            f"Agent exited rc={outcome.rc_display} "
            f"(saw_result={outcome.saw_result}, "
            f"pending_tools={outcome.pending_tools}, "
            f"stream_closed={outcome.stream_closed_msg})"
        )
        warn(abnormal_msg, tag)
        lifecycle_guard.emit("agent_exit_abnormal", {
            "model": model,
            "tag": tag,
            "attempt": attempt,
            "rc": outcome.rc,
            "rc_display": outcome.rc_display,
            "saw_result": outcome.saw_result,
            "pending_tools": outcome.pending_tools,
            "stream_closed": outcome.stream_closed_msg,
            "message": abnormal_msg,
        })

        if outcome.should_resume and attempt < max_resume_attempts:
            attempt += 1
            warn(
                f"Session ended abnormally — auto-resuming "
                f"(attempt {attempt}/{max_resume_attempts})…",
                tag,
            )
            lifecycle_guard.emit("agent_resume_started", {
                "model": model,
                "tag": tag,
                "attempt": attempt,
                "max_resume_attempts": max_resume_attempts,
            })
            continue

        if outcome.should_resume:
            giveup_msg = (
                f"Giving up after {max_resume_attempts} resume attempt(s); "
                f"agent is still exiting abnormally (last rc={outcome.rc_display})."
            )
            warn(giveup_msg, tag)
            lifecycle_guard.emit("agent_resume_giveup", {
                "model": model,
                "tag": tag,
                "attempts": attempt,
                "max_resume_attempts": max_resume_attempts,
                "rc": outcome.rc,
                "rc_display": outcome.rc_display,
                "message": giveup_msg,
            })
        break

    # Final aggregated summary if the structured sink misbehaved at any
    # point during this ``run_agent`` invocation. The per-session /
    # per-guard warnings above already fired once each, so the total is
    # purely informational — but it gives operators a single line that
    # says "N events were lost" instead of forcing them to count
    # individual warnings scattered across the log.
    total_sink_errors = stream_sink_errors + lifecycle_guard.errors
    if total_sink_errors > 0:
        first = stream_first_sink_error or lifecycle_guard.first_error
        warn(
            f"run log sink reported {total_sink_errors} failure(s) "
            f"during this agent run (first: {first}); events.jsonl / "
            "state.json are incomplete for this session",
            tag,
        )

    return last_rc, "\n".join(all_text)
