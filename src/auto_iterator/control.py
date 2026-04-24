"""Control-file drain for the review-loop runner.

Operator intents (guidance, rewind, prompt/context replacement, pause) are
expressed as files dropped into ``<run_dir>/control/`` by the ``ai`` CLI.
This module knows how to consume them atomically at a single well-defined
boundary — currently ``inner_started`` — and translate them into typed
``Intent`` objects the runner can apply.

Why file-based
--------------
The alternative (a socket broker or long-lived supervisor) is rejected by
the design: files survive CLI crashes, SSH disconnects, and runner
restarts, and they need no serialization protocol beyond what Python's
stdlib already provides. Two concurrent ``ai send`` calls both land in
``guidance.txt`` because we append with ``O_APPEND`` (atomic for lines
below PIPE_BUF); the drain then consumes the file via a rename so any
writer that was holding an fd at the moment of rename still appends
safely into the drained snapshot.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .events import EventLog, RunState
from .run_dir import (
    CTL_CONTEXT,
    CTL_GUIDANCE,
    CTL_PAUSE,
    CTL_PROMPT,
    CTL_REWIND,
    RunPaths,
    now_iso,
)


# ── Intent types ─────────────────────────────────────────────────────────────


@dataclass
class RewindIntent:
    """Operator asked to jump the loop back to (outer, inner, phase).

    ``phase`` is one of ``review`` / ``fix`` / ``after_impl`` — i.e. the
    three boundaries the runner knows how to resume at. Validation happens
    in ``_load_rewind`` so the runner never sees a malformed one."""

    outer: int
    inner: int
    phase: str


# ── Drain entry point ────────────────────────────────────────────────────────


def drain_control(
    paths: RunPaths,
    state: RunState,
    log: EventLog,
) -> Optional[RewindIntent]:
    """Consume all control files in order and apply them to *state*.

    Called at each ``inner_started`` boundary by the runner. Each intent
    type has its own consume function; every one of them:

      1. Atomically takes the file out of ``control/`` (rename or unlink).
      2. Mutates ``state`` and/or returns a structured Intent.
      3. Emits an event via ``log.emit`` so readers see what happened.
      4. Appends the raw payload to ``control-applied.jsonl`` so the
         audit trail survives even if the event log is rotated later.

    Returns a :class:`RewindIntent` if a rewind was picked up (the runner
    applies it immediately); otherwise ``None``. Rewind is handled last so
    it supersedes any other mutations that landed in the same window — if
    an operator drops both ``prompt.txt`` and ``rewind.json``, the new
    prompt is applied *before* the rewind, which is what we want: the
    rewound flow uses the updated prompt on its next review.
    """
    # Order matters: prompt/context/guidance mutations first so the
    # post-rewind flow sees the new values; rewind goes last because the
    # runner may short-circuit the rest of the inner-loop body on its
    # return.
    _consume_prompt(paths, state, log)
    _consume_context(paths, state, log)
    _consume_guidance(paths, state, log)
    _apply_pause_state(paths, state, log)
    return _consume_rewind(paths, state, log)


# ── Individual consumers ─────────────────────────────────────────────────────


def _consume_prompt(paths: RunPaths, state: RunState, log: EventLog) -> None:
    f = paths.control_file(CTL_PROMPT)
    text = _take_text(f)
    if text is None:
        return
    state.prompt = text
    evt = log.emit("prompt_updated", preview=text[:200], length=len(text))
    log.audit({"event": "prompt_updated", "seq": evt["seq"], "text": text})


def _consume_context(paths: RunPaths, state: RunState, log: EventLog) -> None:
    f = paths.control_file(CTL_CONTEXT)
    text = _take_text(f)
    if text is None:
        return
    state.context = text
    evt = log.emit("context_updated", preview=text[:200], length=len(text))
    log.audit({"event": "context_updated", "seq": evt["seq"], "text": text})


def _consume_guidance(paths: RunPaths, state: RunState, log: EventLog) -> None:
    """Drain append-only ``guidance.txt`` into the state's queue.

    Uses rename-then-read so a concurrent ``ai send`` landing *after* the
    rename lands in a freshly-created ``guidance.txt`` that we pick up on
    the next drain tick. Writers who were mid-write through an open fd
    at the moment of rename still complete their ``O_APPEND`` into the
    renamed inode; we read whatever bytes are flushed before the unlink.
    Any bytes flushed *after* the unlink go to a deleted inode and are
    lost — acceptable because ``ai send`` holds the fd only for the
    length of one line."""
    f = paths.control_file(CTL_GUIDANCE)
    if not f.exists():
        return
    # Rename into a drain-suffix so concurrent `ai send` calls can keep
    # appending to a freshly created guidance.txt without their writes
    # disappearing into the drain window.
    draining = f.with_suffix(f.suffix + f".draining.{os.getpid()}")
    try:
        os.rename(f, draining)
    except FileNotFoundError:
        return
    try:
        data = draining.read_text(encoding="utf-8", errors="replace")
    except OSError:
        data = ""
    finally:
        try:
            draining.unlink()
        except OSError:
            pass

    if not data.strip():
        return

    entries: list[dict[str, str]] = []
    for raw_line in data.splitlines():
        line = raw_line.rstrip("\n").rstrip("\r")
        if not line.strip():
            continue
        # Expected ``<ISO8601>\t<text>`` shape — fall back gracefully if
        # an operator wrote plain text directly.
        if "\t" in line:
            ts, text = line.split("\t", 1)
        else:
            ts, text = now_iso(), line
        entries.append({"timestamp": ts, "text": text})

    if not entries:
        return

    for e in entries:
        state.guidance_queue.append(e["text"])
        evt = log.emit("guidance_received", text=e["text"], sent_at=e["timestamp"])
        log.audit({
            "event": "guidance_received",
            "seq": evt["seq"],
            "sent_at": e["timestamp"],
            "text": e["text"],
        })


def _consume_rewind(
    paths: RunPaths, state: RunState, log: EventLog,
) -> Optional[RewindIntent]:
    """Drain ``rewind.json`` via rename-then-read so a concurrent
    ``ai rewind`` landing a new payload mid-drain survives: the writer's
    ``atomic_write_json`` lands a fresh file at the original name, which
    the next drain tick picks up — instead of being clobbered by our
    post-read unlink of the old name."""
    f = paths.control_file(CTL_REWIND)
    draining = f.with_suffix(f.suffix + f".draining.{os.getpid()}")
    try:
        os.rename(f, draining)
    except FileNotFoundError:
        return None
    try:
        raw = draining.read_text(encoding="utf-8")
    except OSError as exc:
        log.emit("control_rejected", kind="rewind", reason=f"read failed: {exc}")
        try:
            draining.unlink()
        except OSError:
            pass
        return None
    try:
        draining.unlink()
    except OSError:
        pass

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        evt = log.emit("control_rejected", kind="rewind", reason=f"bad json: {exc}")
        log.audit({"event": "control_rejected", "seq": evt["seq"], "payload": raw})
        return None

    intent = _validate_rewind(payload)
    if isinstance(intent, str):
        evt = log.emit("control_rejected", kind="rewind", reason=intent, payload=payload)
        log.audit({"event": "control_rejected", "seq": evt["seq"], "payload": payload})
        return None

    evt = log.emit("rewind_applied", to={
        "outer": intent.outer, "inner": intent.inner, "phase": intent.phase,
    })
    log.audit({
        "event": "rewind_applied",
        "seq": evt["seq"],
        "to": {"outer": intent.outer, "inner": intent.inner, "phase": intent.phase},
    })
    return intent


def _apply_pause_state(paths: RunPaths, state: RunState, log: EventLog) -> None:
    """Observe the presence/absence of ``pause`` and drive ``paused`` edges.

    The runner's wait-loop does the actual stalling; this function only
    emits ``paused`` / ``resumed`` events on level transitions. The pause
    file itself is never unlinked here — the operator owns its lifecycle
    via ``ai resume`` so we don't race with them."""
    paused_now = paths.control_file(CTL_PAUSE).exists()
    if paused_now and not state.paused:
        state.paused = True
        log.emit("paused")
    elif not paused_now and state.paused:
        state.paused = False
        log.emit("resumed")


def wait_while_paused(paths: RunPaths, state: RunState, log: EventLog,
                      poll_interval: float = 0.5) -> None:
    """Block until ``control/pause`` disappears. Safe to call at boundaries.

    We check the pause file directly (not just ``state.paused``) so a
    file-drop-then-rm in the same boundary window doesn't cause us to
    stall forever on stale state. Emits ``paused`` / ``resumed`` exactly
    once per edge even across repeated checks."""
    import time

    pause_file = paths.control_file(CTL_PAUSE)
    if not pause_file.exists():
        if state.paused:
            state.paused = False
            log.emit("resumed")
        return
    if not state.paused:
        state.paused = True
        log.emit("paused")
        log.refresh_snapshot()
    while pause_file.exists():
        time.sleep(poll_interval)
    state.paused = False
    log.emit("resumed")
    log.refresh_snapshot()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _take_text(path: Path) -> Optional[str]:
    """Rename *path* out of the way, read it, and unlink the drained copy.

    Rename-then-read matches ``_consume_guidance``: a concurrent
    ``atomic_write_text`` that lands a new payload at *path* during the
    drain window creates a fresh file at the original name, so the next
    drain picks it up. A naive read-then-unlink would instead unlink the
    new writer's file along with the old one, silently dropping the
    intent."""
    draining = path.with_suffix(path.suffix + f".draining.{os.getpid()}")
    try:
        os.rename(path, draining)
    except FileNotFoundError:
        return None
    try:
        return draining.read_text(encoding="utf-8")
    except OSError:
        return None
    finally:
        try:
            draining.unlink()
        except OSError:
            pass


def _validate_rewind(payload: object):
    """Return a RewindIntent or a short string explaining the rejection."""
    if not isinstance(payload, dict):
        return "rewind payload must be an object"
    try:
        outer = int(payload.get("outer"))
        inner = int(payload.get("inner"))
    except (TypeError, ValueError):
        return "outer / inner must be integers"
    phase = payload.get("phase") or "review"
    if phase not in ("review", "fix", "after_impl"):
        return f"invalid phase '{phase}'"
    if phase == "after_impl":
        # after_impl always resets counters — the payload's outer/inner
        # values are purely informational at that point.
        outer, inner = 0, 0
    else:
        if outer < 1:
            return "outer must be >= 1"
        if inner < 1:
            return "inner must be >= 1"
    return RewindIntent(outer=outer, inner=inner, phase=phase)


def parse_rewind_to(expr: str) -> RewindIntent:
    """Parse the ``ai rewind --to outer=N,inner=M[,phase=...]`` shorthand.

    Accepts comma-separated ``key=value`` pairs; missing keys get sensible
    defaults (``phase=review``). Raises ``ValueError`` so the CLI can
    surface a clean exit code without traceback noise."""
    parts = [p.strip() for p in expr.split(",") if p.strip()]
    kv: dict[str, str] = {}
    for p in parts:
        if "=" not in p:
            raise ValueError(f"expected key=value, got '{p}'")
        k, v = p.split("=", 1)
        kv[k.strip()] = v.strip()
    phase = kv.get("phase", "review")
    if phase not in ("review", "fix", "after_impl"):
        raise ValueError(f"invalid phase '{phase}'")
    if phase == "after_impl":
        return RewindIntent(outer=0, inner=0, phase="after_impl")
    if "outer" not in kv or "inner" not in kv:
        raise ValueError("outer and inner are required unless phase=after_impl")
    try:
        outer = int(kv["outer"])
        inner = int(kv["inner"])
    except ValueError as exc:
        raise ValueError(f"outer/inner must be integers: {exc}") from None
    if outer < 1 or inner < 1:
        raise ValueError("outer and inner must be >= 1")
    return RewindIntent(outer=outer, inner=inner, phase=phase)
