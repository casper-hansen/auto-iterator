"""Human-readable rendering for ``ai show`` and ``ai tail``.

This module owns the *presentation* layer for the operator CLI. The
underlying data — ``meta.json``, ``state.json``, ``events.jsonl``,
``logs/agent.log`` — is read by the existing modules; we just shape it
into something pleasant to look at in a terminal.

Two responsibilities live here:

* :func:`render_status_view` — a labelled key/value block built from the
  reconciled status, ``meta.json`` and ``state.json``. This is what the
  default ``ai show RUN_ID`` prints.
* :func:`render_event` — a one-line summary of a single event, used by
  the default ``ai tail RUN_ID``. JSON / raw output (for scripting) is
  still available via ``ai tail --raw``.

ANSI styling is deferred to :mod:`auto_iterator.colors` so ``NO_COLOR``
and non-TTY callers automatically get plain text.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from .colors import BOLD, DIM, GREEN, RED, YELLOW, CYAN, NC
from .ls import reconcile_status
from .meta import read_meta
from .run_dir import RunPaths, read_json


# ── ``ai show`` — labelled status view ────────────────────────────────────────


# Color hints for the reconciled status column. Keep this map small so
# new statuses default to plain text rather than guessing a color.
_STATUS_COLORS = {
    "running": GREEN,
    "approved": GREEN,
    "exited": CYAN,
    "stuck": YELLOW,
    "unapproved": YELLOW,
    "killed": RED,
    "crashed": RED,
}


def _status_str(status: str) -> str:
    color = _STATUS_COLORS.get(status, "")
    if color:
        return f"{color}{status}{NC}"
    return status


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _or_dash(value: Any) -> str:
    """Render *value* as a string, falling back to ``—`` for None/empty."""
    if value is None:
        return "—"
    text = str(value)
    if not text.strip():
        return "—"
    return text


def render_status_view(paths: RunPaths) -> str:
    """Build the human-readable status block printed by ``ai show``.

    Reads meta + state + reconciled status (the same triple ``ai ls``
    uses) and renders an aligned key/value list plus a prompt preview.
    Always returns a string with a trailing newline so callers can
    ``sys.stdout.write`` it without fiddling with line endings."""
    meta = read_meta(paths) or {}
    try:
        state = read_json(paths.state)
    except (FileNotFoundError, ValueError):
        state = None
    status = reconcile_status(paths, meta, state=state)
    state = state or {}

    fields: list[tuple[str, str]] = [
        ("status", _status_str(status)),
        ("phase", _or_dash(state.get("phase") or meta.get("status"))),
        ("outer/inner", f"{int(state.get('outer', 0) or 0)}/"
                       f"{int(state.get('inner', 0) or 0)}"),
        ("paused", _yes_no(state.get("paused"))),
        ("approved", _yes_no(state.get("approved"))),
        ("last verdict", _or_dash(state.get("last_verdict"))),
        ("total reviews", _or_dash(state.get("total_reviews"))),
        ("exit code", _or_dash(state.get("exit_code"))),
        ("pid", _or_dash(meta.get("pid"))),
        ("workspace", _or_dash(meta.get("workspace") or state.get("workspace"))),
        ("started", _or_dash(meta.get("started_at") or state.get("started_at"))),
        ("updated", _or_dash(state.get("updated_at") or meta.get("heartbeat_at"))),
        ("finished", _or_dash(state.get("finished_at") or meta.get("finished_at"))),
    ]

    label_w = max(len(label) for label, _ in fields)
    out_lines: list[str] = []
    title = f"{BOLD}Run {paths.run_id}{NC}"
    out_lines.append(title)
    out_lines.append(f"{DIM}{'─' * 60}{NC}")
    for label, value in fields:
        out_lines.append(f"{DIM}{label.ljust(label_w)}{NC}  {value}")

    preview = (state.get("prompt_preview") or "").strip()
    if preview:
        out_lines.append("")
        out_lines.append(f"{BOLD}prompt{NC}")
        for raw in preview.splitlines():
            out_lines.append(f"  {raw}")

    out_lines.append("")
    out_lines.append(
        f"{DIM}tip: ai show {paths.run_id} --json for raw state · "
        f"--logs for agent log{NC}"
    )
    return "\n".join(out_lines) + "\n"


# ── ``ai tail`` — compact event lines ─────────────────────────────────────────


# Per-event-type color hints. Match what operators expect to scan for
# (problems in red, completions in green, neutral progress dim).
_EVENT_COLORS = {
    "run_started": CYAN,
    "run_finished": BOLD,
    "outer_started": CYAN,
    "inner_started": CYAN,
    "review_started": "",
    "review_finished": "",
    "fix_started": "",
    "fix_finished": "",
    "impl_started": "",
    "impl_finished": "",
    "guidance_received": YELLOW,
    "rewind_applied": YELLOW,
    "rewind_narrowed": YELLOW,
    "prompt_updated": YELLOW,
    "paused": YELLOW,
    "resumed": GREEN,
    "control_rejected": RED,
    "outer_finished": "",
}


def _short_ts(ts: str) -> str:
    """Trim ISO timestamps to ``HH:MM:SS`` for compact tail lines."""
    if not ts:
        return "--:--:--"
    # Expect ``YYYY-MM-DDTHH:MM:SS[.us][+TZ]``; pull the time-of-day.
    if "T" in ts:
        time_part = ts.split("T", 1)[1]
        # Trim sub-second + timezone tail (``12:34:56.789012+00:00``).
        for stop in (".", "+", "-", "Z"):
            if stop in time_part:
                time_part = time_part.split(stop, 1)[0]
                break
        return time_part[:8]
    return ts[:8]


def _truncate(text: str, limit: int) -> str:
    text = text.replace("\n", " ").replace("\r", " ").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def render_event(evt: dict, *, color: bool = True) -> str:
    """Render one event as a single line.

    Format: ``HH:MM:SS  #seq  type[colored]  k=v k=v   "summary"``.

    The summary captures the most useful free-text payload (guidance
    text, prompt preview, error reason) so the operator doesn't have to
    drop into ``--raw`` to see what just happened."""
    ts = _short_ts(str(evt.get("timestamp", "")))
    typ = str(evt.get("type", "?"))
    seq = evt.get("seq")
    seq_str = f"#{seq}".rjust(5) if isinstance(seq, int) else "     "

    type_styled = typ.ljust(18)
    if color and _EVENT_COLORS.get(typ):
        type_styled = f"{_EVENT_COLORS[typ]}{type_styled}{NC}"

    extras: list[str] = []
    if "outer" in evt and "inner" in evt:
        extras.append(f"o/i={evt['outer']}/{evt['inner']}")
    elif "outer" in evt:
        extras.append(f"outer={evt['outer']}")
    elif "inner" in evt:
        extras.append(f"inner={evt['inner']}")
    if evt.get("phase"):
        extras.append(f"phase={evt['phase']}")
    if "verdict" in evt:
        verdict = str(evt.get("verdict", ""))
        extras.append(f"verdict={verdict}")
    if "approved" in evt:
        extras.append(f"approved={_yes_no(evt['approved'])}")
    if "exit_code" in evt and evt["exit_code"] is not None:
        extras.append(f"exit={evt['exit_code']}")
    if evt.get("model"):
        extras.append(f"model={evt['model']}")
    if isinstance(evt.get("to"), dict):
        t = evt["to"]
        extras.append(
            f"to={t.get('outer', '')}/{t.get('inner', '')}/{t.get('phase', '')}"
        )
    if evt.get("reason"):
        extras.append(f"reason={_truncate(str(evt['reason']), 40)}")

    summary = ""
    text_payload = (
        evt.get("text")
        or evt.get("prompt_preview")
        or evt.get("message")
        or evt.get("guidance")
    )
    if text_payload:
        summary = f'  "{_truncate(str(text_payload), 80)}"'

    base = f"{DIM if color else ''}{ts}{NC if color else ''}  {seq_str}  {type_styled}"
    if extras:
        base += "  " + " ".join(extras)
    base += summary
    return base


def render_events(events: Iterable[dict], *, color: bool = True) -> str:
    """Render a sequence of events as one event-per-line text block."""
    return "\n".join(render_event(e, color=color) for e in events)


# ── ``ai show --logs`` / ``ai tail --agent-log`` ──────────────────────────────


def render_agent_log_tail(paths: RunPaths, *, lines: int = 50) -> str:
    """Return the last *lines* of ``logs/agent.log``, or a friendly placeholder.

    The agent log is the raw subprocess transcript and may be huge; we
    only ever read the tail. Missing-file is a normal condition before
    the runner has emitted anything, so we surface that explicitly
    instead of raising."""
    try:
        with paths.agent_log.open("r", encoding="utf-8", errors="replace") as fh:
            buf = fh.readlines()
    except FileNotFoundError:
        return f"(no agent log yet at {paths.agent_log})\n"
    if not buf:
        return f"(agent log is empty: {paths.agent_log})\n"
    tail = buf[-max(1, lines):]
    return "".join(tail)


# ── State JSON helpers (kept here so ``--json`` paths share a formatter) ──────


def state_json_text(paths: RunPaths) -> str:
    """Pretty-printed ``state.json`` — falls back to meta if state is missing.

    Used by ``ai show --json`` so the scripting contract stays a single
    JSON object even before the runner has produced its first snapshot."""
    try:
        text = paths.state.read_text(encoding="utf-8")
    except FileNotFoundError:
        meta = read_meta(paths) or {}
        return json.dumps(meta, indent=2) + "\n"
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = {"raw": text}
    return json.dumps(obj, indent=2) + "\n"


__all__ = [
    "render_status_view",
    "render_event",
    "render_events",
    "render_agent_log_tail",
    "state_json_text",
]
