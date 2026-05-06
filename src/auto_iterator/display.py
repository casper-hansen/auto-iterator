"""Human-readable rendering for ``ai show``.

This module owns the *presentation* layer for the operator CLI. The
underlying data — ``meta.json``, ``state.json``, ``events.jsonl``,
``logs/agent.log`` — is read by the existing modules; we just shape it
into something pleasant to look at in a terminal.

The single user-facing observation command is :func:`render_combined_view`,
which folds three sections into one screen:

1. Status — labelled key/value block built from the reconciled status,
   ``meta.json`` and ``state.json``. Stable across refreshes so the eye
   has a quiet anchor.
2. Recent events — the most recent structured events rendered through
   :func:`render_event`.
3. Agent output — the tail of ``logs/agent.log`` so operators don't
   need to know about, ``cat``, or ``tail -f`` that file directly.

In an interactive terminal, :func:`run_live_show` repaints this view on
a polling timer. Outside a TTY, callers print the same text once and
exit; ``--json`` keeps emitting raw ``state.json`` for scripts.

ANSI styling is deferred to :mod:`auto_iterator.colors` so ``NO_COLOR``
and non-TTY callers automatically get plain text.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple

from .colors import BOLD, DIM, GREEN, RED, YELLOW, CYAN, NC
from .events import tail_events
from .ls import reconcile_status
from .meta import read_meta
from .run_dir import RunPaths, read_json


# ANSI escape stripping for visible-width measurements. The status
# section pads colored values into aligned columns, and ``str.ljust``
# can't tell that ``"\x1b[32mok\x1b[0m"`` is 2 visible characters wide.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _visible_len(s: str) -> int:
    return len(_ANSI_RE.sub("", s))


def _vpad(s: str, width: int) -> str:
    """Right-pad *s* with spaces to *width* visible characters."""
    delta = width - _visible_len(s)
    if delta <= 0:
        return s
    return s + " " * delta


def _truncate_visible(s: str, max_cols: int) -> str:
    """Truncate *s* to ``max_cols`` visible columns, keeping ANSI intact.

    The live renderer measures terminal *rows* via :func:`fit_section_caps`,
    but rows alone aren't enough: a line wider than the terminal's
    columns wraps to a second physical row, silently doubling the
    section's height. That's the regression this helper exists for —
    truncating each output line to the column budget guarantees the
    rendered line consumes exactly one terminal row, so the row-based
    fit logic stays accurate even with long workspace paths or long
    agent transcript lines.

    Properties:

    * Visible width is measured with :data:`_ANSI_RE` stripped, so
      ``"\\x1b[32mok\\x1b[0m"`` counts as 2 visible columns even though
      it has ANSI escape bytes around it.
    * ANSI escape sequences are preserved verbatim in the output so
      colors are not severed mid-character. We never slice through an
      escape's bytes.
    * When truncation occurs, an ellipsis (``…``) is appended within
      the visible budget and a final :data:`NC` reset is emitted so a
      colored prefix can't bleed into the next line.
    * Already-fitting strings are returned unchanged.
    * ``max_cols <= 0`` returns the empty string (defensive default).

    The result is guaranteed to have ``_visible_len(result) <= max_cols``."""
    if max_cols <= 0:
        return ""
    if _visible_len(s) <= max_cols:
        return s
    budget = max_cols - 1  # reserve one column for the "…" indicator
    out_parts: list[str] = []
    visible = 0
    pos = 0
    for m in _ANSI_RE.finditer(s):
        chunk = s[pos:m.start()]
        for ch in chunk:
            if visible >= budget:
                return "".join(out_parts) + "…" + NC
            out_parts.append(ch)
            visible += 1
        out_parts.append(m.group(0))
        pos = m.end()
    for ch in s[pos:]:
        if visible >= budget:
            return "".join(out_parts) + "…" + NC
        out_parts.append(ch)
        visible += 1
    return "".join(out_parts)


# ── Section: status ──────────────────────────────────────────────────────────


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


def _status_section_lines(paths: RunPaths) -> list[str]:
    """Build the labelled status block as a list of lines (no trailing \\n).

    Two-column layout: short fields are paired so the section stays
    bounded to roughly 9-10 lines even with a prompt preview. That
    matters because :func:`run_live_show` has to fit status + recent
    events + agent output inside the operator's terminal height; a
    tall status block would push the lower sections off screen on a
    24-row terminal.

    Long-valued fields (timestamps, workspace) get their own line so
    they aren't truncated. The prompt preview is collapsed to a single
    line so a multi-line task description doesn't blow up the section
    height."""
    meta = read_meta(paths) or {}
    try:
        state = read_json(paths.state)
    except (FileNotFoundError, ValueError):
        state = None
    status = reconcile_status(paths, meta, state=state)
    state = state or {}

    # Paired short fields. Each tuple is (left_label, left_value,
    # right_label, right_value). Keep the list in operator-priority
    # order so the eye finds the most important info first.
    pairs: list[tuple[str, str, str, str]] = [
        ("status", _status_str(status),
         "phase", _or_dash(state.get("phase") or meta.get("status"))),
        ("outer/inner", f"{int(state.get('outer', 0) or 0)}/"
                        f"{int(state.get('inner', 0) or 0)}",
         "paused", _yes_no(state.get("paused"))),
        ("approved", _yes_no(state.get("approved")),
         "last verdict", _or_dash(state.get("last_verdict"))),
        ("total reviews", _or_dash(state.get("total_reviews")),
         "exit code", _or_dash(state.get("exit_code"))),
        ("pid", _or_dash(meta.get("pid")),
         "workspace", _or_dash(meta.get("workspace") or state.get("workspace"))),
    ]

    # Long-valued solo fields stay on their own line.
    solo: list[tuple[str, str]] = [
        ("started", _or_dash(meta.get("started_at") or state.get("started_at"))),
        ("updated", _or_dash(state.get("updated_at") or meta.get("heartbeat_at"))),
        ("finished", _or_dash(state.get("finished_at") or meta.get("finished_at"))),
    ]

    label_w_left = max(len(p[0]) for p in pairs)
    label_w_right = max(len(p[2]) for p in pairs)
    label_w_solo = max(len(label) for label, _ in solo)
    # Visible-width pad for the left value so the right column's label
    # always starts at the same offset. ANSI codes (e.g. _status_str's
    # green wrap) are zero-width so we measure with _visible_len.
    value_w_left = max(_visible_len(p[1]) for p in pairs)

    lines: list[str] = [f"{BOLD}Run {paths.run_id}{NC}"]
    for ll, lv, rl, rv in pairs:
        left = f"{DIM}{ll.ljust(label_w_left)}{NC}  {_vpad(lv, value_w_left)}"
        right = f"{DIM}{rl.ljust(label_w_right)}{NC}  {rv}"
        lines.append(f"{left}    {right}")
    for label, value in solo:
        lines.append(f"{DIM}{label.ljust(label_w_solo)}{NC}  {value}")

    preview = (state.get("prompt_preview") or "").strip()
    if preview:
        # Collapse newlines and truncate so the prompt occupies exactly
        # one line. Multi-line tasks would otherwise inflate the status
        # block past the screen budget the live renderer assumes.
        flat = preview.replace("\n", " / ").replace("\r", " ").strip()
        lines.append(f"{BOLD}prompt{NC}  {_truncate(flat, 100)}")
    return lines


def render_status_view(paths: RunPaths) -> str:
    """Compatibility wrapper used by older callers and unit tests.

    Returns the status block plus a trailing newline. The combined view
    builds its own composite output via :func:`render_combined_view`."""
    lines = _status_section_lines(paths)
    lines.append("")
    lines.append(
        f"{DIM}tip: ai show {paths.run_id} --json for raw state{NC}"
    )
    return "\n".join(lines) + "\n"


# ── Section: events ──────────────────────────────────────────────────────────


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
    """Trim ISO timestamps to ``HH:MM:SS`` for compact lines."""
    if not ts:
        return "--:--:--"
    if "T" in ts:
        time_part = ts.split("T", 1)[1]
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
    drop into raw JSON to see what just happened."""
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


def _events_section_lines(paths: RunPaths, *, lines: int) -> list[str]:
    """Build the recent-events section. Returns an empty placeholder
    section when nothing has been written yet so the section header
    stays in place across refreshes.

    The bold header alone separates this from the status section above
    — we deliberately drop a separator rule so the live renderer's
    fixed overhead stays small enough that all three sections fit on a
    24-row terminal."""
    out: list[str] = [
        f"{BOLD}Recent events{NC}  {DIM}(last {lines}){NC}",
    ]
    events = tail_events(paths.events, n=max(1, lines))
    if not events:
        out.append(f"{DIM}(no events yet){NC}")
        return out
    for evt in events:
        out.append(render_event(evt))
    return out


# ── Section: agent output ────────────────────────────────────────────────────


def tail_text_file_with_offset(
    path: Path,
    *,
    lines: int = 30,
    chunk_per_line: int = 4096,
) -> Tuple[list[str], int]:
    """Atomic version of :func:`tail_text_file` that also returns the EOF offset.

    Returns a pair ``(tail_lines, end_offset)`` where ``end_offset`` is
    the byte position in *path* immediately past the bytes consulted
    to produce ``tail_lines``. A :class:`LogTailer` parked at that
    offset (via :meth:`LogTailer.seek_to`) will surface only future
    appends — no overlap, no missed bytes.

    This matters for ``ai show --stream``: a naive implementation that
    reads the seed via :func:`tail_text_file` and *then* calls
    :meth:`LogTailer.seek_to_end` introduces a race window. Anything
    written to ``agent.log`` between the seed read and the second
    ``stat()`` would land below the tailer's offset and never make it
    to the operator's terminal — exactly the bytes you most want to
    see when starting a follow. Doing both reads inside one ``open``
    closes that window: the tailer starts where the seed stopped.

    Returns ``([], 0)`` for missing or empty files."""
    try:
        with path.open("rb") as f:
            f.seek(0, 2)  # SEEK_END
            size = f.tell()
            if size == 0:
                return [], 0
            chunk = max(1, lines) * max(64, chunk_per_line)
            seeked_inside = size > chunk
            if seeked_inside:
                f.seek(size - chunk)
            else:
                f.seek(0)
            data = f.read()
            end_offset = f.tell()
    except OSError:
        return [], 0
    if seeked_inside:
        # The seek likely landed mid-line, so drop everything up to and
        # including the first newline in our window. If the window
        # contains *no* newline at all (e.g. the log is one giant line
        # that hasn't been flushed with a \n yet, or a single line
        # longer than ``chunk``), keep the bytes we have rather than
        # discarding them — the docstring promises a truncated tail in
        # that case, not an empty result. The dropped bytes were still
        # *read*, so ``end_offset`` correctly reflects the file
        # position past them and the tailer won't re-emit them.
        nl = data.find(b"\n")
        if nl != -1:
            data = data[nl + 1 :]
    text = data.decode("utf-8", errors="replace")
    out = text.splitlines()
    if not out:
        return [], end_offset
    return out[-max(1, lines) :], end_offset


def tail_text_file(
    path: Path,
    *,
    lines: int = 30,
    chunk_per_line: int = 4096,
) -> list[str]:
    """Return the last *lines* of *path* without reading the whole file.

    For large agent transcripts (which can grow to many MiB over a long
    run) we only sip the tail. We over-read by ``lines * chunk_per_line``
    bytes which is enough for any reasonable line length; if a single
    line is longer than that we still get the *content* of the last
    line, just possibly truncated at the front. That tradeoff is worth
    keeping the reader cheap and bounded.

    Returns ``[]`` for a missing or empty file so callers can render a
    friendly placeholder.

    Callers that also need to start tailing from where this read
    stopped should use :func:`tail_text_file_with_offset` directly so
    the seed read and the follow handoff share a single file handle
    (closing the race window between ``stat()`` and the tailer's
    offset). This wrapper is kept for the many existing callers that
    only care about the lines."""
    out, _ = tail_text_file_with_offset(
        path, lines=lines, chunk_per_line=chunk_per_line,
    )
    return out


def _agent_output_section_lines(paths: RunPaths, *, lines: int) -> list[str]:
    """Build the agent-output section.

    The label says "agent output" rather than "agent.log" so users stop
    treating the file path as the workflow. Like the events section,
    we keep the per-section overhead to a single header line so the
    live combined view fits in a 24-row terminal."""
    out: list[str] = [
        f"{BOLD}Agent output{NC}  {DIM}(last {lines}){NC}",
    ]
    if not paths.agent_log.exists():
        out.append(f"{DIM}(agent has not produced output yet){NC}")
        return out
    tail = tail_text_file(paths.agent_log, lines=lines)
    if not tail:
        out.append(f"{DIM}(agent output is empty){NC}")
        return out
    for raw in tail:
        # Render lines verbatim; stripping trailing whitespace is fine
        # but never strip leading whitespace (indentation can matter
        # in tracebacks, diffs, etc.).
        out.append(raw.rstrip("\r"))
    return out


# ── Combined view + live runner ──────────────────────────────────────────────


def render_combined_view(
    paths: RunPaths,
    *,
    event_lines: int = 12,
    log_lines: int = 30,
    cols: Optional[int] = None,
) -> str:
    """Render the full ``ai show`` view as a single string.

    Three sections separated by a blank line: status, recent events,
    agent output. Always returns a string with a trailing newline so
    callers can ``sys.stdout.write`` it directly.

    When *cols* is provided each rendered line is truncated to that
    many visible columns via :func:`_truncate_visible` so the live
    renderer's row-budgeted layout can't be defeated by long workspace
    paths or unbroken agent-output lines wrapping into multiple
    physical rows. Tabs are expanded before measurement so each visible
    character corresponds to exactly one column on the terminal.

    The one-shot/non-TTY callers leave *cols* as ``None`` so piped
    output remains lossless — wrapping doesn't matter when the
    consumer is a file or another program."""
    blocks: list[str] = []
    blocks.append("\n".join(_status_section_lines(paths)))
    blocks.append("\n".join(_events_section_lines(paths, lines=event_lines)))
    blocks.append("\n".join(_agent_output_section_lines(paths, lines=log_lines)))
    text = "\n\n".join(blocks)
    if cols is not None and cols > 0:
        text = "\n".join(
            _truncate_visible(line.expandtabs(8), cols)
            for line in text.split("\n")
        )
    return text + "\n"


# Number of "fixed" rows the combined view consumes besides the
# status section's lines and the per-section content rows. Made up of:
#
#   2 blank lines between the three sections (rendered by ``\n\n.join``)
#   1 line for the recent-events section header
#   1 line for the agent-output section header
#   2 lines for the live-view footer (one blank + one footer line)
#
# Kept as a module constant so :func:`fit_section_caps` and any
# future renderer agree on the geometry without re-deriving it.
_LIVE_VIEW_OVERHEAD = 6


def fit_section_caps(
    paths: RunPaths,
    *,
    rows: int,
    requested_event_lines: int,
    requested_log_lines: int,
) -> Tuple[int, int]:
    """Pick ``(event_lines, log_lines)`` so the live view fits in *rows*.

    The status section is always rendered in full — it's the stable
    anchor at the top of the screen. The remaining vertical budget,
    after accounting for blank gaps, section headers and the footer,
    is split between recent events and agent output:

    * About a third goes to recent events so they don't drown in agent
      transcript, capped at ``requested_event_lines``.
    * The rest goes to agent output, capped at ``requested_log_lines``.
    * Each section gets at least one content line so its header never
      sits alone over an empty body.

    On absurdly tight terminals (or if the status section by itself
    already overflows) we still return ``(1, 1)`` rather than zero —
    the live renderer knows what to do, and a partially-truncated view
    is better than a missing one."""
    status_h = len(_status_section_lines(paths))
    available = max(0, int(rows) - status_h - _LIVE_VIEW_OVERHEAD)
    req_e = max(1, int(requested_event_lines))
    req_l = max(1, int(requested_log_lines))
    if available <= 2:
        return (1, 1)
    # Roughly 1/3 for events, the rest for agent output. The min-1
    # floor keeps each section visually present.
    events = min(req_e, max(1, available // 3))
    logs = min(req_l, max(1, available - events))
    # If the user requested fewer events than budget allows, hand the
    # leftover to agent output (still capped at the user's request).
    leftover = available - events - logs
    if leftover > 0 and logs < req_l:
        logs = min(req_l, logs + leftover)
    return (events, logs)


# ANSI control sequences for the live renderer. Kept as module
# constants so they're easy to grep for and easy to no-op in tests
# (the live runner accepts a custom ``out`` so capsys-style harnesses
# never see the raw escape bytes).
_ALT_SCREEN_ON = "\033[?1049h"
_ALT_SCREEN_OFF = "\033[?1049l"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_CLEAR_HOME = "\033[H\033[2J"


def _default_get_size() -> Tuple[int, int]:
    """Return ``(columns, rows)`` for the controlling terminal.

    Wraps :func:`shutil.get_terminal_size` so tests can monkeypatch
    a deterministic size without poking environment variables. The
    fallback ``(80, 24)`` matches the canonical small-terminal we
    optimise the layout for."""
    sz = shutil.get_terminal_size((80, 24))
    return (sz.columns, sz.lines)


def run_live_show(
    paths: RunPaths,
    *,
    event_lines: int = 12,
    log_lines: int = 30,
    refresh_seconds: float = 0.4,
    out=None,
    sleep: Callable[[float], None] = time.sleep,
    should_continue: Optional[Callable[[int], bool]] = None,
    use_alt_screen: bool = True,
    get_size: Optional[Callable[[], Tuple[int, int]]] = None,
) -> int:
    """Repaint the combined view on a polling timer until interrupted.

    The renderer is deliberately simple and stdlib-only:

    * Switches to the alternate screen buffer (so the operator's
      scrollback is not polluted) and hides the cursor.
    * Each tick: home cursor, clear screen, redraw the combined view
      sized to the terminal so status + events + agent output all
      stay on screen, and print a single-line footer.
    * Polls ``state.json`` / ``events.jsonl`` / ``logs/agent.log`` —
      no inotify, no daemon, no contention with the writer.
    * Exits cleanly on ``KeyboardInterrupt`` (Ctrl-C) and always
      restores cursor + alternate-screen state before returning.

    Each iteration re-measures the terminal via *get_size* (defaults to
    :func:`shutil.get_terminal_size`) so a window resize takes effect
    on the next tick. The requested ``event_lines`` / ``log_lines``
    are treated as upper bounds: :func:`fit_section_caps` shrinks them
    to fit available rows, but never grows them past the user's
    request.

    The ``should_continue`` / ``sleep`` / ``get_size`` hooks are for
    tests: they let us drive the loop deterministically without
    touching a real terminal or wall clock."""
    if out is None:
        out = sys.stdout
    if get_size is None:
        get_size = _default_get_size

    if use_alt_screen:
        out.write(_ALT_SCREEN_ON)
    out.write(_HIDE_CURSOR)
    try:
        out.flush()
    except (AttributeError, OSError):
        pass

    interrupted = False
    iteration = 0
    try:
        while True:
            iteration += 1
            try:
                cols, rows = get_size()
            except OSError:
                # Terminal size lookup can fail mid-run on weird ttys;
                # fall back to an 80x24 layout rather than blowing up.
                cols, rows = 80, 24
            cols = max(1, int(cols))
            fitted_events, fitted_logs = fit_section_caps(
                paths,
                rows=int(rows),
                requested_event_lines=event_lines,
                requested_log_lines=log_lines,
            )
            # Pass ``cols`` so each rendered line is truncated to the
            # terminal width — without this a single long agent-output
            # line or a long workspace path would wrap to a second
            # physical row and break the row budget that
            # :func:`fit_section_caps` worked out, scrolling status or
            # events off screen.
            view = render_combined_view(
                paths,
                event_lines=fitted_events,
                log_lines=fitted_logs,
                cols=cols,
            )
            footer_text = (
                f"{DIM}[ Ctrl-C to exit · refreshing every "
                f"{refresh_seconds:.2f}s ]{NC}"
            )
            # Footer geometry: one blank separator row + one footer
            # row = 2 visible rows, exactly the budget reserved by
            # ``_LIVE_VIEW_OVERHEAD``. We deliberately do NOT emit a
            # trailing newline after the footer text. With the perfect
            # 24-row fit, the footer lands on the bottom row; a final
            # ``\n`` would advance the cursor off the screen, which
            # most terminals turn into a scroll — silently knocking
            # the top status row out of view tick after tick. Leaving
            # the cursor parked at the end of the footer row is safe:
            # the next iteration's ``_CLEAR_HOME`` repositions it to
            # (1,1) before the next redraw begins.
            footer = "\n" + _truncate_visible(footer_text, cols)
            out.write(_CLEAR_HOME)
            out.write(view)
            out.write(footer)
            try:
                out.flush()
            except (AttributeError, OSError):
                pass

            if should_continue is not None and not should_continue(iteration):
                break
            try:
                sleep(max(0.05, float(refresh_seconds)))
            except KeyboardInterrupt:
                interrupted = True
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        out.write(_SHOW_CURSOR)
        if use_alt_screen:
            out.write(_ALT_SCREEN_OFF)
        try:
            out.flush()
        except (AttributeError, OSError):
            pass

    # After leaving the alternate screen the user expects to see *some*
    # final state on the regular screen so the run isn't invisible
    # post-exit. Print one final snapshot — the combined view, plus an
    # explicit "exited" hint when the user pressed Ctrl-C — so the
    # transcript records what was on screen. The post-exit snapshot
    # uses the user's requested caps (not the fitted ones), since it
    # ends up in regular scrollback where vertical room isn't bounded.
    if use_alt_screen:
        out.write(render_combined_view(
            paths,
            event_lines=max(1, event_lines),
            log_lines=max(1, log_lines),
        ))
        if interrupted:
            out.write(f"{DIM}(exited live view){NC}\n")
        try:
            out.flush()
        except (AttributeError, OSError):
            pass
    return 0


# ── Streaming tail (native-scrollback friendly) ─────────────────────────────


def stream_log(
    paths: RunPaths,
    *,
    log_lines: int = 30,
    poll_seconds: float = 0.4,
    out=None,
    sleep: Callable[[float], None] = time.sleep,
    should_continue: Optional[Callable[[int], bool]] = None,
) -> int:
    """Tail the agent transcript to stdout without owning the screen.

    The pyratatui detail TUI redraws frames in raw mode + alternate
    screen, which means every keystroke (``j``/``k``/PageUp/PageDown,
    even mouse wheel) is consumed server-side and the resulting frame
    diff has to ride one network round-trip back to the operator. Over
    a high-latency SSH link to the auto-iterator host that loop is
    visible — scrolling feels like wading through molasses.

    The streaming mode trades the live UI for the local terminal's
    *native* scrollback. Output is plain text written to the regular
    screen buffer (no ``\\033[?1049h``, no cursor games). Mouse-wheel,
    Shift+PageUp, tmux's copy-mode, ``less`` piping, etc. all then
    work entirely client-side at zero latency: bytes have already been
    delivered to the local emulator's scrollback, and navigating that
    buffer never touches the remote host.

    Layout (top → bottom, all printed once at start):

    1. **Status header** — one snapshot of the labelled status block
       so the operator has context for the run they're tailing.
    2. **Seed tail** — last ``log_lines`` lines of ``logs/agent.log``
       so recent context shows up immediately rather than waiting for
       the next agent write.

    After the seed, the function enters a poll loop that surfaces new
    appends via :class:`LogTailer` and writes them straight to *out*
    as they arrive. Lines are flushed eagerly so a pipe into ``less``
    or ``grep`` sees output in real time rather than block-buffered.

    Exits cleanly on ``KeyboardInterrupt`` (Ctrl-C). Returns ``0``.

    The ``should_continue`` / ``sleep`` hooks are test affordances
    matching :func:`run_live_show`: they let unit tests drive the
    loop deterministically without a real wall clock or terminal.
    """
    if out is None:
        out = sys.stdout

    def _flush() -> None:
        try:
            out.flush()
        except (AttributeError, OSError):
            pass

    for line in _status_section_lines(paths):
        out.write(line + "\n")
    out.write("\n")
    out.write(
        f"{BOLD}Agent output{NC}  "
        f"{DIM}(streaming · Ctrl-C to exit · "
        f"native terminal scrollback){NC}\n"
    )

    # Read the seed and capture the EOF offset in a single ``open`` so
    # the tailer can be anchored exactly past the last seed byte. If we
    # instead read the seed and then ``stat()``-ed the file again,
    # bytes appended between the two operations would be neither in
    # the seed nor surfaced by the tailer — exactly the lines an
    # operator opening ``--stream`` would most want to see. See
    # :func:`tail_text_file_with_offset` for the race analysis.
    seed_end_offset = 0
    if paths.agent_log.exists():
        seed, seed_end_offset = tail_text_file_with_offset(
            paths.agent_log, lines=max(1, int(log_lines)),
        )
        if not seed:
            out.write(f"{DIM}(agent has not produced output yet){NC}\n")
        else:
            for raw in seed:
                out.write(raw.rstrip("\r") + "\n")
    else:
        out.write(f"{DIM}(agent has not produced output yet){NC}\n")
    _flush()

    tailer = LogTailer(paths.agent_log)
    tailer.seek_to(seed_end_offset)

    iteration = 0
    try:
        while True:
            iteration += 1
            new_lines = tailer.read_new_lines()
            if new_lines:
                for raw in new_lines:
                    out.write(raw.rstrip("\r") + "\n")
                _flush()
            if should_continue is not None and not should_continue(iteration):
                break
            try:
                sleep(max(0.05, float(poll_seconds)))
            except KeyboardInterrupt:
                break
    except KeyboardInterrupt:
        pass
    return 0


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


# ── LogTailer (incremental file reader for the TUI) ──────────────────────────


class LogTailer:
    """Stream new lines from a growing text file by remembering the offset.

    The pyratatui agent-output panel polls this on every refresh
    (~0.4 s) so it must:

    * Read **only the new bytes** since the last call. We track
      ``self._offset`` and seek there on every read; an ``stat()`` is
      cheap enough that polling at 5 Hz for a single run is invisible
      even in ``htop``.
    * Survive **truncation / rotation**. If ``st_size`` shrinks below
      the cached offset, we reset to 0 so the widget rebuilds from the
      new beginning rather than seeking past EOF.
    * Buffer **partial trailing lines**. Agents write line-buffered
      output, but a ``read`` between two writes can still split a line
      down the middle of a multi-byte UTF-8 codepoint. We hold the
      bytes after the last ``\\n`` in ``self._partial`` and prepend
      them on the next read so a line never appears truncated.
    * Decode with ``errors="replace"`` so a partial multi-byte read
      can never raise — the missing bytes are recovered on the next
      tick when the rest of the codepoint is on disk.

    The class is deliberately stdlib-only: it can be unit-tested
    without spinning up the TUI, and other callers (a future ``ai
    show --follow``, e.g.) can re-use the same primitive.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._offset = 0
        self._partial = b""

    def reset(self) -> None:
        """Forget the cached offset and partial line.

        Used both manually (re-open the file from scratch) and
        automatically by :meth:`read_new_lines` when truncation is
        detected. Tests rely on this being idempotent."""
        self._offset = 0
        self._partial = b""

    def seek_to_end(self) -> int:
        """Jump the cached offset directly to the file's current EOF.

        The TUI seeds the agent-output panel with a small bounded tail
        (the last N lines) and then wants subsequent ``read_new_lines``
        calls to surface only future appends. Calling
        :meth:`read_new_lines` once to "burn" the existing bytes is
        unsafe for large logs because that method caps each read at a
        few MiB; on a 50 MiB log the second tick would surface the
        next chunk of *historical* bytes, not the new ones. This
        primitive sidesteps that by ``stat``-ing the file and parking
        the offset at EOF without reading anything.

        Also clears the partial-line buffer because the bytes we'd
        otherwise hold belong to the historical content that the
        seed was responsible for rendering, not to a future write.

        Returns the new offset (== ``st_size`` if the file exists, ``0``
        otherwise) so callers can assert in tests.

        Note: there is a small race window between the seed read in
        the TUI panel and this ``stat()`` — anything written in
        between is silently dropped. The pyratatui panel polls at
        ~5 Hz so a missed line is recovered visually on the next
        tick (the seed is re-rendered on every refresh anyway), but
        callers like :func:`stream_log` that *only* render once
        should prefer :meth:`seek_to` with the offset returned by
        :func:`tail_text_file_with_offset`."""
        try:
            size = self.path.stat().st_size
        except OSError:
            self._offset = 0
            self._partial = b""
            return 0
        self._offset = size
        self._partial = b""
        return size

    def seek_to(self, offset: int) -> int:
        """Park the cached offset at exactly *offset* bytes into the file.

        Unlike :meth:`seek_to_end`, this does not consult ``stat()`` —
        the caller is asserting "I know precisely how many bytes of
        this file I have already consumed; surface anything past that
        on the next read." Used by the streaming tail
        (:func:`stream_log`) which gets *both* the seed lines and the
        EOF offset from one call to :func:`tail_text_file_with_offset`,
        so there is no window where appends can slip through unseen.

        Negative offsets are clamped to zero (treated as "rewind to the
        start") so a buggy caller can't make the tailer seek past EOF
        on the next ``read_new_lines``. Also clears the partial-line
        buffer for the same reason :meth:`seek_to_end` does."""
        self._offset = max(0, int(offset))
        self._partial = b""
        return self._offset

    @property
    def offset(self) -> int:
        """Current byte offset into the file. Exposed for tests + the TUI."""
        return self._offset

    def read_new_lines(self) -> list[str]:
        """Read everything appended since the last call, return complete lines.

        Returns:
            A list of complete lines (no trailing ``\\n``). Bytes
            after the last newline are buffered for the next call.
            Returns ``[]`` if the file doesn't exist, is empty, or
            hasn't grown since the last call.

        The method **does not** raise on missing or unreadable files
        — the run-dir layout makes ``logs/agent.log`` lazy (the
        runner only opens it when it has something to write), and we
        don't want the polling loop to see exceptions during the
        startup window."""
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size == 0:
            # File was truncated to empty since the last read.
            self._offset = 0
            self._partial = b""
            return []
        if size < self._offset:
            # Rotation / explicit truncation: start over.
            self.reset()
        if size == self._offset:
            return []
        try:
            with self.path.open("rb") as f:
                f.seek(self._offset)
                # Bound the per-tick read to a few MiB so a runaway
                # log can't make a single tick allocate a giant
                # buffer. The bound is intentionally generous —
                # 4 MiB easily covers the per-tick burst of a
                # chatty agent — but caps the worst case.
                chunk = f.read(min(size - self._offset, 4 * 1024 * 1024))
        except OSError:
            return []
        self._offset += len(chunk)
        if not chunk:
            return []
        data = self._partial + chunk
        # Hold back any bytes after the last newline for the next call.
        nl_idx = data.rfind(b"\n")
        if nl_idx == -1:
            self._partial = data
            return []
        complete = data[: nl_idx + 1]
        self._partial = data[nl_idx + 1 :]
        text = complete.decode("utf-8", errors="replace")
        # ``splitlines`` collapses ``\r\n`` and trailing ``\n`` so we
        # don't have to strip per-line; behaviour matches what the
        # non-TTY ``tail_text_file`` path returns to operators.
        return text.splitlines()


__all__ = [
    "LogTailer",
    "fit_section_caps",
    "render_status_view",
    "render_event",
    "render_events",
    "render_combined_view",
    "run_live_show",
    "state_json_text",
    "stream_log",
    "tail_text_file",
    "tail_text_file_with_offset",
]
