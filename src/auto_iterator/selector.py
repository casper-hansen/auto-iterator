"""Lightweight terminal selector for picking a run from ``list_runs``.

This is a small, stdlib-only widget used by ``ai`` whenever an operator
omits ``run_id`` on a run-targeting subcommand and stdin/stdout are both
TTYs. It draws a compact list of :class:`RunRow` values, lets the user
move with arrow keys / ``j`` / ``k``, picks with Enter, and cancels with
``q`` / Esc / Ctrl-C.

The selector is deliberately:

* **Snapshot-based.** It calls :func:`list_runs` once and never refreshes
  in the background. Operators wanting a live dashboard already have
  ``ai ls`` + ``watch``.
* **Stdlib-only.** ``termios`` / ``tty`` for raw input, plain ANSI escapes
  for redraw. No curses, no third-party TUI dependency.
* **TTY-gated.** Non-TTY callers must check :func:`is_interactive` first
  and fall back to a clean error; we never try to read raw bytes from a
  pipe.

Terminal state is restored in a ``finally`` block so a Ctrl-C in the
middle of a redraw can't leave the user's shell in cbreak mode.
"""

from __future__ import annotations

import os
import select
import sys
from contextlib import contextmanager
from typing import List, Optional, Sequence

from .ls import RunRow


# ── Capability detection ──────────────────────────────────────────────────────


def _use_color() -> bool:
    """ANSI styling on iff stdout is a TTY and ``NO_COLOR`` is unset."""
    if "NO_COLOR" in os.environ:
        return False
    try:
        return sys.stdout.isatty()
    except (AttributeError, OSError):
        return False


def is_interactive() -> bool:
    """True iff both stdin and stdout are TTYs.

    The selector reads raw bytes from stdin and repaints stdout, so both
    sides must be a real terminal. Pipes, redirects, or non-interactive
    runners (CI, ``ssh -T``) all fall through to the explicit-id error
    path in the CLI."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, OSError):
        return False


# ── Raw-mode helpers ──────────────────────────────────────────────────────────


@contextmanager
def _raw_terminal(fd: int):
    """Cbreak mode for *fd*, restoring the original termios on exit.

    Cbreak (rather than full raw) keeps signals working: Ctrl-C still
    raises ``KeyboardInterrupt`` in the parent, which is desirable — the
    selector is one step in a CLI session, not a fullscreen app."""
    import termios  # POSIX only; the selector is a no-op on Windows today
    import tty

    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key(fd: int) -> str:
    """Block until one logical key press is read from *fd*.

    Returns a small vocabulary of strings (``UP``, ``DOWN``, ``ENTER``,
    ``ESC``, ``QUIT``, ``CTRL_C``, ``CTRL_D``) plus the literal char for
    everything else. Arrow keys arrive as ``ESC [ A`` — we peek with
    ``select`` so a bare Esc doesn't hang."""
    ch = os.read(fd, 1)
    if not ch:
        return "CTRL_D"
    if ch == b"\x1b":
        # Disambiguate bare Esc vs CSI sequence with a tiny timeout.
        rlist, _, _ = select.select([fd], [], [], 0.05)
        if not rlist:
            return "ESC"
        seq = os.read(fd, 2)
        if seq == b"[A":
            return "UP"
        if seq == b"[B":
            return "DOWN"
        if seq == b"[C":
            return "RIGHT"
        if seq == b"[D":
            return "LEFT"
        if seq == b"[H":
            return "HOME"
        if seq == b"[F":
            return "END"
        return "ESC"
    if ch in (b"\r", b"\n"):
        return "ENTER"
    if ch == b"\x03":
        return "CTRL_C"
    if ch == b"\x04":
        return "CTRL_D"
    if ch == b"q":
        return "QUIT"
    if ch == b"j":
        return "DOWN"
    if ch == b"k":
        return "UP"
    if ch == b"g":
        return "HOME"
    if ch == b"G":
        return "END"
    try:
        return ch.decode("utf-8")
    except UnicodeDecodeError:
        return ""


# ── Row formatting ────────────────────────────────────────────────────────────


def _short_workspace(ws: str) -> str:
    """Collapse ``$HOME/...`` to ``~/...`` so rows fit on narrow terminals."""
    if not ws:
        return ""
    home = os.path.expanduser("~")
    if home and ws.startswith(home):
        return "~" + ws[len(home):]
    return ws


def _short_updated(ts: str) -> str:
    """Trim ISO timestamps to ``YYYY-MM-DDTHH:MM:SS`` for compactness."""
    if not ts:
        return ""
    return ts[:19]


def _term_size() -> tuple[int, int]:
    try:
        sz = os.get_terminal_size()
        return max(60, sz.columns), max(10, sz.lines)
    except OSError:
        return 120, 40


# Column widths used by both the selector and the (future) header. Keep
# these in sync with ``_format_row`` below — the spacing here defines the
# visual contract.
_COL_RUN_ID = 26
_COL_STATUS = 9
_COL_PHASE = 11
_COL_OI = 6
_COL_UPDATED = 19
_COL_WORKSPACE = 22


def _format_row(row: RunRow, width: int) -> str:
    """Render one run as a single line, truncated to *width* columns."""
    workspace = _short_workspace(row.workspace)
    if len(workspace) > _COL_WORKSPACE:
        # Keep the trailing path component visible when truncating long
        # workspaces — operators recognise their projects by basename.
        base = os.path.basename(workspace.rstrip("/")) or workspace
        workspace = "…/" + base if len(base) <= _COL_WORKSPACE - 2 else base[:_COL_WORKSPACE]
    parts = [
        row.run_id.ljust(_COL_RUN_ID)[:_COL_RUN_ID],
        (row.status or "").ljust(_COL_STATUS)[:_COL_STATUS],
        (row.phase or "").ljust(_COL_PHASE)[:_COL_PHASE],
        f"{row.outer}/{row.inner}".ljust(_COL_OI)[:_COL_OI],
        _short_updated(row.updated_at).ljust(_COL_UPDATED)[:_COL_UPDATED],
        workspace.ljust(_COL_WORKSPACE)[:_COL_WORKSPACE],
        (row.prompt_preview or "").replace("\n", " ").strip(),
    ]
    line = "  ".join(parts)
    if width and len(line) > width:
        line = line[: max(0, width - 1)] + "…"
    return line


def _header_line(width: int) -> str:
    parts = [
        "RUN_ID".ljust(_COL_RUN_ID),
        "STATUS".ljust(_COL_STATUS),
        "PHASE".ljust(_COL_PHASE),
        "O/I".ljust(_COL_OI),
        "UPDATED".ljust(_COL_UPDATED),
        "WORKSPACE".ljust(_COL_WORKSPACE),
        "PROMPT",
    ]
    line = "  ".join(parts)
    if width and len(line) > width:
        line = line[:width]
    return line


# ── Selector ──────────────────────────────────────────────────────────────────


def select_run(
    rows: Sequence[RunRow],
    *,
    prompt: str = "Select a run",
) -> Optional[RunRow]:
    """Render an interactive picker and return the chosen row.

    Returns ``None`` if the user cancels (``q`` / Esc / Ctrl-C / Ctrl-D)
    or if *rows* is empty. Raises :class:`RuntimeError` if called from a
    non-TTY context — callers should gate on :func:`is_interactive`
    first."""
    if not is_interactive():
        raise RuntimeError("select_run requires an interactive terminal")
    if not rows:
        return None

    rows = list(rows)
    fd = sys.stdin.fileno()
    cursor = 0
    use_color = _use_color()
    bold = "\033[1m" if use_color else ""
    dim = "\033[2m" if use_color else ""
    inv = "\033[7m" if use_color else ""
    nc = "\033[0m" if use_color else ""

    out = sys.stdout.write
    flush = sys.stdout.flush

    drawn_lines = 0
    # Cap the visible window so very long run histories don't push the
    # prompt off-screen. We always show the cursor row.
    cols, rows_avail = _term_size()
    # 3 lines of chrome (title, header, footer) + breathing room.
    max_visible = max(5, rows_avail - 5)

    def _visible_window() -> tuple[int, int]:
        """Return (start, end) indices of the rows we'll render right now."""
        if len(rows) <= max_visible:
            return 0, len(rows)
        half = max_visible // 2
        start = max(0, cursor - half)
        end = min(len(rows), start + max_visible)
        start = max(0, end - max_visible)
        return start, end

    def render() -> None:
        nonlocal drawn_lines
        cols_now, _ = _term_size()
        if drawn_lines:
            # Move cursor up to top of the previous render and clear to
            # end-of-screen so we redraw onto a clean canvas without
            # touching the scrollback above us.
            out(f"\033[{drawn_lines}A\033[J")
        lines: list[str] = []
        title = f"{bold}{prompt}{nc}"
        hint = f"{dim}↑/↓ or j/k · Enter=select · q/Esc=cancel{nc}"
        lines.append(f"{title}  {hint}")
        lines.append(f"{dim}{_header_line(cols_now)}{nc}")
        start, end = _visible_window()
        for i in range(start, end):
            text = _format_row(rows[i], cols_now - 2)
            if i == cursor:
                lines.append(f"{inv}> {text}{nc}")
            else:
                lines.append(f"  {text}")
        if end < len(rows) or start > 0:
            lines.append(
                f"{dim}  [{cursor + 1}/{len(rows)}]{nc}"
            )
        for line in lines:
            out(line + "\n")
        drawn_lines = len(lines)
        flush()

    # Hide the hardware cursor while we own the screen — the inverse-
    # video pointer is the user-visible cursor instead.
    if use_color:
        out("\033[?25l")
        flush()

    try:
        with _raw_terminal(fd):
            render()
            while True:
                try:
                    key = _read_key(fd)
                except KeyboardInterrupt:
                    return None
                if key == "UP":
                    cursor = (cursor - 1) % len(rows)
                elif key == "DOWN":
                    cursor = (cursor + 1) % len(rows)
                elif key == "HOME":
                    cursor = 0
                elif key == "END":
                    cursor = len(rows) - 1
                elif key == "ENTER":
                    return rows[cursor]
                elif key in ("QUIT", "ESC", "CTRL_C", "CTRL_D"):
                    return None
                else:
                    continue
                render()
    finally:
        if use_color:
            out("\033[?25h")
            flush()


# ── Convenience: format rows for non-interactive previews ─────────────────────


def format_rows_plain(rows: Sequence[RunRow]) -> List[str]:
    """Render *rows* without ANSI styling. Useful for tests + previews."""
    cols, _ = _term_size()
    out = [_header_line(cols)]
    for row in rows:
        out.append(_format_row(row, cols))
    return out
