"""pyratatui TUI for ``ai`` and ``ai show``.

Two screens, one app, polled timers, **filesystem-as-protocol**:

* :class:`RunListScreen` is what ``ai`` (no subcommand) opens — a
  live-updating run table backed by :func:`auto_iterator.ls.list_runs`,
  with single-key bindings for the operator verbs (``s``end, ``p``ause,
  ``r``esume, ``k``ill, etc.). Every action writes a control file or
  hands off to :func:`auto_iterator.actions.spawn_runner_detached`;
  none of them holds a runner pid.
* :class:`RunDetailScreen` is what ``ai show <run_id>`` opens (when
  stdout is a TTY and ``--once`` / ``--logs`` / ``--json`` are not
  set). Two stacked widgets: a single-line minimal status bar
  summarizing reconciled run state (status / phase / outer-inner /
  verdict / paused) and a scrollable agent-output paragraph filling
  the rest of the screen, backed by :class:`LogTailer`.

The render layer is `pyratatui <https://github.com/pyratatui/pyratatui>`_
— Python bindings around Rust's ratatui. The previous Textual
implementation paid one Python-side widget repaint per frame and was
visibly laggy over high-latency SSH; pyratatui pushes the diff /
double-buffer / cell repaint through native code, which is what the
operator wanted ("butter smooth ... like claude code or codex").

Architecture:

* **Screens are pure-Python state machines.** Every screen class
  (``RunListScreen``, ``RunDetailScreen``) and every modal
  (``_PromptModal``, ``_ConfirmModal``, ``_BackendChoiceModal``,
  ``_DiffViewer``) holds its own data, exposes a ``handle_key(ev)``
  entry point, and has filesystem side effects routed through
  :mod:`auto_iterator.actions` so the CLI and the TUI cannot drift on
  the protocol. None of them touches a ``Terminal`` directly — that's
  the App's job.
* **The App is the draw loop.** ``RunListApp.run()`` /
  ``RunDetailApp.run()`` open a pyratatui ``Terminal``, poll input
  with a small timeout (~30 ms = ~30 fps cap), forward each event to
  the current screen, run the screen's periodic ``tick()`` work, then
  paint by calling ``term.draw(render_fn)``. Pyratatui's ratatui
  backend takes care of the cell-level diff and only ships changed
  cells to the terminal; we don't have to coalesce updates by hand
  the way the Textual layer used to with ``app.batch_update``.
* **No widget tree to query.** Tests drive screens directly:
  ``screen.handle_key(KeyEvent("s"))`` → assert the screen's modal
  stack now contains a ``_PromptModal``; ``screen.modals[-1].submit("x")``
  → assert the side-effect file was written. There's no ``Pilot``,
  no ``query_one``, no headless render — the state machine *is* the
  contract, and the renderer is a stateless function over it.

Design rules baked in (preserved verbatim from the Textual era):

* **TUI never owns runner lifecycles.** Quitting the app does not
  signal anything. The only way the TUI signals a runner pid is when
  the operator presses ``k`` (kill) — and even then we route through
  :func:`actions.signal_runner` so the same code path the CLI uses
  is exercised, no shortcut.
* **Polling, not inotify.** Each screen owns its own poll cadence
  (≈1 s for the run list, ≈``refresh_seconds`` for the detail status
  bar, ≈0.4 s for the agent log). The App's draw loop holds the
  timer; no background thread, no subscribed file watcher.
* **Lazy import.** :mod:`pyratatui` is loaded inside ``App.run`` so
  ``import auto_iterator.tui`` stays cheap — the non-TTY ``ai show``
  paths import this module only to monkeypatch ``run_detail_app`` in
  tests and never need the native binding loaded.

The TUI is *not* the source of truth for any state; every refresh
re-reads ``meta.json`` / ``state.json`` / ``events.jsonl`` /
``agent.log``. If the file goes away (worktree removed, run cleaned
up), the screens degrade gracefully — they show a placeholder rather
than crashing.
"""

from __future__ import annotations

import re
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Iterable, List, Optional, Tuple

from . import actions
from .display import LogTailer, tail_text_file
from .ls import RunRow, list_runs, reconcile_status
from .meta import read_meta
from .run_dir import RunPaths, read_json


# Upper bound on how many lines of the existing agent transcript the
# detail screen will materialise into the log buffer on open and how
# many it keeps live in the steady-state ring buffer.
#
# pyratatui's ``Paragraph`` re-wraps and re-renders every frame — the
# Rust diff engine is cheap enough that we don't pay the Textual-era
# cost of pre-baked ``Strip`` lists going stale on resize. The cap is
# kept primarily to bound memory: a multi-MiB transcript would
# otherwise pin every byte ever emitted in our deque. 10 000 lines is
# well above the 200-line floor enforced by
# ``test_pressing_enter_seeds_full_agent_log`` and any realistic
# "what happened recently" need.
_FULL_LOG_SEED_CAP = 10_000


# ── Helpers shared by both screens ──────────────────────────────────────────


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    """Drop ANSI escape codes — the existing renderers emit them, but
    pyratatui's text widgets render plain strings (styling flows
    through ``Style`` / ``Span`` instead). Without this pass the status
    panel ends up showing literal ``\\x1b[1m`` bytes."""
    return _ANSI_RE.sub("", text)


def _minimal_status_line(paths: RunPaths) -> str:
    """One-line reconciled status summary for the per-run detail view.

    The detail screen strips the verbose status block, prompt preview,
    and events panel so the agent transcript fills the screen. We
    still want a single-glance answer to "is this run alive, and how
    far has it gotten?", so this helper folds the small subset of
    fields an operator scans first into a compact bar:

    ``<run_id> · status · phase · outer/inner · verdict: … · paused: …``

    Reads ``meta.json`` and ``state.json`` directly (same source the
    full status block uses) so the bar survives a run-dir whose
    state is partially written. Missing fields collapse to ``—`` so
    the bar's shape stays stable as the run progresses."""
    meta = read_meta(paths) or {}
    try:
        state = read_json(paths.state)
    except (FileNotFoundError, ValueError):
        state = None
    status = reconcile_status(paths, meta, state=state)
    state = state or {}
    phase = state.get("phase") or meta.get("status") or "—"
    outer = int(state.get("outer", 0) or 0)
    inner = int(state.get("inner", 0) or 0)
    verdict = state.get("last_verdict") or "—"
    paused = "yes" if state.get("paused") else "no"
    return (
        f"{paths.run_id} · {status} · {phase} · "
        f"{outer}/{inner} · verdict: {verdict} · paused: {paused}"
    )


def _row_cells(row: RunRow) -> tuple[str, ...]:
    """One :class:`RunRow` rendered as the cells of a Table row.

    Mirrors the columns documented for ``ai ls`` so an operator who
    knows the CLI's output format reads the TUI table without
    re-learning anything."""
    return (
        row.run_id,
        row.status or "",
        row.phase or "",
        f"{row.outer}/{row.inner}",
        row.last_verdict or "",
        (row.updated_at or "")[:25],
        (row.prompt_preview or "").replace("\n", " ")[:60],
    )


# ── Key event normalization ─────────────────────────────────────────────────


@dataclass(frozen=True)
class KeyEvent:
    """Backend-agnostic key event.

    Mirrors the public surface of pyratatui's ``PyKeyEvent`` (``code``,
    ``ctrl``, ``alt``, ``shift``) but without holding a reference to
    the native object — tests construct these directly,
    :func:`_to_key_event` adapts the live pyratatui events at runtime.
    """

    code: str
    ctrl: bool = False
    alt: bool = False
    shift: bool = False


def _to_key_event(raw: Any) -> Optional[KeyEvent]:
    """Best-effort adapter from an arbitrary key-event-shaped object.

    Accepts:

    * ``KeyEvent`` (returns as-is).
    * pyratatui ``PyKeyEvent`` (has ``.code``/``.ctrl``/etc. attrs).
    * ``None`` (poll timed out without an event) → returns ``None``.
    * A plain string (treated as ``KeyEvent(code=string)``).

    Anything else returns ``None`` so the loop can fall through.
    """
    if raw is None:
        return None
    if isinstance(raw, KeyEvent):
        return raw
    if isinstance(raw, str):
        return KeyEvent(code=raw)
    code = getattr(raw, "code", None)
    if code is None:
        return None
    return KeyEvent(
        code=str(code),
        ctrl=bool(getattr(raw, "ctrl", False)),
        alt=bool(getattr(raw, "alt", False)),
        shift=bool(getattr(raw, "shift", False)),
    )


# ── Notification ────────────────────────────────────────────────────────────


@dataclass
class Notification:
    """In-app toast / status message.

    Mirrors what ``self.notify(...)`` produced in the Textual era — a
    short message with a severity hint. Screens append to a list; the
    renderer pulls the most recent notification and shows it in a
    one-row footer. Tests inspect the list directly to assert the
    user-facing feedback (e.g. "guidance queued for ...")."""

    text: str
    severity: str = "information"  # "information" | "warning" | "error"


# ── Modal: text prompt ──────────────────────────────────────────────────────


@dataclass
class _PromptModal:
    """Single-input modal: title + hint + editable line + submit/cancel.

    State machine, not widget. ``handle_key`` mutates ``value`` for
    printable keys, advances the cursor for navigation keys, and sets
    ``done`` once the operator submits or cancels. The owning screen
    polls ``done`` after each ``handle_key`` call, pops the modal and
    invokes ``on_submit(value)`` or ``on_cancel()`` as appropriate.

    The submit-on-Enter behaviour and Esc-to-cancel match the Textual
    implementation 1:1 so existing operator muscle memory carries over.
    """

    title: str
    hint: str = ""
    placeholder: str = ""
    value: str = ""
    cursor: int = field(default=-1)
    done: bool = False
    submitted: bool = False
    on_submit: Optional[Callable[[str], None]] = None
    on_cancel: Optional[Callable[[], None]] = None

    def __post_init__(self) -> None:
        # ``-1`` sentinel means "park cursor at end of initial value" so
        # an operator who wants to edit a pre-populated workspace path
        # can hit Backspace immediately.
        if self.cursor < 0:
            self.cursor = len(self.value)

    # ── side-effects ──

    def submit(self, value: Optional[str] = None) -> None:
        """Programmatic submit (used by tests). Equivalent to Enter
        once ``value`` has been typed in.

        Passing ``value`` overrides ``self.value`` first so callers can
        compress the "set the input field, then press Enter" flow into
        a single call without poking the dataclass directly. Never
        re-fires once the modal is already ``done``."""
        if self.done:
            return
        if value is not None:
            self.value = value
            self.cursor = len(self.value)
        self.done = True
        self.submitted = True

    def cancel(self) -> None:
        """Programmatic cancel (used by tests). Equivalent to Esc."""
        if self.done:
            return
        self.done = True
        self.submitted = False

    # ── input ──

    def handle_key(self, ev: KeyEvent) -> bool:
        """Apply *ev* to the modal. Returns ``True`` once ``done``.

        The owning screen MUST stop processing the event chain when
        this returns ``True`` — the modal is on its way out and any
        further dispatch would land on a stale state."""
        if self.done:
            return True
        code = ev.code
        if code == "Enter":
            self.submit()
            return True
        if code == "Esc" or code == "Escape":
            self.cancel()
            return True
        if code == "Backspace":
            if self.cursor > 0:
                self.value = (
                    self.value[: self.cursor - 1] + self.value[self.cursor :]
                )
                self.cursor -= 1
            return False
        if code == "Delete":
            if self.cursor < len(self.value):
                self.value = (
                    self.value[: self.cursor] + self.value[self.cursor + 1 :]
                )
            return False
        if code == "Left":
            self.cursor = max(0, self.cursor - 1)
            return False
        if code == "Right":
            self.cursor = min(len(self.value), self.cursor + 1)
            return False
        if code == "Home":
            self.cursor = 0
            return False
        if code == "End":
            self.cursor = len(self.value)
            return False
        if len(code) == 1 and code.isprintable() and not ev.ctrl and not ev.alt:
            self.value = (
                self.value[: self.cursor] + code + self.value[self.cursor :]
            )
            self.cursor += 1
            return False
        return False


# ── Modal: yes/no confirmation ──────────────────────────────────────────────


@dataclass
class _ConfirmModal:
    """Yes/no confirmation for destructive verbs (kill, restart, revert).

    Same gate the CLI's ``_maybe_confirm`` flow puts in front of an
    operator who selected the run from a list rather than typing its
    id: one extra keystroke between "selected the wrong row" and
    "killed the runner"."""

    title: str
    body: str = ""
    done: bool = False
    confirmed: bool = False
    on_result: Optional[Callable[[bool], None]] = None

    def confirm(self) -> None:
        if self.done:
            return
        self.done = True
        self.confirmed = True

    def cancel(self) -> None:
        if self.done:
            return
        self.done = True
        self.confirmed = False

    def handle_key(self, ev: KeyEvent) -> bool:
        if self.done:
            return True
        code = ev.code
        if code in ("y", "Y", "Enter"):
            self.confirm()
            return True
        if code in ("n", "N", "Esc", "Escape"):
            self.cancel()
            return True
        return False


# ── Modal: backend preset picker ────────────────────────────────────────────


@dataclass
class _BackendChoiceModal:
    """Backend / preset picker shown as the third step of "New run".

    Shipped with two recommended layouts:

    1. **Cursor — Opus impl + GPT reviewer.** Single-backend Cursor
       run; the cursor backend's defaults give the canonical Claude-
       Opus implementer / GPT-5.5 reviewer pairing.
    2. **Claude Code + Codex — mixed.** Claude Code as
       implementer/fixer, Codex as the fresh-eyes reviewer.

    Both presets pass ``ignore_env_overrides=True`` so the runner gets
    exactly the layout the operator picked: a stray
    ``AGENT_REVIEWER_BACKEND`` / ``AGENT_CMD`` in the surrounding shell
    cannot silently rewrite a "Cursor" pick into a mixed Claude/Codex
    run, or vice-versa.
    """

    PRESETS: Tuple[dict, ...] = (
        {
            "id": "cursor",
            "label": "1 · Cursor — Opus impl + GPT reviewer  (recommended)",
            "summary": (
                "All phases run through Cursor's CLI.\n"
                "  impl/fix : claude-opus-4-7-thinking-max\n"
                "  reviewer : gpt-5.5-extra-high"
            ),
            "kwargs": {
                "backend": "cursor",
                "ignore_env_overrides": True,
            },
        },
        {
            "id": "claude-codex",
            "label": "2 · Claude Code + Codex — Claude impl/fix, Codex reviewer",
            "summary": (
                "Mixed-backend fallback when Cursor isn't available.\n"
                "  impl/fix : claude (opus, claude-code CLI)\n"
                "  reviewer : codex (fresh-eyes review)"
            ),
            "kwargs": {
                "backend": "claude-code",
                "reviewer_backend": "codex",
                "ignore_env_overrides": True,
            },
        },
    )

    done: bool = False
    chosen: Optional[dict] = None
    selected_idx: int = 0
    on_result: Optional[Callable[[Optional[dict]], None]] = None

    def _kwargs_for(self, preset_id: str) -> Optional[dict]:
        for preset in self.PRESETS:
            if preset["id"] == preset_id:
                return dict(preset["kwargs"])
        return None

    def pick(self, idx: int) -> None:
        if self.done:
            return
        if 0 <= idx < len(self.PRESETS):
            self.selected_idx = idx
            self.chosen = self._kwargs_for(self.PRESETS[idx]["id"])
            self.done = True

    def cancel(self) -> None:
        if self.done:
            return
        self.done = True
        self.chosen = None

    def handle_key(self, ev: KeyEvent) -> bool:
        if self.done:
            return True
        code = ev.code
        if code == "1":
            self.pick(0)
            return True
        if code == "2":
            self.pick(1)
            return True
        if code in ("Esc", "Escape"):
            self.cancel()
            return True
        if code == "Up":
            self.selected_idx = max(0, self.selected_idx - 1)
            return False
        if code == "Down":
            self.selected_idx = min(
                len(self.PRESETS) - 1, self.selected_idx + 1,
            )
            return False
        if code == "Enter":
            self.pick(self.selected_idx)
            return True
        return False


# ── Modal: read-only diff/status viewer ─────────────────────────────────────


@dataclass
class _DiffViewer:
    """Plain-text scrollable viewer for the diff/status preview."""

    title: str
    body: str
    scroll: int = 0
    done: bool = False
    on_result: Optional[Callable[[], None]] = None

    @property
    def lines(self) -> list[str]:
        """``body`` split into lines for the renderer + tests."""
        return self.body.splitlines()

    def close(self) -> None:
        if self.done:
            return
        self.done = True

    def handle_key(self, ev: KeyEvent) -> bool:
        if self.done:
            return True
        code = ev.code
        if code in ("q", "Q", "Esc", "Escape", "Enter"):
            self.close()
            return True
        if code in ("j", "Down"):
            self.scroll = min(max(0, len(self.lines) - 1), self.scroll + 1)
            return False
        if code in ("k", "Up"):
            self.scroll = max(0, self.scroll - 1)
            return False
        if code in ("g", "Home"):
            self.scroll = 0
            return False
        if code in ("G", "End"):
            self.scroll = max(0, len(self.lines) - 1)
            return False
        return False


# ── RunListScreen ────────────────────────────────────────────────────────────


# Single-letter binding map. Lifted out of the screen body so tests can
# enumerate the contract without instantiating a screen, mirroring the
# old Textual ``BINDINGS`` array.
_RUN_LIST_BINDINGS: dict[str, str] = {
    "Enter": "open_selected",
    "n": "new_run",
    "s": "send",
    "p": "pause",
    "r": "resume",
    "k": "kill",
    "R": "restart",
    "w": "rewind",
    "a": "apply",
    "v": "revert",
    "d": "diff",
    "q": "quit",
    "Up": "cursor_up",
    "Down": "cursor_down",
    "Home": "cursor_top",
    "End": "cursor_bottom",
}


class RunListScreen:
    """Live run table. The home screen of the bare ``ai`` command.

    Columns mirror ``ai ls``. ``Enter`` opens the per-run detail
    screen; the operator verbs are bound to single keys (see
    :data:`_RUN_LIST_BINDINGS`) and each one resolves the row currently
    under the cursor before opening its modal.

    The poll cadence (1 s) is intentionally slow: ``list_runs``
    iterates every run-dir and reconciles its status, which is cheap
    but not free. A faster refresh wouldn't change what an operator
    can do — runs don't transition that quickly.
    """

    REFRESH_SECONDS: float = 1.0

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self.rows: List[RunRow] = []
        self._rows_by_key: dict[str, RunRow] = {}
        self.cursor_row: int = 0
        # Modals stack on top of the screen. ``handle_key`` routes to
        # the topmost modal first; once that modal flips ``done`` we
        # pop it and invoke its ``on_*`` callback. Stack instead of a
        # single slot because the new-run flow chains three modals
        # (prompt → workspace → backend) and the callbacks push the
        # next layer from inside the previous layer's ``on_submit``.
        self.modals: list[Any] = []
        self.notifications: list[Notification] = []
        # When the operator hits Enter, we hand the App a target
        # ``RunPaths`` for the run they want to open. The App reads
        # this on the next tick, exits the run-list TUI cleanly, and
        # the CLI hands off to the streaming tail (so the local
        # terminal's native scrollback owns navigation). Reading
        # resets the field so multiple Enters don't queue up.
        self.pending_detail: Optional[RunPaths] = None
        # Whether the operator pressed ``q``. The App's outer loop
        # treats this as "exit", same as Textual's ``app.exit()``.
        self.should_exit: bool = False

    # ── lifecycle hooks (called by the App loop) ──

    def on_mount(self) -> None:
        self.refresh_rows()

    def tick(self) -> None:
        """Periodic refresh hook. Driven by the App's draw loop.

        Kept as a single method (instead of multiple ``set_interval``
        timers like the Textual era) because pyratatui's loop already
        ticks at frame rate; we just rate-limit ourselves here."""
        # The App accumulates elapsed ms per loop iteration; we use
        # a simple wall-clock check so a long redraw doesn't compound.
        now = _monotonic()
        if now - self._last_refresh >= self.REFRESH_SECONDS:
            self._last_refresh = now
            self.refresh_rows()

    # ── data plumbing ──

    _last_refresh: float = 0.0

    def refresh_rows(self) -> None:
        """Re-read :func:`list_runs` and reconcile the in-memory state.

        The cursor is pinned by ``run_id`` rather than by row index so a
        structural change (a newer run appearing above the selection,
        the selected run being deleted, runs reordering) never silently
        retargets destructive operator verbs like ``k``/``R``/``a``/``v``
        to a different run. We snapshot the previously-selected run's
        ``run_id`` *before* mutating ``self.rows``, then:

        * if it still exists in the new list, move the cursor to its
          new index (this is the common case — the same run, possibly
          shifted by inserts above it);
        * if it disappeared, fall back to clamping the old numeric
          index into the new range so the cursor lands on the
          neighbour that would have been below the deleted row, which
          is the same affordance an operator gets in any list UI.

        The empty-list case resets to ``0`` so the next non-empty
        refresh starts from the top.
        """
        prev_run_id: Optional[str] = None
        if 0 <= self.cursor_row < len(self.rows):
            prev_run_id = self.rows[self.cursor_row].run_id
        try:
            rows = list(list_runs(self.runs_dir))
        except OSError:
            rows = []
        self.rows = rows
        self._rows_by_key = {r.run_id: r for r in rows}
        if not rows:
            self.cursor_row = 0
            return
        if prev_run_id is not None and prev_run_id in self._rows_by_key:
            for idx, r in enumerate(rows):
                if r.run_id == prev_run_id:
                    self.cursor_row = idx
                    return
        self.cursor_row = max(0, min(self.cursor_row, len(rows) - 1))

    # ── selection helpers ──

    def _selected_run(self) -> Optional[RunRow]:
        if not self.rows:
            return None
        if not (0 <= self.cursor_row < len(self.rows)):
            return None
        return self.rows[self.cursor_row]

    def _selected_paths(self) -> Optional[RunPaths]:
        sel = self._selected_run()
        if sel is None:
            return None
        return RunPaths(runs_dir=self.runs_dir, run_id=sel.run_id)

    def notify(self, text: str, *, severity: str = "information") -> None:
        self.notifications.append(Notification(text=text, severity=severity))

    # ── input dispatch ──

    def handle_key(self, ev: KeyEvent) -> None:
        # If a modal is on top, route there first; pop on completion.
        while self.modals:
            top = self.modals[-1]
            top.handle_key(ev)
            if not getattr(top, "done", False):
                return
            self.modals.pop()
            self._dispatch_modal_result(top)
            # The callback may have pushed a new modal — drop back
            # into the loop if so. Otherwise return: a single key
            # press never kicks off two screen-level actions.
            if self.modals:
                return
            return
        action_name = _RUN_LIST_BINDINGS.get(ev.code)
        if action_name is None:
            return
        method = getattr(self, f"action_{action_name}", None)
        if method is not None:
            method()

    def _dispatch_modal_result(self, modal: Any) -> None:
        """Run *modal*'s registered callback now that it's done."""
        if isinstance(modal, _PromptModal):
            if modal.submitted:
                if modal.on_submit is not None:
                    modal.on_submit(modal.value)
            else:
                if modal.on_cancel is not None:
                    modal.on_cancel()
            return
        if isinstance(modal, _ConfirmModal):
            if modal.on_result is not None:
                modal.on_result(modal.confirmed)
            return
        if isinstance(modal, _BackendChoiceModal):
            if modal.on_result is not None:
                modal.on_result(modal.chosen)
            return
        if isinstance(modal, _DiffViewer):
            if modal.on_result is not None:
                modal.on_result()
            return

    # ── cursor movement ──

    def action_cursor_up(self) -> None:
        if self.rows:
            self.cursor_row = max(0, self.cursor_row - 1)

    def action_cursor_down(self) -> None:
        if self.rows:
            self.cursor_row = min(len(self.rows) - 1, self.cursor_row + 1)

    def action_cursor_top(self) -> None:
        self.cursor_row = 0

    def action_cursor_bottom(self) -> None:
        if self.rows:
            self.cursor_row = len(self.rows) - 1

    # ── verbs ──

    def action_open_selected(self) -> None:
        sel = self._selected_run()
        if sel is None:
            self.notify("(no run selected)", severity="warning")
            return
        # ``initial_log_lines=None`` is the "press Enter on a run"
        # contract: the operator asked for the *full* raw transcript,
        # not a bounded tail. The App reads ``pending_detail`` on the
        # next tick and pushes the detail screen.
        self.pending_detail = RunPaths(
            runs_dir=self.runs_dir, run_id=sel.run_id,
        )

    def action_quit(self) -> None:
        self.should_exit = True

    def action_new_run(self) -> None:
        # Three-step chain: prompt → workspace → backend preset.
        # Each step pushes the next from inside its ``on_submit`` so a
        # cancel at any layer cleanly aborts the flow.
        def on_prompt(text: str) -> None:
            if not text or not text.strip():
                return

            def on_workspace(ws: str) -> None:
                if not ws or not ws.strip():
                    return
                workspace = str(Path(ws).expanduser().resolve())

                def on_backend(choice: Optional[dict]) -> None:
                    if choice is None:
                        return
                    try:
                        cfg = actions.default_run_config(
                            task=text.strip(),
                            workspace=workspace,
                            **choice,
                        )
                    except ValueError as exc:
                        self.notify(
                            f"start failed: {exc}", severity="error",
                        )
                        return
                    result = actions.spawn_runner_detached(self.runs_dir, cfg)
                    if result.ok:
                        if cfg.has_mixed_backends:
                            msg = (
                                f"started run {result.run_id} "
                                f"(impl={cfg.backend_for('impl')}, "
                                f"fix={cfg.backend_for('fix')}, "
                                f"reviewer={cfg.backend_for('reviewer')})"
                            )
                        else:
                            msg = (
                                f"started run {result.run_id} "
                                f"(backend={cfg.backend})"
                            )
                        self.notify(msg, severity="information")
                        self.refresh_rows()
                    else:
                        self.notify(
                            f"start failed: {result.message}",
                            severity="error",
                        )

                self.modals.append(
                    _BackendChoiceModal(on_result=on_backend),
                )

            self.modals.append(
                _PromptModal(
                    title="New run · workspace",
                    hint="Path to the source workspace (the agent's cwd).",
                    placeholder=str(Path.cwd()),
                    value=str(Path.cwd()),
                    on_submit=on_workspace,
                ),
            )

        self.modals.append(
            _PromptModal(
                title="New run · prompt",
                hint="The task description for the agent.",
                placeholder="Implement feature X carefully.",
                on_submit=on_prompt,
            ),
        )

    def _ensure_runner_alive(self, paths: RunPaths) -> bool:
        """Mirror the CLI's mutation liveness gate before a control-file
        write. Re-fetches meta on each call so a runner that exited
        *while the operator was typing in the modal* is still rejected
        at the moment of submit."""
        meta = actions.reload_meta(paths)
        if actions.runner_is_alive(meta):
            return True
        status = meta.get("status") or "unknown"
        pid = meta.get("pid")
        self.notify(
            f"{paths.run_id} is no longer alive "
            f"(status={status}, pid={pid}); not writing control file",
            severity="error",
        )
        return False

    def action_send(self) -> None:
        paths = self._selected_paths()
        if paths is None:
            self.notify("(no run selected)", severity="warning")
            return

        def on_text(text: str) -> None:
            if not text or not text.strip():
                return
            if not self._ensure_runner_alive(paths):
                return
            try:
                actions.write_guidance(paths, text.strip())
            except OSError as exc:
                self.notify(f"send failed: {exc}", severity="error")
                return
            self.notify(
                f"guidance queued for {paths.run_id}",
                severity="information",
            )

        self.modals.append(
            _PromptModal(
                title=f"Send guidance · {paths.run_id}",
                hint="Text steered into the next review prompt.",
                placeholder="Focus on the failing test in foo_test.py",
                on_submit=on_text,
            ),
        )

    def action_pause(self) -> None:
        paths = self._selected_paths()
        if paths is None:
            self.notify("(no run selected)", severity="warning")
            return
        try:
            actions.write_pause(paths)
        except OSError as exc:
            self.notify(f"pause failed: {exc}", severity="error")
            return
        self.notify(f"paused {paths.run_id}", severity="information")

    def action_resume(self) -> None:
        paths = self._selected_paths()
        if paths is None:
            self.notify("(no run selected)", severity="warning")
            return
        try:
            actions.clear_pause(paths)
        except OSError as exc:
            self.notify(f"resume failed: {exc}", severity="error")
            return
        self.notify(f"resumed {paths.run_id}", severity="information")

    def action_kill(self) -> None:
        paths = self._selected_paths()
        if paths is None:
            self.notify("(no run selected)", severity="warning")
            return

        def on_confirm(yes: bool) -> None:
            if not yes:
                return
            meta = actions.reload_meta(paths)
            if actions.signal_runner(paths, meta):
                self.notify(f"killed {paths.run_id}", severity="warning")
            else:
                self.notify(
                    f"{paths.run_id} was already gone",
                    severity="information",
                )
            self.refresh_rows()

        self.modals.append(
            _ConfirmModal(
                title=f"Kill run {paths.run_id}?",
                body="Sends SIGTERM (then SIGKILL after 5 s) to the runner.",
                on_result=on_confirm,
            ),
        )

    def action_restart(self) -> None:
        sel = self._selected_run()
        if sel is None:
            self.notify("(no run selected)", severity="warning")
            return
        run_id = sel.run_id
        paths = RunPaths(runs_dir=self.runs_dir, run_id=run_id)

        def on_confirm(yes: bool) -> None:
            if not yes:
                return
            try:
                from .runner import spec_to_cfg

                spec = read_json(paths.spec)
                cfg = spec_to_cfg(spec)
            except (OSError, KeyError, ValueError) as exc:
                self.notify(
                    f"restart failed: cannot read spec: {exc}",
                    severity="error",
                )
                return
            meta = actions.reload_meta(paths)
            actions.signal_runner(paths, meta)
            agent_type = spec.get("agent_type", "review-loop")
            result = actions.spawn_runner_detached(
                self.runs_dir, cfg,
                agent_type=agent_type,
                restarted_from=run_id,
            )
            if result.ok:
                self.notify(
                    f"restarted as {result.run_id}",
                    severity="information",
                )
                self.refresh_rows()
            else:
                self.notify(
                    f"restart failed: {result.message}",
                    severity="error",
                )

        self.modals.append(
            _ConfirmModal(
                title=f"Restart run {run_id}?",
                body="Old runner is killed; a fresh one spawns from spec.json.",
                on_result=on_confirm,
            ),
        )

    def action_rewind(self) -> None:
        paths = self._selected_paths()
        if paths is None:
            self.notify("(no run selected)", severity="warning")
            return

        def on_text(text: str) -> None:
            if not text or not text.strip():
                return
            if not self._ensure_runner_alive(paths):
                return
            try:
                actions.write_rewind_from_to_string(paths, text.strip())
            except (OSError, ValueError) as exc:
                self.notify(f"rewind failed: {exc}", severity="error")
                return
            self.notify(
                f"rewind queued for {paths.run_id}",
                severity="information",
            )

        self.modals.append(
            _PromptModal(
                title=f"Rewind · {paths.run_id}",
                hint="Format: outer=N,inner=M[,phase=review|fix|after_impl]",
                placeholder="outer=1,inner=1,phase=review",
                on_submit=on_text,
            ),
        )

    def action_apply(self) -> None:
        paths = self._selected_paths()
        if paths is None:
            self.notify("(no run selected)", severity="warning")
            return

        def on_confirm(yes: bool) -> None:
            if not yes:
                return
            from .worktree import apply_to_source

            ok, msg = apply_to_source(paths)
            severity = "information" if ok else "error"
            self.notify(msg, severity=severity)

        self.modals.append(
            _ConfirmModal(
                title=f"Apply worktree changes for {paths.run_id}?",
                body="Applies the run's diff to the source workspace.",
                on_result=on_confirm,
            ),
        )

    def action_revert(self) -> None:
        paths = self._selected_paths()
        if paths is None:
            self.notify("(no run selected)", severity="warning")
            return

        def on_confirm(yes: bool) -> None:
            if not yes:
                return
            from .worktree import revert_from_source

            ok, msg = revert_from_source(paths)
            severity = "information" if ok else "error"
            self.notify(msg, severity=severity)

        self.modals.append(
            _ConfirmModal(
                title=f"Revert applied changes for {paths.run_id}?",
                body="Reverses a previous apply against the source workspace.",
                on_result=on_confirm,
            ),
        )

    def action_diff(self) -> None:
        """Open a read-only modal showing the per-file change summary.

        We deliberately don't try to pipe ``git diff`` through the
        terminal here — operators who want the full diff can drop
        back to ``ai diff <run_id>``. The modal previews the change
        summary so the operator can see which files are affected
        without context-switching."""
        paths = self._selected_paths()
        if paths is None:
            self.notify("(no run selected)", severity="warning")
            return
        from .worktree import (
            is_applied,
            load_worktree_info,
            make_diff_stat,
            make_status_short,
        )

        info = load_worktree_info(paths)
        if info is None:
            self.notify(
                f"{paths.run_id} has no worktree", severity="warning",
            )
            return
        try:
            short = make_status_short(info)
            stat = make_diff_stat(info)
        except RuntimeError as exc:
            self.notify(f"diff failed: {exc}", severity="error")
            return
        applied = is_applied(paths)
        body = (
            f"worktree:        {info.path}\n"
            f"source workspace: {info.source_workspace}\n"
            f"base commit:     {info.base_commit[:12]}\n"
            f"applied to source: {'yes' if applied else 'no'}\n\n"
            f"{short or '(no changes)'}\n{stat}"
        )
        self.modals.append(
            _DiffViewer(title=f"Diff · {paths.run_id}", body=body),
        )


# ── RunDetailScreen ─────────────────────────────────────────────────────────


_RUN_DETAIL_BINDINGS: dict[str, str] = {
    "q": "quit",
    "Esc": "back",
    "Escape": "back",
    "j": "scroll_log_down",
    "Down": "scroll_log_down",
    "k": "scroll_log_up",
    "Up": "scroll_log_up",
    "g": "scroll_log_top",
    "Home": "scroll_log_top",
    "G": "scroll_log_bottom",
    "End": "scroll_log_bottom",
    "f": "toggle_follow",
    "PageDown": "page_down",
    "PageUp": "page_up",
}


class RunDetailScreen:
    """Per-run detail view: minimal status bar + full agent transcript.

    Layout (top → bottom):

    1. **Status bar** — a single line built by
       :func:`_minimal_status_line` summarizing the reconciled run
       state (status / phase / outer-inner / verdict / paused).
    2. **Agent log paragraph** — wraps long lines, fills the rest of
       the screen, scrollable via ``j``/``k``/``g``/``G``/``f``.

    Wrap re-flow on resize is *free* with pyratatui: the ``Paragraph``
    widget is rendered fresh every frame from the raw ``_lines`` ring
    buffer, so a SIGWINCH simply produces a wider/narrower paint on
    the next tick. The Textual era's ``_WrapAwareRichLog`` (which
    mirrored the raw text into a parallel deque so it could replay
    against the new geometry) is no longer needed — every line is
    already the source of truth.

    Follow logic:

    * ``_follow=True`` (default) means ``refresh_log`` keeps the
      viewport pinned to the tail. Incoming appends scroll into view.
    * ``_follow=False`` means appends accumulate in ``_lines`` but
      the viewport doesn't move. The operator can scroll back up
      to inspect history without being yanked back.
    * ``_user_forced_follow_off`` latches the manual override toggled
      by ``f`` so a "near-the-bottom" tick can't silently re-enable
      follow against the operator's wishes.
    """

    def __init__(
        self,
        paths: RunPaths,
        *,
        refresh_seconds: float = 0.5,
        initial_log_lines: Optional[int] = 30,
    ) -> None:
        self.paths = paths
        self.refresh_seconds = max(0.1, float(refresh_seconds))
        self.initial_log_lines: Optional[int] = (
            None if initial_log_lines is None
            else max(1, int(initial_log_lines))
        )

        self._tailer = LogTailer(paths.agent_log)
        self._follow: bool = True
        self._user_forced_follow_off: bool = False

        # Raw text buffer. Capped so a long-lived run can't pin
        # unbounded memory; oldest lines roll off the top. Same shape
        # as the seed cap so the seed and steady-state share a buffer.
        self._lines: Deque[str] = deque(maxlen=_FULL_LOG_SEED_CAP)

        # ``_scroll_offset`` is the number of *rendered rows* to show
        # *before* the tail when not following: 0 means "show the most
        # recent ``viewport_height`` rendered rows", N>0 means "scroll
        # N rendered rows back from the tail". Bounded by
        # ``_total_rendered_rows() - 1``. Counting rendered rows
        # (rather than logical lines) is what makes a single 500-char
        # line scrollable on a narrow terminal: a logical-line count
        # would short-circuit at ``len(_lines) <= 1`` and clip the
        # rest of the wrapped output below the panel border. When
        # follow is on we ignore this (snap to 0 each render).
        self._scroll_offset: int = 0
        # Last viewport geometry the renderer reported. Height drives
        # PageUp / PageDown step size and the "did the operator scroll
        # past the visible bottom" comparison; width feeds the
        # rendered-row math used by the scroll actions and the
        # wrap-aware visible-window helper. The Paragraph renders at
        # the live width every frame, but the action layer needs a
        # cached width because the operator can press ``j``/``k``
        # before the first paint (e.g. immediately after ``ai show``
        # opens) and we still need a sensible clamp.
        self._viewport_height: int = 24
        self._viewport_width: int = 80

        # Status bar: cached so the renderer skips an update when
        # nothing changed (the Textual era paid one diff per tick;
        # pyratatui's diff is cheaper, but skipping a no-op string
        # comparison is still free).
        self.status_text: str = "(loading status...)"
        self._last_status_text: Optional[str] = None

        self.notifications: list[Notification] = []
        self.should_exit: bool = False
        self.should_pop: bool = False

        self._last_status_refresh: float = 0.0
        self._last_log_refresh: float = 0.0
        # Pace the log poll deliberately slower than the previous
        # 0.2 s default. 0.4 s is still well below the threshold at
        # which an agent transcript stops feeling "live" — bursts are
        # drained in a single tick — but it halves the timer-driven
        # render budget the screen pays while the operator scrolls.
        self._log_poll_seconds: float = 0.4

    # ── lifecycle hooks ──

    def on_mount(self) -> None:
        self._seed_initial_log()
        self._refresh_status()

    def tick(self) -> None:
        now = _monotonic()
        if now - self._last_status_refresh >= self.refresh_seconds:
            self._last_status_refresh = now
            self._refresh_status()
        if now - self._last_log_refresh >= self._log_poll_seconds:
            self._last_log_refresh = now
            self._refresh_log()

    # ── refresh handlers ──

    def _refresh_status(self) -> None:
        try:
            text = _strip_ansi(_minimal_status_line(self.paths))
        except Exception as exc:
            text = f"(status unavailable: {exc})"
        if text == self._last_status_text:
            return
        self._last_status_text = text
        self.status_text = text

    def _refresh_log(self) -> None:
        new_lines = self._tailer.read_new_lines()
        if not new_lines:
            return
        appended_rendered = sum(
            _rendered_rows_for(line, self._viewport_width)
            for line in new_lines
        )
        for line in new_lines:
            self._lines.append(line)
        if self._follow:
            # Pinned to tail: the renderer derives the visible window
            # from ``_scroll_offset=0`` so resetting here is the
            # equivalent of the old widget's ``scroll_end``.
            self._scroll_offset = 0
        else:
            # Operator scrolled away. Keep them parked at the same
            # *content* position by bumping ``_scroll_offset`` for
            # every appended *rendered row* — otherwise a burst of
            # writes (especially long agent payloads that fold across
            # multiple rows) would visually shift the viewport down
            # even though we don't auto-snap to tail.
            total = self._total_rendered_rows()
            self._scroll_offset = min(
                max(0, total - 1),
                self._scroll_offset + appended_rendered,
            )

    def _seed_initial_log(self) -> None:
        # Two seeding modes share one EOF-park step at the end:
        #
        # * ``initial_log_lines is None`` — render the trailing
        #   :data:`_FULL_LOG_SEED_CAP` lines of the agent log.
        # * ``initial_log_lines`` is an int — render only that many
        #   trailing lines.
        #
        # Either way :func:`tail_text_file` reads at most a handful of
        # MiB off the tail of the file rather than the whole thing.
        # In both branches we then park the tailer's offset *directly*
        # at EOF — calling ``read_new_lines`` to "burn" historical
        # bytes would be wrong for logs larger than the per-tick cap
        # (~4 MiB), because the second tick would surface historical
        # bytes instead of new appends.
        if self.paths.agent_log.exists():
            seed_lines = (
                _FULL_LOG_SEED_CAP
                if self.initial_log_lines is None
                else self.initial_log_lines
            )
            for line in tail_text_file(
                self.paths.agent_log, lines=seed_lines,
            ):
                self._lines.append(line)
        # Seed always ends at the tail.
        self._scroll_offset = 0
        self._tailer.seek_to_end()

    # ── input ──

    def notify(self, text: str, *, severity: str = "information") -> None:
        self.notifications.append(Notification(text=text, severity=severity))

    def handle_key(self, ev: KeyEvent) -> None:
        action_name = _RUN_DETAIL_BINDINGS.get(ev.code)
        if action_name is None:
            return
        method = getattr(self, f"action_{action_name}", None)
        if method is not None:
            method()

    # ── visible window for the renderer + tests ──

    def _total_rendered_rows(self) -> int:
        """Total number of rendered rows the buffer occupies once
        wrapped at the current ``_viewport_width``.

        Used by the scroll actions to clamp ``_scroll_offset`` to
        valid territory and by :meth:`visible_lines` /
        :func:`_compute_log_window` to find the visible suffix."""
        return _total_rendered_rows(self._lines, self._viewport_width)

    def visible_lines(self) -> list[str]:
        """Logical lines whose rendered rows intersect the visible
        viewport.

        Tests use this to assert what the operator would actually see
        after a sequence of scrolls/appends — without rendering. The
        result is a slice of :attr:`_lines`; with ``.wrap(True)``
        enabled, a single long entry can span multiple rendered rows,
        so the returned list may be shorter than the panel height."""
        body, _scroll_y = _compute_log_window(
            list(self._lines),
            self._viewport_width,
            self._viewport_height,
            self._scroll_offset,
            self._follow,
        )
        return body

    # ── actions ──

    def action_quit(self) -> None:
        self.should_exit = True

    def action_back(self) -> None:
        # Pop back to the run list (if there is one) — otherwise treat
        # ``Esc`` as quit. The bare ``ai`` entry pushes a list; the
        # standalone ``ai show`` pushes only the detail screen.
        self.should_pop = True

    def action_scroll_log_down(self) -> None:
        # Moving towards the tail: shrink ``_scroll_offset`` by one
        # *rendered* row. Hitting 0 means we're back at the tail;
        # auto-resume follow there.
        if self._scroll_offset > 0:
            self._scroll_offset -= 1
        if self._scroll_offset == 0 and not self._user_forced_follow_off:
            self._follow = True

    def action_scroll_log_up(self) -> None:
        # Moving away from the tail: grow ``_scroll_offset`` by one
        # *rendered* row. We clamp by ``total_rendered_rows - 1`` —
        # not by ``len(_lines) - 1`` — so a single long agent line
        # that wraps to many rows is still scrollable. Any move off
        # the tail disengages follow until the operator either
        # presses ``G`` or scrolls all the way back down.
        total = self._total_rendered_rows()
        if total <= 1:
            return
        self._scroll_offset = min(total - 1, self._scroll_offset + 1)
        if self._scroll_offset > 0:
            self._follow = False

    def action_scroll_log_top(self) -> None:
        # ``g`` jumps to the head of the buffer. That's an explicit
        # off-tail gesture, so disengage follow but do NOT latch the
        # manual override — pressing ``G`` from here should still
        # re-arm tailing without a separate ``f``. Clamp uses
        # rendered-row count so a single wrapped line is still
        # scrollable to its first wrapped row.
        self._scroll_offset = max(0, self._total_rendered_rows() - 1)
        if self._scroll_offset > 0:
            self._follow = False

    def action_scroll_log_bottom(self) -> None:
        # Snapping to the bottom is also an implicit "resume follow"
        # gesture, so clear the explicit-off override.
        self._scroll_offset = 0
        self._user_forced_follow_off = False
        self._follow = True

    def action_page_down(self) -> None:
        step = max(1, self._viewport_height - 1)
        self._scroll_offset = max(0, self._scroll_offset - step)
        if self._scroll_offset == 0 and not self._user_forced_follow_off:
            self._follow = True

    def action_page_up(self) -> None:
        # Pages are measured in rendered rows, matching the unit
        # the renderer uses to scroll (Paragraph.scroll(y)). This is
        # what lets PageUp on a long single-line wrapped buffer
        # actually move the visible window — the previous logical-
        # line clamp short-circuited at ``len(_lines) <= 1``.
        total = self._total_rendered_rows()
        if total <= 1:
            return
        step = max(1, self._viewport_height - 1)
        self._scroll_offset = min(total - 1, self._scroll_offset + step)
        if self._scroll_offset > 0:
            self._follow = False

    def action_toggle_follow(self) -> None:
        # ``f`` flips the explicit override. Turning follow ON snaps
        # to the tail so the next tick keeps streaming; turning it OFF
        # latches the override so an operator parked at EOF doesn't
        # have follow silently re-enabled by a stray geometry change.
        if self._follow:
            self._user_forced_follow_off = True
            self._follow = False
        else:
            self._user_forced_follow_off = False
            self._follow = True
            self._scroll_offset = 0
        self.notify(
            f"follow: {'on' if self._follow else 'off'}",
            severity="information",
        )


# ── Apps ────────────────────────────────────────────────────────────────────


def _monotonic() -> float:
    """Wrapped ``time.monotonic`` so tests can monkeypatch it."""
    import time as _time

    return _time.monotonic()


class _AppBase:
    """Common scaffolding for both Apps.

    Holds the screen stack, dispatches keys to the topmost screen, and
    provides ``tick_once`` / ``dispatch_key`` entry points the test
    suite drives directly. Subclasses implement :meth:`run` (the live
    pyratatui loop) and :meth:`_initial_screens`."""

    def __init__(self) -> None:
        self.screens: list[Any] = []
        self._exited: bool = False
        # Set when the run-list screen surfaces a "stream this run"
        # selection (operator hit Enter on a row). The CLI reads this
        # *after* :meth:`run` returns and dispatches into
        # :func:`auto_iterator.display.stream_log`, so the local
        # terminal's native scrollback owns navigation rather than
        # the pyratatui frame loop. ``None`` means "no follow-up,
        # plain exit".
        self.streamed_run: Optional[RunPaths] = None

    @property
    def screen(self) -> Any:
        return self.screens[-1] if self.screens else None

    @property
    def _exit(self) -> bool:
        """Compatibility alias matching the old Textual ``app._exit`` flag.

        Kept because some callers (smoke tests, external scripts) read
        this attribute as a "did the app's loop terminate cleanly?"
        sentinel."""
        return self._exited

    def push_screen(self, screen: Any) -> None:
        self.screens.append(screen)
        if hasattr(screen, "on_mount"):
            screen.on_mount()

    def pop_screen(self) -> None:
        if self.screens:
            self.screens.pop()

    # ── unit-test friendly drivers ──

    def dispatch_key(self, ev: Any) -> None:
        """Forward *ev* to the topmost screen and reconcile transitions.

        ``Ctrl-C`` and ``Ctrl-D`` short-circuit here as an unconditional
        exit — same UX contract as ``claude code`` and ``codex``. We do
        this at the App layer rather than in every screen/modal binding
        because pyratatui delivers the interrupt as a regular key event
        (``ctrl=True, code="c"``) once the terminal is in raw mode, and
        in raw mode the kernel will *not* synthesize SIGINT for us.
        Without this short-circuit an operator who hit Ctrl-C inside a
        modal (or any screen whose binding map didn't list it) would be
        stuck in the alternate screen with no way out short of ``q``."""
        norm = _to_key_event(ev)
        if norm is None or self.screen is None:
            return
        if norm.ctrl and norm.code in ("c", "C", "d", "D"):
            self._exited = True
            self.screens.clear()
            return
        self.screen.handle_key(norm)
        self._reconcile()

    def tick_once(self) -> bool:
        """Run one iteration of the loop without a render. Returns
        ``False`` when the app should exit (for the live loop)."""
        if self.screen is None:
            return False
        if hasattr(self.screen, "tick"):
            self.screen.tick()
        self._reconcile()
        return self.screen is not None

    def _reconcile(self) -> None:
        """Resolve any pending screen transitions from the top screen."""
        scr = self.screen
        if scr is None:
            return
        if getattr(scr, "should_exit", False):
            self._exited = True
            self.screens.clear()
            return
        if getattr(scr, "should_pop", False):
            scr.should_pop = False
            self.pop_screen()
            if not self.screens:
                self._exited = True
            return
        pending = getattr(scr, "pending_detail", None)
        if pending is not None:
            scr.pending_detail = None
            # Hand the selected run to the CLI so it can stream the
            # transcript on the regular screen buffer. We deliberately
            # do NOT push a pyratatui ``RunDetailScreen`` here: that
            # path puts every PageUp / mouse-wheel keystroke through a
            # network round-trip back to the runner host, which is
            # exactly the lag this whole change exists to remove.
            # Instead, exit the app cleanly; the CLI inspects
            # ``self.streamed_run`` and calls ``stream_log`` so the
            # local terminal's native scrollback handles navigation.
            self.streamed_run = pending
            self._exited = True
            self.screens.clear()
            return


class RunListApp(_AppBase):
    """Top-level app for the bare ``ai`` command (no subcommand)."""

    TITLE = "auto-iterator"
    SUB_TITLE = "operator console"

    def __init__(self, runs_dir: Path) -> None:
        super().__init__()
        self.runs_dir = runs_dir
        self.push_screen(RunListScreen(runs_dir))

    def run(self) -> int:
        return _run_app_loop(self)


class RunDetailApp(_AppBase):
    """Top-level app for ``ai show <run_id>`` (TTY default).

    Pushes only the detail screen; ``Esc`` becomes a quit instead of
    pop because there's no parent screen to fall back to."""

    TITLE = "auto-iterator"

    def __init__(
        self,
        paths: RunPaths,
        *,
        refresh_seconds: float = 0.5,
        initial_log_lines: Optional[int] = 30,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.refresh_seconds = refresh_seconds
        self.initial_log_lines = initial_log_lines
        self.SUB_TITLE = paths.run_id
        self.push_screen(
            RunDetailScreen(
                paths,
                refresh_seconds=refresh_seconds,
                initial_log_lines=initial_log_lines,
            )
        )

    def run(self) -> int:
        return _run_app_loop(self)


# ── pyratatui rendering / event loop ────────────────────────────────────────


def _run_app_loop(app: _AppBase) -> int:
    """Drive *app* through a live pyratatui render/event loop.

    Lazy imports :mod:`pyratatui` so the non-TTY paths in
    :mod:`auto_iterator.cli` never pull in the native binding.
    """
    import pyratatui as pr  # noqa: F401  (loaded for side effects)

    poll_timeout_ms = 30  # ~33 fps cap; keystrokes preempt this anyway.

    with pr.Terminal() as term:
        while True:
            # 1. Drain input. ``poll_event`` blocks up to ``timeout_ms``.
            #
            # Defense in depth for Ctrl-C: pyratatui *normally* delivers
            # the interrupt as a key event (handled by
            # :meth:`_AppBase.dispatch_key`), but if the binding ever
            # escalates SIGINT to a Python ``KeyboardInterrupt`` instead,
            # we still want to cleanly tear down the alt-screen rather
            # than leave the operator stranded with garbled echo.
            try:
                ev = term.poll_event(timeout_ms=poll_timeout_ms)
            except KeyboardInterrupt:
                break
            if ev is not None:
                app.dispatch_key(ev)
                if app._exit:
                    break

            # 2. Periodic state refresh.
            try:
                app.tick_once()
            except KeyboardInterrupt:
                break
            if app._exit:
                break

            # 3. Paint the current screen.
            scr = app.screen
            if scr is None:
                break

            def _ui(frame: Any, _scr: Any = scr) -> None:
                _render_screen(_scr, frame, pr)

            try:
                term.draw(_ui)
            except Exception as exc:  # noqa: BLE001
                # A render failure is bad but should not crash the
                # app — surface it as a notification so the operator
                # sees something rather than a stack trace, then keep
                # going. The next tick will retry.
                #
                # We attach the exception class + message so a broken
                # widget API (e.g. a pyratatui upgrade that changes a
                # constructor signature) is *visible* in the footer
                # rather than silently presenting an empty screen.
                msg = f"render error: {type(exc).__name__}: {exc}"
                if hasattr(scr, "notify"):
                    scr.notify(msg, severity="error")

    return 0


def _render_screen(scr: Any, frame: Any, pr: Any) -> None:
    """Top-level dispatch into per-screen ratatui paint routines."""
    if isinstance(scr, RunListScreen):
        _render_run_list(scr, frame, pr)
    elif isinstance(scr, RunDetailScreen):
        _render_run_detail(scr, frame, pr)


def _render_run_list(scr: RunListScreen, frame: Any, pr: Any) -> None:
    """Paint the run-list screen: header / table / modal overlay / footer."""
    Block = pr.Block
    Constraint = pr.Constraint
    Direction = pr.Direction
    Layout = pr.Layout
    Paragraph = pr.Paragraph
    Style = pr.Style
    Color = pr.Color
    Row = pr.Row
    Table = pr.Table
    TableState = pr.TableState

    rows = (
        Layout()
        .direction(Direction.Vertical)
        .constraints([
            Constraint.length(1),  # title bar
            Constraint.fill(1),    # table
            Constraint.length(1),  # footer / latest notification
        ])
        .split(frame.area)
    )

    # Title bar.
    title = "auto-iterator · operator console"
    frame.render_widget(
        Paragraph.from_string(title)
            .style(Style().fg(Color.cyan()).bold()),
        rows[0],
    )

    # Table.
    header = Row.from_strings(
        ["RUN_ID", "STATUS", "PHASE", "O/I", "VERDICT", "UPDATED", "PROMPT"],
    ).style(Style().fg(Color.cyan()).bold())
    table_rows = [
        Row.from_strings(list(_row_cells(r))) for r in scr.rows
    ]
    state = TableState()
    if scr.rows:
        state.select(min(scr.cursor_row, len(scr.rows) - 1))
    constraints = [
        Constraint.length(22),  # RUN_ID
        Constraint.length(10),  # STATUS
        Constraint.length(10),  # PHASE
        Constraint.length(6),   # O/I
        Constraint.length(14),  # VERDICT
        Constraint.length(25),  # UPDATED
        Constraint.fill(1),     # PROMPT
    ]
    # pyratatui 0.2.x: ``Table`` accepts a single positional ``rows``
    # arg; column widths and the header must be set via the chained
    # ``.column_widths(...).header(...)`` builder methods. Passing the
    # widths or header positionally raises ``TypeError`` and the live
    # run-list never paints — the migration regressed once on this and
    # the render-path test pins it now.
    table = (
        Table(table_rows)
        .column_widths(constraints)
        .header(header)
        .block(Block().bordered().title("runs"))
        .highlight_style(Style().fg(Color.yellow()).bold())
        .highlight_symbol("▶ ")
    )
    frame.render_stateful_table(table, rows[1], state)

    # Footer: latest notification + key hints.
    footer_text = _latest_notification_text(scr.notifications) or (
        "n new · s send · p pause · r resume · k kill · "
        "R restart · w rewind · a apply · v revert · d diff · q quit"
    )
    frame.render_widget(
        Paragraph.from_string(footer_text)
            .style(Style().fg(Color.gray())),
        rows[2],
    )

    # Modal overlay.
    if scr.modals:
        _render_modal(scr.modals[-1], frame, pr)


def _render_run_detail(scr: RunDetailScreen, frame: Any, pr: Any) -> None:
    """Paint the per-run detail screen: status bar + scrollable log."""
    Block = pr.Block
    Constraint = pr.Constraint
    Direction = pr.Direction
    Layout = pr.Layout
    Paragraph = pr.Paragraph
    Style = pr.Style
    Color = pr.Color

    rows = (
        Layout()
        .direction(Direction.Vertical)
        .constraints([
            Constraint.length(1),
            Constraint.fill(1),
            Constraint.length(1),
        ])
        .split(frame.area)
    )

    # Status bar.
    frame.render_widget(
        Paragraph.from_string(scr.status_text)
            .style(Style().fg(Color.cyan()).bold()),
        rows[0],
    )

    # Update the screen's notion of the viewport geometry so its
    # scroll math (PageUp / PageDown step, the rendered-row clamp on
    # ``_scroll_offset``, the "did the operator scroll past the
    # visible bottom?" comparison) matches what's actually painted.
    # Subtract 2 from height/width for the panel's top/bottom and
    # left/right borders so the visible cell count matches the
    # renderable inner area.
    log_area = rows[1]
    log_height = max(1, log_area.height - 2)
    log_width = max(1, log_area.width - 2)
    scr._viewport_height = log_height
    scr._viewport_width = log_width

    # Compose the visible window.
    #
    # The Paragraph is rendered with ``.wrap(True, False)`` so a long
    # agent line (a 500-char tool-call payload, a wide diff hunk, a
    # stack trace with embedded paths) folds onto the next visible
    # row instead of being clipped at the terminal's right edge. The
    # ``trim=False`` keeps leading whitespace on continuation rows so
    # indentation in tool/diff output reads correctly. This restores
    # the Textual era's ``RichLog(wrap=True)`` contract.
    #
    # ``_compute_log_window`` returns the slice of ``_lines`` whose
    # rendered rows intersect the visible window plus an intra-line
    # ``scroll_y`` offset. We then ask pyratatui to scroll the
    # Paragraph by ``scroll_y`` rendered rows — that's how a long
    # wrapped line becomes scrollable past the first screenful.
    # Without ``Paragraph.scroll``, pyratatui's wrap renders the line
    # from the top and clips overflow at the panel border, so any
    # rendered row beyond the first ``log_height`` rows of a long
    # line would be unreachable for the operator.
    #
    # In follow mode we always want the *latest* rendered rows
    # visible. ``_compute_log_window`` derives that target window from
    # the buffer + viewport; the renderer just hands the result to
    # pyratatui. In paused mode the operator's ``_scroll_offset``
    # (in rendered rows) selects the window.
    body_lines, scroll_y = _compute_log_window(
        list(scr._lines), log_width, log_height,
        scr._scroll_offset, scr._follow,
    )
    body_text = "\n".join(body_lines)
    follow_marker = "FOLLOW" if scr._follow else "PAUSED"
    frame.render_widget(
        Paragraph.from_string(body_text)
            .wrap(True, False)
            .scroll(scroll_y, 0)
            .block(
                Block()
                .bordered()
                .title(f"agent · {follow_marker}")
            ),
        log_area,
    )

    # Footer: latest notification + key hints.
    footer_text = _latest_notification_text(scr.notifications) or (
        "j/k scroll · g/G top/bottom · f follow · q quit"
    )
    frame.render_widget(
        Paragraph.from_string(footer_text)
            .style(Style().fg(Color.gray())),
        rows[2],
    )


_CURSOR_GLYPH = "▏"


def _build_prompt_modal_text(modal: "_PromptModal", pr: Any) -> Any:
    """Build the styled ``pyratatui.Text`` that the prompt modal paints.

    Lifted out of :func:`_render_modal` so tests can introspect the
    resulting ``Line`` / ``Span`` tree without round-tripping through
    a live ``Paragraph`` (which doesn't expose its text on the Python
    side). The cursor position is rendered two ways at once:

    1. A visible :data:`_CURSOR_GLYPH` (``▏``, LEFT ONE EIGHTH BLOCK)
       inserted *between* characters at ``modal.cursor``. This makes
       the cell at the cursor position change *content* on every
       Left/Right/Home/End, which is what the diff render reliably
       catches — pyratatui emits a buggy SGR sequence for cell
       updates that change *only* the style modifier
       (``\\x1b[7m\\x1b[;m`` cancels itself), so a style-only marker
       was invisible after navigating in the live TUI.
    2. A reverse-video span on the character *under* the cursor
       (``modal.value[modal.cursor]``). This survives the same diff
       path on the open / value-mutation paths and gives the operator
       a thicker visual anchor than the thin glyph alone.

    When ``value`` is empty and a ``placeholder`` is set, the cursor
    sits at column 0 — we paint the glyph plus a reversed space, then
    render the placeholder dimmed inside square brackets so the
    operator can tell the placeholder isn't yet "real" input. The
    next printable keystroke replaces this branch with the
    value-bearing one.
    """
    Color = pr.Color
    Line = pr.Line
    Span = pr.Span
    Style = pr.Style
    Text = pr.Text

    cursor_style = Style().reversed()
    if not modal.value and modal.placeholder:
        prompt_line = Line(spans=[
            Span("> "),
            Span(_CURSOR_GLYPH),
            Span(" ", style=cursor_style),
            Span(
                f"[{modal.placeholder}]",
                style=Style().fg(Color.gray()),
            ),
        ])
    else:
        cursor = max(0, min(len(modal.value), modal.cursor))
        before = modal.value[:cursor]
        at = modal.value[cursor:cursor + 1] or " "
        after = (
            modal.value[cursor + 1:]
            if cursor < len(modal.value)
            else ""
        )
        prompt_line = Line(spans=[
            Span("> "),
            Span(before),
            Span(_CURSOR_GLYPH),
            Span(at, style=cursor_style),
            Span(after),
        ])

    text = Text()
    text.push_line(Line.from_string(modal.title))
    text.push_line(Line.from_string(""))
    if modal.hint:
        text.push_line(Line.from_string(modal.hint))
        text.push_line(Line.from_string(""))
    text.push_line(prompt_line)
    text.push_line(Line.from_string(""))
    text.push_line(Line.from_string("Enter to submit · Esc to cancel"))
    return text


_PROMPT_MODAL_FOOTER = "Enter to submit · Esc to cancel"


def _compute_prompt_modal_scroll(
    modal: "_PromptModal", inner_width: int, inner_height: int,
) -> int:
    """Pick the ``Paragraph.scroll`` y-offset that keeps ``modal.cursor``
    visible inside the modal's bordered inner area.

    Mirrors :func:`_compute_log_window`'s job for the agent-log panel:
    when the wrapped prompt text exceeds ``inner_height`` rendered rows
    (a 500-char paste in an 80×24 terminal lands here), the cursor row
    can fall below the modal's bottom border and the operator can no
    longer see what they're typing — the exact symptom the original
    bug flagged. We simulate ratatui's word-wrap for every line in
    the modal text (title, blank, optional hint+blank, prompt, blank,
    footer), find the rendered row the cursor glyph lands on, and
    return enough scroll to pin that row to the bottom of the visible
    window. Content that already fits returns ``0`` so the title
    stays at the top.

    The prompt line is reconstructed exactly as
    :func:`_build_prompt_modal_text` builds it (``"> "`` prefix +
    value-before-cursor + cursor glyph + value-at-cursor +
    value-after-cursor) so the wrap simulation processes the same
    grapheme stream pyratatui paints. Reusing
    :func:`_word_wrap_indices` (the WordWrapper port) for both the
    surrounding lines and the prompt line keeps the math honest for
    space-separated prompts: a 40-word sentence at width 60 lands on
    ~40 rendered rows, not the ~21 the previous
    ``ceil(cells / width)`` estimate predicted, so the cursor stays
    on screen for ordinary typed prompts (the reviewer's pin).
    """
    inner_width = max(1, inner_width)
    inner_height = max(1, inner_height)

    # Reconstruct the prompt line the same way
    # ``_build_prompt_modal_text`` does — span boundaries don't
    # affect ratatui's wrap (it walks graphemes, not spans), so the
    # joined string is what gets reflowed.
    cursor = max(0, min(len(modal.value), modal.cursor))
    if not modal.value and modal.placeholder:
        # Spans: "> " + "▏" + " " + "[placeholder]" → joined.
        prefix_chars = 2  # "> "
        prompt_line = (
            "> "
            + _CURSOR_GLYPH
            + " "
            + f"[{modal.placeholder}]"
        )
        cursor_grapheme_idx = prefix_chars
    else:
        before = modal.value[:cursor]
        at = modal.value[cursor:cursor + 1] or " "
        after = (
            modal.value[cursor + 1:]
            if cursor < len(modal.value)
            else ""
        )
        prompt_line = "> " + before + _CURSOR_GLYPH + at + after
        cursor_grapheme_idx = 2 + len(before)  # index of cursor glyph

    # Each surrounding line's row count via the shared word-wrap
    # primitive so they agree with how ratatui actually flows them.
    title_rows = _rendered_rows_for(modal.title, inner_width)
    hint_rows = (
        _rendered_rows_for(modal.hint, inner_width) if modal.hint else 0
    )
    footer_rows = _rendered_rows_for(_PROMPT_MODAL_FOOTER, inner_width)

    prompt_wrapped = _word_wrap_indices(
        prompt_line, inner_width, trim=False,
    )
    prompt_rows = max(1, len(prompt_wrapped))

    # Locate the cursor glyph's wrapped row inside the prompt line.
    # If we somehow miss it (truncated by a width-larger-than-row
    # codepoint, which ratatui drops), fall back to the last row
    # so the cursor stays visible at the bottom rather than off the
    # top.
    cursor_row_in_prompt = prompt_rows - 1
    for ri, row in enumerate(prompt_wrapped):
        if cursor_grapheme_idx in row:
            cursor_row_in_prompt = ri
            break

    # Title (1 line block) + blank + optional (hint + blank) sit
    # above the prompt; their rendered-row sum is where the prompt
    # actually starts in the wrapped output.
    prompt_start_row = title_rows + 1
    if modal.hint:
        prompt_start_row += hint_rows + 1

    cursor_row = prompt_start_row + cursor_row_in_prompt
    total_rows = (
        prompt_start_row + prompt_rows + 1 + footer_rows
    )

    if total_rows <= inner_height:
        return 0

    if cursor_row >= inner_height:
        scroll_y = cursor_row - inner_height + 1
    else:
        scroll_y = 0

    # Don't scroll past the tail — would just paint blank rows below
    # the content and waste vertical space the operator could see.
    scroll_y = min(scroll_y, max(0, total_rows - inner_height))
    return scroll_y


def _render_modal(modal: Any, frame: Any, pr: Any) -> None:
    """Centered overlay box for any of the modal state classes."""
    Block = pr.Block
    Clear = pr.Clear
    Color = pr.Color
    Constraint = pr.Constraint
    Direction = pr.Direction
    Layout = pr.Layout
    Paragraph = pr.Paragraph
    Style = pr.Style

    # Centered 80% wide / auto-tall box.
    horiz = (
        Layout()
        .direction(Direction.Horizontal)
        .constraints([
            Constraint.percentage(10),
            Constraint.percentage(80),
            Constraint.percentage(10),
        ])
        .split(frame.area)
    )
    vert = (
        Layout()
        .direction(Direction.Vertical)
        .constraints([
            Constraint.percentage(20),
            Constraint.percentage(60),
            Constraint.percentage(20),
        ])
        .split(horiz[1])
    )
    area = vert[1]

    # Wipe the underlying area before drawing the modal.
    frame.render_widget(Clear(), area)

    if isinstance(modal, _PromptModal):
        text = _build_prompt_modal_text(modal, pr)
        # ``wrap(True, False)`` folds a long ``modal.value`` onto the
        # next visible row instead of clipping at the modal's right
        # edge. ``trim=False`` preserves leading whitespace on the
        # continuation rows so a multi-line prompt that's been pasted
        # in keeps its indentation legible.
        #
        # Wrapping alone isn't enough: a long value can render more
        # rows than the modal's inner height, which would clip the
        # cursor below the bottom border. ``_compute_prompt_modal_scroll``
        # picks the ``Paragraph.scroll`` offset that pins the cursor's
        # row to the bottom of the visible window when it would
        # otherwise overflow, mirroring the agent-log panel's
        # rendered-row scroll model (see :func:`_compute_log_window`).
        inner_width = max(1, area.width - 2)
        inner_height = max(1, area.height - 2)
        scroll_y = _compute_prompt_modal_scroll(
            modal, inner_width, inner_height,
        )
        frame.render_widget(
            Paragraph(text)
                .wrap(True, False)
                .scroll(scroll_y, 0)
                .block(Block().bordered().title("prompt"))
                .style(Style().fg(Color.white())),
            area,
        )
        return

    if isinstance(modal, _ConfirmModal):
        body_lines = [modal.title, ""]
        if modal.body:
            body_lines.extend(modal.body.splitlines())
            body_lines.append("")
        body_lines.append("y to confirm · n / Esc to cancel")
        frame.render_widget(
            Paragraph.from_string("\n".join(body_lines))
                .block(Block().bordered().title("confirm"))
                .style(Style().fg(Color.yellow())),
            area,
        )
        return

    if isinstance(modal, _BackendChoiceModal):
        body_lines = [
            "New run · backend",
            "",
            "Pick the backend layout for this run.",
            "Press 1 / 2, ↑/↓+Enter, or Esc to cancel.",
            "",
        ]
        for idx, preset in enumerate(modal.PRESETS):
            marker = "▶" if idx == modal.selected_idx else " "
            body_lines.append(f"{marker} {preset['label']}")
            for sub in preset["summary"].splitlines():
                body_lines.append(f"    {sub}")
            body_lines.append("")
        frame.render_widget(
            Paragraph.from_string("\n".join(body_lines))
                .block(Block().bordered().title("backend"))
                .style(Style().fg(Color.white())),
            area,
        )
        return

    if isinstance(modal, _DiffViewer):
        height = max(1, area.height - 2)
        lines = modal.lines
        start = min(modal.scroll, max(0, len(lines) - 1))
        end = min(len(lines), start + height)
        body = "\n".join(lines[start:end])
        frame.render_widget(
            Paragraph.from_string(f"{modal.title}\n\n{body}")
                .block(Block().bordered().title("diff"))
                .style(Style().fg(Color.white())),
            area,
        )
        return


def _resolve_codepoint_widths(line: str) -> List[int]:
    """For each codepoint in *line*, return the cell width it
    contributes under ratatui's cluster-aware width rules.

    Ratatui's ``cell_width()`` (in ``ratatui-core/src/buffer/
    cell_width.rs``) calls ``UnicodeWidthStr::width()`` from the
    ``unicode-width`` 0.2.x crate, which implements **string-level
    rules** for emoji ligatures on top of the per-codepoint table:

    * Well-formed, fully-qualified **emoji ZWJ sequences**
      (``👨\u200d👩\u200d👦`` family, ``🧑\u200d💻`` profession,
      ``👩\u200d❤️\u200d👨`` couple, …) → **2 cells** total.
    * **Emoji modifier sequences** (a base + skin tone modifier,
      ``👋🏻``) → **2 cells** total.
    * **Emoji presentation sequences** (a narrow base + VS16
      ``\\ufe0f``, ``☁️``, ``❤️``, ``⚡️``) → **2 cells** total.

    These rules apply to a *grapheme cluster* as a unit, not to its
    constituent codepoints. The per-codepoint table would say a
    family ZWJ emoji is ``2+0+2+0+2 = 6`` cells (each emoji base
    East-Asian-Wide, each ZWJ zero-width), but ratatui's
    ``set_string`` renders one as a single 2-cell wide grapheme,
    same as plain ``界``. Without cluster awareness, the wrap math
    over-counted prompt rows by a factor of ~3 for emoji-heavy
    prompts, and the cursor row landed *above* the modal's visible
    window — the symptom the latest review pinned for
    ``👨\u200d👩\u200d👦`` × 100 at 60×8.

    For each codepoint we attribute the cluster's full cell width
    to its first codepoint and zero out the rest. The wrap loop in
    :func:`_word_wrap_indices` walks codepoints (not graphemes) and
    sums these values; concentrating each cluster's width in a
    single codepoint lets that loop break exactly where ratatui's
    grapheme-aware wrapper would, modulo the harmless detail that
    we may carry an entire cluster's codepoints into the wrapped
    row even when only the first one would have crossed the row
    boundary in ratatui (the cluster's other codepoints are 0-cell,
    so they don't shift any visible cell).

    Cluster detection (no dependency on grapheme-segmentation):

    * A cluster begins at any non-extension codepoint.
    * Extensions, consumed greedily after the base:
      - ``\\ufe0f`` (VS16, emoji presentation selector) — sets the
        cluster's ``has_vs16`` flag.
      - ``\\ufe0e`` (VS15, text presentation selector) — extends but
        does not widen.
      - ``\\U0001F3FB``..``\\U0001F3FF`` (emoji modifier base /
        skin-tone modifiers) — sets ``has_modifier``.
      - General category ``Mn`` (nonspacing mark) / ``Me``
        (enclosing mark) — combining marks, extend with no widening.
      - General category ``Cf`` — format chars (other than ZWJ),
        zero-width.
      - ``\\u200d`` (ZWJ) — *only* if there is a next codepoint to
        join. Sets ``has_zwj_join``, consumes the ZWJ + the joined
        base; the loop then continues, picking up the joined base's
        own extensions.

    Cluster width:

    * ``has_zwj_join`` or ``has_modifier`` → 2 cells (per the
      emoji-ligature rules above).
    * ``has_vs16`` (without the above) → 2 cells (emoji-presentation
      sequence widens to 2).
    * Otherwise → sum of per-codepoint widths within the cluster
      (combining marks 0, East Asian W/F 2, rest 1).

    Permissive heuristic: any base + ZWJ + base / + modifier / +
    VS16 widens to 2 even for non-emoji codepoints (Python's stdlib
    doesn't ship the Unicode ``Emoji`` / ``Emoji_Presentation``
    tables). Edge cases like ``a\\u200da`` over-estimate by 1 cell
    — the safe direction (over-scrolling clamps at the tail;
    under-scrolling clips the cursor).
    """
    n = len(line)
    widths = [0] * n
    i = 0
    while i < n:
        start = i
        ch = line[i]
        cp = ord(ch)
        cat = unicodedata.category(ch)

        # Solo extension / zero-width codepoints at a cluster boundary
        # don't form a 2-cell emoji cluster on their own — a stray
        # VS16 / ZWJ / emoji modifier without a preceding base
        # contributes 0 cells, same as a stray combining mark. Without
        # this guard, ``\ufe0f`` repeated 50 times would be modelled
        # as one cluster with ``has_vs16=True`` and credited 2 cells,
        # disagreeing with ratatui.
        is_extension = (
            ch in ("\ufe0f", "\ufe0e", "\u200d")
            or 0x1F3FB <= cp <= 0x1F3FF
            or cat in ("Mn", "Me", "Cf", "Cc")
        )
        if is_extension:
            i += 1
            continue

        i += 1
        has_zwj_join = False
        has_modifier = False
        has_vs16 = False
        while i < n:
            ch = line[i]
            cp = ord(ch)
            if ch == "\ufe0f":
                has_vs16 = True
                i += 1
                continue
            if ch == "\ufe0e":
                # VS15 — text presentation. Extend, no widening.
                i += 1
                continue
            if 0x1F3FB <= cp <= 0x1F3FF:
                has_modifier = True
                i += 1
                continue
            cat = unicodedata.category(ch)
            if cat in ("Mn", "Me"):
                i += 1
                continue
            if ch == "\u200d":
                # ZWJ only joins if there's a next base codepoint.
                if i + 1 < n:
                    has_zwj_join = True
                    i += 1  # consume the ZWJ
                    i += 1  # consume the joined base; its own
                            # extensions will be picked up by the
                            # next iteration of this loop.
                    continue
                break
            if cat == "Cf":
                # Other format chars (zero-width).
                i += 1
                continue
            break

        if has_zwj_join or has_modifier or has_vs16:
            cluster_w = 2
        else:
            cluster_w = 0
            for k in range(start, i):
                ch = line[k]
                cat = unicodedata.category(ch)
                if cat in ("Mn", "Me", "Cf", "Cc"):
                    continue
                if unicodedata.east_asian_width(ch) in ("W", "F"):
                    cluster_w += 2
                else:
                    cluster_w += 1
        widths[start] = cluster_w
    return widths


def _display_cell_width(s: str) -> int:
    """Display width of *s* in terminal cells.

    Ratatui's wrap implementation breaks lines based on terminal cell
    width (via the ``unicode-width`` crate's grapheme-aware
    ``cell_width()``), not Python codepoint count. We approximate
    that here so the prompt-modal scroll math and the agent-log row
    math agree with how pyratatui actually flows the rendered
    Paragraph.

    Width is the sum of :func:`_resolve_codepoint_widths`, which
    encodes both the per-codepoint width rules (combining marks /
    format chars / control chars → 0; East Asian W or F → 2; rest
    → 1) and the cluster-level emoji-ligature rules from
    ``unicode-width`` 0.2.x — emoji ZWJ sequences, emoji modifier
    sequences, and emoji presentation sequences each widen to a
    single 2-cell grapheme.

    Without those cluster rules, prompts containing 100 family
    emojis were modelled as ~600 cells (each family =
    ``2+0+2+0+2``), but ratatui paints them as 200 cells (each
    family = a single 2-cell wide grapheme). The wrap math
    over-counted rows, the cursor row was computed on a non-existent
    row above the visible window, and the operator typed blind —
    the reviewer's latest pin.
    """
    return sum(_resolve_codepoint_widths(s))


def _word_wrap_indices(
    line: str, max_width: int, trim: bool = False,
) -> List[List[int]]:
    """Word-wrap *line* to *max_width* cells, returning one list of
    Python codepoint indices per rendered row.

    This is a Python port of ratatui's ``WordWrapper`` (see
    ``ratatui-widgets/src/reflow.rs``) with the same semantics
    pyratatui's ``Paragraph.wrap(True, trim)`` invokes:

    * Words are runs of non-whitespace graphemes; whitespace runs
      separate them. ``\\xa0`` (NBSP) is *not* a separator, matching
      ``str::is_whitespace``-style rules in ratatui.
    * When the running line plus the next word would overflow, the
      running line is committed and the word lands on a fresh row.
      With ``trim=False`` the leading whitespace of a wrapped row is
      preserved (so pasted multi-line prompts keep their
      indentation legible); with ``trim=True`` it's discarded.
    * A word longer than ``max_width`` itself gets character-wrapped:
      the algorithm emits as many full rows as fit, leaving the tail
      on the next row.
    * Per-codepoint width comes from :func:`_resolve_codepoint_widths`,
      which detects emoji clusters (ZWJ sequences, modifier
      sequences, VS16 presentation sequences) and concentrates each
      cluster's full 2-cell width on its first codepoint. The
      remaining codepoints in the cluster contribute 0 cells, so
      walking codepoints sums to the same total ratatui's
      grapheme-aware ``cell_width()`` would compute, and wrap breaks
      land on the same row boundaries the renderer paints.

    The returned indices map back to the original *line* so callers
    can ask "which rendered row does ``line[idx]`` land on" — the
    primitive :func:`_compute_prompt_modal_scroll` uses to track the
    cursor glyph through wrapped output. Callers needing only row
    counts (the agent-log scroll math) use :func:`_rendered_rows_for`,
    which is a thin wrapper around this.

    The previous helper used ``ceil(cell_width / width)`` as a row
    estimate, which under-counted rows for ordinary sentence-like
    prompts: words can't be split mid-grapheme, so a 30-char word
    in a 60-cell row that already has 32 cells of content gets
    pushed onto its own line, leaving the row half-empty. The
    cell-slicing math ignored that, mis-positioned the cursor by
    enough rows to fall outside the modal's visible window for
    space-separated prompts, and forced the operator to type
    blind — exactly the symptom the latest review pinned.
    """
    if max_width <= 0:
        return [list(range(len(line)))]

    wrapped: List[List[int]] = []
    pending_line: List[int] = []
    line_width = 0
    pending_word: List[int] = []
    word_width = 0
    pending_ws: Deque[Tuple[int, int]] = deque()
    whitespace_width = 0
    non_whitespace_previous = False

    # Pre-resolve per-codepoint widths so emoji ZWJ sequences /
    # modifier sequences / VS16 presentation sequences each
    # contribute a single 2-cell wide cluster (with the cluster's
    # full width concentrated on its first codepoint). Computing
    # this once outside the loop also keeps the cluster-detection
    # state machine from leaking into the wrap algorithm.
    sym_widths = _resolve_codepoint_widths(line)

    for idx, ch in enumerate(line):
        sym_w = sym_widths[idx]
        is_ws = ch.isspace() and ch != "\xa0"

        # Symbols wider than the line itself are dropped — ratatui
        # does the same; rendering one would blow past the modal
        # border without us being able to fold it.
        if sym_w > max_width:
            continue

        word_found = non_whitespace_previous and is_ws
        trimmed_overflow = (
            not pending_line
            and trim
            and word_width + sym_w > max_width
        )
        whitespace_overflow = (
            not pending_line
            and trim
            and whitespace_width + sym_w > max_width
        )
        untrimmed_overflow = (
            not pending_line
            and not trim
            and word_width + whitespace_width + sym_w > max_width
        )

        if word_found or trimmed_overflow or whitespace_overflow or untrimmed_overflow:
            if pending_line or not trim:
                while pending_ws:
                    pending_line.append(pending_ws.popleft()[0])
                line_width += whitespace_width
            pending_line.extend(pending_word)
            line_width += word_width

            pending_ws.clear()
            whitespace_width = 0
            pending_word = []
            word_width = 0

        line_full = line_width >= max_width
        pending_word_overflow = (
            sym_w > 0
            and line_width + whitespace_width + word_width >= max_width
        )

        if line_full or pending_word_overflow:
            remaining = max_width - line_width
            wrapped.append(pending_line)
            pending_line = []
            line_width = 0

            # Trim trailing whitespace that fits in the remainder of
            # the just-emitted row — ratatui drops it so a wrapped
            # line doesn't start with a stray space.
            while pending_ws:
                _, ws_w = pending_ws[0]
                if ws_w > remaining:
                    break
                whitespace_width -= ws_w
                remaining -= ws_w
                pending_ws.popleft()

            if is_ws and not pending_ws:
                continue

        if is_ws:
            whitespace_width += sym_w
            pending_ws.append((idx, sym_w))
        else:
            word_width += sym_w
            pending_word.append(idx)

        non_whitespace_previous = not is_ws

    if not pending_line and not pending_word and pending_ws and trim:
        wrapped.append([])

    if pending_line or not trim:
        while pending_ws:
            pending_line.append(pending_ws.popleft()[0])

    pending_line.extend(pending_word)

    if pending_line:
        wrapped.append(pending_line)

    if not wrapped:
        wrapped.append([])

    return wrapped


def _rendered_rows_for(line: str, width: int) -> int:
    """How many rendered rows the *line* will occupy when soft-wrapped
    at *width* cells under ratatui's word-wrap (``trim=False``).

    Backed by :func:`_word_wrap_indices` so the answer matches what
    pyratatui's ``Paragraph.wrap(True, False)`` actually paints —
    including the case where a long word can't share a row with
    leading content (forcing the previous row to commit half-empty)
    and the case where wide Unicode contributes two cells per
    grapheme. Always at least 1 (an empty line still occupies a
    rendered row).

    Hard tabs are still counted as zero cells (they're category
    ``Cc`` per :func:`_display_cell_width`); the renderer expands
    them on its side, but we don't see TTY-tab expansion in log
    lines so the inaccuracy doesn't surface in practice."""
    if width <= 0:
        return 1
    rows = _word_wrap_indices(line, width, trim=False)
    return max(1, len(rows))


def _total_rendered_rows(lines: Iterable[str], width: int) -> int:
    """Sum of :func:`_rendered_rows_for` over *lines*.

    Used by :meth:`RunDetailScreen._total_rendered_rows` to clamp the
    scroll offset to valid territory and by
    :func:`_compute_log_window` to find the visible suffix when the
    operator is parked off the tail."""
    return sum(_rendered_rows_for(line, width) for line in lines)


def _compute_log_window(
    lines: List[str], width: int, height: int,
    scroll_offset: int, follow: bool,
) -> Tuple[List[str], int]:
    """Pick the visible slice of *lines* and the intra-line scroll
    offset that the renderer should hand to ``Paragraph.scroll``.

    Returns ``(visible_lines, scroll_y)`` where:

    * ``visible_lines`` is a contiguous sub-sequence of *lines* whose
      rendered rows (after wrapping at *width*) intersect the visible
      window. With wrap enabled, a single long line can occupy
      multiple rendered rows, so the slice may be shorter than
      *height*.
    * ``scroll_y`` is the number of rendered rows to skip from the
      top of the wrapped output. This is non-zero whenever the first
      visible logical line wraps and only its *later* rendered rows
      should be on screen — the case the previous logical-line
      slicing missed, which made long lines unreachable past the
      first screenful.

    Behaviour:

    * ``follow=True`` keeps the *last* ``height`` rendered rows
      visible, ignoring ``scroll_offset``.
    * ``follow=False`` interprets ``scroll_offset`` as the number of
      rendered rows back from the tail; the visible window ends
      ``scroll_offset`` rows before the tail.

    All width/height/offset inputs are sanitized so a degenerate
    geometry (zero width / zero height / negative offset) returns a
    safe fallback rather than dividing by zero."""
    if not lines:
        return [], 0
    width = max(1, width)
    height = max(1, height)

    row_counts = [_rendered_rows_for(line, width) for line in lines]
    total = sum(row_counts)
    if total <= 0:
        return [], 0

    if follow:
        target_top = max(0, total - height)
    else:
        offset = max(0, min(scroll_offset, max(0, total - 1)))
        target_top = max(0, total - height - offset)
    target_bottom = target_top + height

    # Walk forward to find the first logical line whose rendered
    # range intersects the target window, then keep walking until we
    # cover ``target_bottom``.
    cum = 0
    first_idx = None
    first_start = 0
    for idx, n_rows in enumerate(row_counts):
        if cum + n_rows > target_top:
            first_idx = idx
            first_start = cum
            break
        cum += n_rows
    if first_idx is None:
        return [], 0

    cum2 = first_start
    last_idx = first_idx
    for idx in range(first_idx, len(lines)):
        cum2 += row_counts[idx]
        last_idx = idx
        if cum2 >= target_bottom:
            break

    visible = lines[first_idx:last_idx + 1]
    scroll_y = max(0, target_top - first_start)
    return visible, scroll_y


def _wrap_aware_tail(
    lines: Iterable[str], width: int, height: int,
) -> List[str]:
    """Pick the suffix of ``lines`` that fits in ``height`` rendered
    rows when soft-wrapped at ``width`` cells.

    The agent transcript Paragraph is rendered with ``.wrap(True)``
    so long lines fold onto continuation rows. That means a buffer
    slice taken purely in *logical* lines (the way
    :meth:`RunDetailScreen.visible_lines` does) can exceed the panel
    height once wrapping is applied — and pyratatui clips overflow
    at the bottom border, which would hide the very latest output
    in follow mode. This helper trims from the *head* until the
    suffix fits, so the tail (newest content) stays painted.

    The math approximates "how many rendered rows will this line
    occupy" with ``ceil(len / width)`` (at least 1 row even for an
    empty line). It doesn't account for wide unicode glyphs or tabs,
    but the inaccuracy only causes us to drop one extra leading
    line in pathological cases — i.e. it errs on the side of "show
    fewer than expected" rather than "overflow the panel". The next
    render tick reflows from the live buffer either way."""
    seq = list(lines)
    if not seq:
        return []
    if width <= 0 or height <= 0:
        return seq[-max(1, height):]

    selected: List[str] = []
    used = 0
    for line in reversed(seq):
        rendered = max(1, -(-len(line) // width))
        if selected and used + rendered > height:
            break
        selected.insert(0, line)
        used += rendered
        if used >= height:
            break
    return selected


def _latest_notification_text(notifications: Iterable[Notification]) -> str:
    """Return the most recent notification's text, or ``""`` if none.

    The renderer parks the latest notification in the screen's footer
    so an operator sees feedback after pressing a verb without a
    pop-up window. Older notifications stay in the list (tests look
    for *all* of them) but are not rendered."""
    last: Optional[Notification] = None
    for n in notifications:
        last = n
    return last.text if last is not None else ""


# ── Public entry points used by ``cli`` ─────────────────────────────────────


def run_list_app(runs_dir: Path) -> int:
    """Launch the run-list TUI. Returns a CLI exit code.

    On a clean Enter-on-row selection the run-list TUI now exits and
    the *caller* is responsible for streaming the chosen run (so the
    local terminal's native scrollback owns navigation rather than
    the in-process pyratatui frame loop). For that handoff use
    :func:`run_list_app_with_selection` instead, which returns both
    the exit code and the selected ``RunPaths`` (or ``None`` when the
    operator quit without selecting anything).
    """
    rc, _selected = run_list_app_with_selection(runs_dir)
    return rc


def run_list_app_with_selection(
    runs_dir: Path,
) -> Tuple[int, Optional[RunPaths]]:
    """Launch the run-list TUI and report the selection (if any).

    Returns ``(exit_code, paths_or_None)``. ``paths_or_None`` is the
    ``RunPaths`` the operator selected via Enter on a row; ``None``
    when they quit (``q`` / Ctrl-C / Esc) without picking a run.

    The CLI uses this two-step shape so a selection cleanly tears
    down the alt-screen TUI and *then* drops into the streaming tail
    (:func:`auto_iterator.display.stream_log`). Doing the handoff in
    two phases — exit, then stream — means the streaming output goes
    to the regular screen buffer and the local terminal's native
    scrollback owns it from then on, which is the whole point of the
    high-latency-SSH redesign."""
    app = RunListApp(runs_dir)
    rc = app.run()
    return rc, app.streamed_run


def run_detail_app(
    paths: RunPaths,
    *,
    refresh_seconds: float = 0.5,
    initial_log_lines: Optional[int] = 30,
) -> int:
    """Launch the per-run detail TUI. Returns a CLI exit code.

    ``initial_log_lines=None`` seeds the screen with the entire
    existing agent log; an int caps the seed at that many trailing
    lines."""
    return RunDetailApp(
        paths,
        refresh_seconds=refresh_seconds,
        initial_log_lines=initial_log_lines,
    ).run()


__all__ = [
    "KeyEvent",
    "Notification",
    "RunDetailApp",
    "RunDetailScreen",
    "RunListApp",
    "RunListScreen",
    "run_detail_app",
    "run_list_app",
    "run_list_app_with_selection",
]
