"""Textual TUI for ``ai`` and ``ai show``.

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
  verdict / paused) and a scrollable agent-output panel filling the
  rest of the screen, backed by :class:`LogTailer`. The verbose
  status block, the prompt preview and the structured events stream
  are intentionally absent here: pressing Enter on a run is the
  "watch the agent work" view, and surfacing them would compete with
  the raw transcript for screen space. Operators who need them can
  drop back to ``ai show <run_id> --once`` or ``ai events``.

Design rules baked in:

* **TUI never owns runner lifecycles.** Quitting the app does not
  signal anything. The only way the TUI signals a runner pid is when
  the operator presses ``k`` (kill) — and even then we route through
  :func:`actions.signal_runner` so the same code path the CLI uses
  is exercised, no shortcut.
* **Polling, not inotify.** Each screen owns its own ``set_interval``
  timers (≈1 s for the run list, ≈0.5 s for the detail status bar,
  ≈0.2 s for the agent log). No background thread, no subscribed
  file watcher.
* **Lazy import.** This module is imported by :func:`cli.cmd_show`
  only on the TTY default path so plain ``ai ls`` doesn't pay the
  Textual startup cost.

The TUI is *not* the source of truth for any state; every refresh
re-reads ``meta.json`` / ``state.json`` / ``events.jsonl`` /
``agent.log``. If the file goes away (worktree removed, run cleaned
up), the screens degrade gracefully — they show a placeholder rather
than crashing.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Optional

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.coordinate import Coordinate
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Static,
)

from . import actions
from .display import LogTailer, tail_text_file
from .ls import RunRow, list_runs, reconcile_status
from .meta import read_meta
from .run_dir import RunPaths, read_json


# Upper bound on how many lines of the existing agent transcript the
# detail screen will materialise into the ``RichLog`` widget on open.
#
# The widget keeps every appended line as a pre-wrapped ``Strip`` in
# memory and re-renders the visible region on every scroll/refresh,
# so an unbounded buffer turns into the dominant scrolling cost over
# a high-latency SSH link: a chatty agent run can produce a multi-MiB
# transcript with tens of thousands of lines, and Textual's frame
# diff for a viewport over that many strips is large enough to lag
# noticeably even on a ~100 ms VPS round-trip.
#
# Capping at 10 000 lines keeps the steady-state widget cheap (more
# than a typical operator scrolls back) while still being well above
# the 200-line ``test_pressing_enter_seeds_full_agent_log`` floor and
# any real "see what happened recently" need. Live appends past the
# cap continue to land via :class:`textual.widgets.RichLog`'s built-in
# ``max_lines`` ring-buffer.
_FULL_LOG_SEED_CAP = 10_000


# ── Helpers shared by both screens ──────────────────────────────────────────


def _strip_ansi(text: str) -> str:
    """Drop ANSI escape codes — the existing renderers emit them, but
    Textual widgets render Rich markup, not raw escapes. Without this
    pass the status panel ends up showing literal ``\\x1b[1m`` bytes."""
    import re

    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)


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
    """One :class:`RunRow` rendered as the cells of a DataTable row.

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


# ── Modals ──────────────────────────────────────────────────────────────────


class _PromptModal(ModalScreen[Optional[str]]):
    """Single-input modal: prompt + ``Input`` + Submit/Cancel buttons.

    Returns the entered text via the ``ModalScreen`` result channel
    (``Optional[str]``: ``None`` on cancel, the text on submit). The
    submit button and ``Enter`` on the input both fire the same path
    so keyboard-only operators can hit the action without reaching
    for the mouse."""

    DEFAULT_CSS = """
    _PromptModal {
        align: center middle;
    }
    _PromptModal > Vertical {
        width: 80%;
        max-width: 100;
        height: auto;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    _PromptModal Label.title {
        padding-bottom: 1;
        text-style: bold;
    }
    _PromptModal Label.hint {
        color: $text-muted;
        padding-bottom: 1;
    }
    _PromptModal Horizontal {
        height: auto;
        align: right middle;
        padding-top: 1;
    }
    _PromptModal Button {
        margin-left: 2;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        *,
        title: str,
        hint: str = "",
        placeholder: str = "",
        initial: str = "",
    ) -> None:
        super().__init__()
        self._title = title
        self._hint = hint
        self._placeholder = placeholder
        self._initial = initial

    def compose(self) -> ComposeResult:
        with Vertical():
            # ``markup=False`` so help hints with literal ``[``/``]``
            # (e.g. the rewind ``[review|fix|after_impl]`` tag) are
            # rendered verbatim instead of failing Rich's markup parse.
            yield Label(self._title, classes="title", markup=False)
            if self._hint:
                yield Label(self._hint, classes="hint", markup=False)
            yield Input(value=self._initial, placeholder=self._placeholder, id="entry")
            with Horizontal():
                yield Button("Cancel", id="cancel", variant="default")
                yield Button("Submit", id="submit", variant="primary")

    def on_mount(self) -> None:
        # Focus the input so typing lands directly without an extra Tab.
        self.query_one("#entry", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            self.dismiss(self.query_one("#entry", Input).value)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class _BackendChoiceModal(ModalScreen[Optional[dict]]):
    """Backend / preset picker shown as the third step of "New run".

    Shipped with two recommended layouts:

    1. **Cursor — Opus impl + GPT reviewer.** Single-backend Cursor
       run; ``cursor.py``'s defaults already give the canonical
       Claude-Opus implementer / GPT-5.5 reviewer pairing.
    2. **Claude Code + Codex — mixed.** Claude Code as
       implementer/fixer, Codex as the fresh-eyes reviewer. Maps to
       ``ai run --backend claude-code --reviewer-backend codex``.

    Both presets pass ``ignore_env_overrides=True`` to
    ``default_run_config`` so the runner gets exactly the layout the
    operator picked: a stray ``AGENT_REVIEWER_BACKEND`` / ``AGENT_CMD``
    in the surrounding shell cannot silently rewrite a "Cursor" pick
    into a mixed Claude/Codex run, or vice-versa. Operators who want
    env-driven backend resolution should use ``ai run`` from the
    shell — the TUI's ``n`` verb is intentionally opinionated about
    which two layouts it surfaces.

    Returns the kwargs dict to forward to
    :func:`auto_iterator.actions.default_run_config`, or ``None`` on
    cancel. We deliberately surface only the canonical layouts here:
    full per-phase control still lives behind the ``--{phase}-backend``
    flags on ``ai run`` and the matching env vars; cramming six radio
    buttons into a TUI modal would compete with argparse's ergonomics
    rather than complement them.
    """

    DEFAULT_CSS = """
    _BackendChoiceModal {
        align: center middle;
    }
    _BackendChoiceModal > Vertical {
        width: 90%;
        max-width: 110;
        height: auto;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    _BackendChoiceModal Label.title {
        padding-bottom: 1;
        text-style: bold;
    }
    _BackendChoiceModal Label.hint {
        color: $text-muted;
        padding-bottom: 1;
    }
    _BackendChoiceModal Label.preset-summary {
        color: $text-muted;
        padding: 0 0 1 4;
    }
    _BackendChoiceModal Button.preset {
        width: 100%;
        margin-bottom: 0;
    }
    _BackendChoiceModal Horizontal#actions {
        height: auto;
        align: right middle;
        padding-top: 1;
    }
    _BackendChoiceModal Button.action {
        margin-left: 2;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("1", "pick_one", "Cursor", show=False),
        Binding("2", "pick_two", "Claude+Codex", show=False),
    ]

    # Order matters: the first preset is the recommended default and
    # gets the primary-styled button (focused on mount so Enter starts
    # immediately). Each entry's ``kwargs`` is forwarded verbatim to
    # ``actions.default_run_config``.
    #
    # Both presets pass ``ignore_env_overrides=True`` so the layout
    # the operator sees in the modal is exactly the layout the runner
    # gets — even when the surrounding shell exports ``AGENT_CMD`` /
    # ``AGENT_*_BACKEND`` / ``AGENT_*_CMD``. Without that flag a
    # hostile env (e.g. a stale ``AGENT_REVIEWER_BACKEND=codex`` left
    # over from a previous mixed run) would silently rewrite the
    # Cursor preset into a mixed run, violating the "you can see
    # which backend will run" contract.
    PRESETS: tuple[dict, ...] = (
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

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("New run · backend", classes="title", markup=False)
            yield Label(
                "Pick the backend layout for this run. "
                "Click a preset, or press 1 / 2.",
                classes="hint",
                markup=False,
            )
            for preset in self.PRESETS:
                yield Button(
                    preset["label"],
                    id=f"preset-{preset['id']}",
                    classes="preset",
                    variant=("primary" if preset["id"] == "cursor" else "default"),
                )
                yield Label(
                    preset["summary"],
                    classes="preset-summary",
                    markup=False,
                )
            with Horizontal(id="actions"):
                yield Button(
                    "Cancel", id="cancel", variant="default", classes="action",
                )

    def on_mount(self) -> None:
        # Focus the recommended preset so an operator who pressed ``n``
        # and just wants the default flow only has to hit Enter once
        # more. Falling back silently if the lookup fails keeps the
        # modal usable even if a future refactor renames the id.
        try:
            first = self.PRESETS[0]["id"]
            self.query_one(f"#preset-{first}", Button).focus()
        except Exception:
            pass

    def _kwargs_for(self, preset_id: str) -> Optional[dict]:
        for preset in self.PRESETS:
            if preset["id"] == preset_id:
                return dict(preset["kwargs"])
        return None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "cancel":
            self.dismiss(None)
            return
        if bid.startswith("preset-"):
            kwargs = self._kwargs_for(bid[len("preset-"):])
            self.dismiss(kwargs)

    def action_pick_one(self) -> None:
        self.dismiss(self._kwargs_for(self.PRESETS[0]["id"]))

    def action_pick_two(self) -> None:
        self.dismiss(self._kwargs_for(self.PRESETS[1]["id"]))

    def action_cancel(self) -> None:
        self.dismiss(None)


class _ConfirmModal(ModalScreen[bool]):
    """Yes/no confirmation for destructive verbs (kill, restart, revert).

    The TUI's destructive-action confirmation mirrors the CLI's
    ``_maybe_confirm`` flow: the operator picked the run from a list
    rather than typing its id, so we want one extra keystroke between
    "selected the wrong row" and "killed the runner"."""

    DEFAULT_CSS = """
    _ConfirmModal {
        align: center middle;
    }
    _ConfirmModal > Vertical {
        width: 60%;
        max-width: 80;
        height: auto;
        padding: 1 2;
        background: $panel;
        border: thick $error;
    }
    _ConfirmModal Label.title {
        padding-bottom: 1;
        text-style: bold;
    }
    _ConfirmModal Horizontal {
        height: auto;
        align: right middle;
        padding-top: 1;
    }
    _ConfirmModal Button {
        margin-left: 2;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("y", "confirm", "Confirm", show=False),
        Binding("n", "cancel", "Cancel", show=False),
    ]

    def __init__(self, *, title: str, body: str = "") -> None:
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._title, classes="title", markup=False)
            if self._body:
                yield Label(self._body, markup=False)
            with Horizontal():
                yield Button("No", id="no", variant="default")
                yield Button("Yes", id="yes", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


# ── RunListScreen ────────────────────────────────────────────────────────────


class RunListScreen(Screen):
    """Live run table. The home screen of the bare ``ai`` command.

    Columns mirror ``ai ls``. ``Enter`` opens the per-run detail
    screen; the operator verbs are bound to single keys (see
    ``BINDINGS``) and each one resolves the row currently under the
    cursor before opening its modal.

    The poll interval (1 s) is intentionally slow: ``list_runs``
    iterates every run-dir and reconciles its status, which is cheap
    but not free. A faster refresh wouldn't change what an operator
    can do — runs don't transition that quickly."""

    DEFAULT_CSS = """
    RunListScreen DataTable {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("enter", "open_selected", "Open"),
        Binding("n", "new_run", "New"),
        Binding("s", "send", "Send"),
        Binding("p", "pause", "Pause"),
        Binding("r", "resume", "Resume"),
        Binding("k", "kill", "Kill"),
        Binding("R", "restart", "Restart"),
        Binding("w", "rewind", "Rewind"),
        Binding("a", "apply", "Apply"),
        Binding("v", "revert", "Revert"),
        Binding("d", "diff", "Diff"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, runs_dir: Path) -> None:
        super().__init__()
        self.runs_dir = runs_dir
        self._table: Optional[DataTable] = None
        self._rows_by_key: dict[str, RunRow] = {}
        # Last cell-tuple we pushed into each row, indexed by run_id.
        # The 1 Hz ``refresh_rows`` poll uses this to skip the table
        # rebuild whenever the set + order of runs hasn't changed —
        # only the cells that actually differ get re-painted via
        # ``update_cell_at``. The pre-existing ``clear`` +
        # ``add_row``-per-run path was visibly costly to the operator
        # over a high-latency SSH link because every tick repainted
        # the entire table, even when nothing on disk had changed.
        self._row_cells_by_key: dict[str, tuple[str, ...]] = {}
        # Insertion order of run_ids in the DataTable, mirrored from
        # ``list_runs`` (started_at desc). Compared against the fresh
        # ordering to decide whether a structural rebuild (rows
        # added/removed/reordered) is needed or whether per-cell
        # diffing suffices.
        self._row_order: list[str] = []

    def compose(self) -> ComposeResult:
        # ``show_clock=False``: Textual's Header repaints the clock
        # cell every second when ``show_clock=True``, which over an
        # SSH link is one extra ANSI burst per second forever. The
        # operator already has a wall clock somewhere; the extra
        # chrome isn't worth the steady-state bandwidth.
        yield Header(show_clock=False)
        table = DataTable(zebra_stripes=True, cursor_type="row", id="run-table")
        table.add_columns(
            "RUN_ID", "STATUS", "PHASE", "O/I", "VERDICT", "UPDATED", "PROMPT",
        )
        self._table = table
        yield table
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_rows()
        # 1 s poll keeps the table fresh without saturating I/O on a
        # large runs dir. Kept on a Textual ``set_interval`` so it
        # tears down with the screen — no orphan threads.
        self.set_interval(1.0, self.refresh_rows)

    # ── data plumbing ──

    def refresh_rows(self) -> None:
        """Re-read ``list_runs`` and reconcile the DataTable diff.

        Two paths share one source of truth:

        * **Fast path** — when the set + order of run IDs is
          unchanged from the previous tick, walk the rows and only
          ``update_cell_at`` the cells whose value actually differs.
          This is the common case once the operator has been parked
          on the screen for a few seconds: ``list_runs`` is sorted by
          ``started_at`` desc, so only the mutating cells (status,
          phase, outer/inner, verdict, updated-at) change between
          ticks. Skipping the rebuild for a stable run-set keeps the
          per-tick repaint proportional to *changes*, not *rows*,
          which is the difference between snappy and visibly laggy
          scrolling on a ~100 ms VPS round-trip.
        * **Slow path** — when a run appears, disappears, or the
          ordering shifts, fall back to ``clear`` + ``add_row``-per-
          run. Diffing arbitrary reorders is more bug-prone than the
          gain it would buy on the rare structural change.

        Either way the cursor is pinned back onto the same run_id
        when possible so a refreshing tick never yanks focus."""
        if self._table is None:
            return
        try:
            rows = list_runs(self.runs_dir)
        except OSError:
            rows = []
        self._rows_by_key = {r.run_id: r for r in rows}
        new_order = [r.run_id for r in rows]
        new_cells_by_key = {r.run_id: _row_cells(r) for r in rows}

        prev_run_id: Optional[str] = None
        try:
            cursor_row = self._table.cursor_row
            if 0 <= cursor_row < self._table.row_count:
                prev_run_id = self._table.get_row_at(cursor_row)[0]
        except Exception:
            prev_run_id = None

        # Coalesce every mutation that follows into a single repaint
        # via ``app.batch_update``. ``DataTable.update_cell`` and
        # ``add_row`` each call ``self.refresh`` on the table, so
        # without batching a 20-row diff that touches 4 cells per
        # row would queue 80 refresh requests in one tick. Over a
        # high-latency SSH link those refreshes turn into terminal
        # bytes that compete with the operator's keyboard / scroll
        # input — exactly the symptom the user reports.
        with self.app.batch_update():
            if new_order == self._row_order:
                for row_idx, run_id in enumerate(new_order):
                    new_cells = new_cells_by_key[run_id]
                    old_cells = self._row_cells_by_key.get(run_id)
                    if old_cells == new_cells:
                        continue
                    # Only paint the cells that actually differ. A
                    # typical tick mutates status / phase /
                    # outer-inner / updated — three or four cells out
                    # of seven — so this skips the no-op writes that
                    # would otherwise force a full row repaint.
                    if old_cells is None or len(old_cells) != len(new_cells):
                        differing = range(len(new_cells))
                    else:
                        differing = [
                            i for i, (old, new) in enumerate(
                                zip(old_cells, new_cells)
                            ) if old != new
                        ]
                    for col_idx in differing:
                        try:
                            self._table.update_cell_at(
                                Coordinate(row_idx, col_idx),
                                new_cells[col_idx],
                            )
                        except Exception:
                            # Coordinate drifted (e.g. a concurrent
                            # structural change we didn't expect):
                            # bail to the slow path on the next tick
                            # by forgetting our cached order.
                            self._row_order = []
                            self._row_cells_by_key = {}
                            break
                    else:
                        self._row_cells_by_key[run_id] = new_cells
                        continue
                    break
                else:
                    return
            # Slow path: structural change (or fast-path bailed) —
            # rebuild the table from scratch, then re-cache for the
            # next tick.
            self._table.clear()
            for row in rows:
                self._table.add_row(*_row_cells(row), key=row.run_id)
            self._row_order = list(new_order)
            self._row_cells_by_key = dict(new_cells_by_key)
            if prev_run_id is not None and prev_run_id in self._rows_by_key:
                try:
                    idx = new_order.index(prev_run_id)
                    self._table.move_cursor(row=idx)
                except (ValueError, IndexError):
                    pass

    def _selected_run(self) -> Optional[RunRow]:
        """Return the row currently under the cursor, or ``None`` if empty."""
        if self._table is None or self._table.row_count == 0:
            return None
        try:
            cursor_row = self._table.cursor_row
            row_key = self._table.get_row_at(cursor_row)[0]
        except Exception:
            return None
        return self._rows_by_key.get(row_key)

    def _selected_paths(self) -> Optional[RunPaths]:
        sel = self._selected_run()
        if sel is None:
            return None
        return RunPaths(runs_dir=self.runs_dir, run_id=sel.run_id)

    # ── action handlers ──
    #
    # Every verb routes through one of the helpers in :mod:`actions` so
    # the TUI never re-implements the protocol. The handlers below are
    # thin glue: collect input via a modal, drop the file, refresh.

    def action_open_selected(self) -> None:
        sel = self._selected_run()
        if sel is None:
            self.notify("(no run selected)", severity="warning")
            return
        paths = RunPaths(runs_dir=self.runs_dir, run_id=sel.run_id)
        # Pressing Enter on a row is the "watch the agent work" gesture;
        # the operator explicitly asked for the *full* raw transcript
        # rather than a bounded tail. ``initial_log_lines=None`` tells
        # the detail screen to seed the panel with the whole agent log
        # before parking the tailer at EOF, so older lines stay
        # scrollable. The standalone ``ai show`` path still honors
        # ``--lines`` because it instantiates the screen explicitly.
        self.app.push_screen(RunDetailScreen(paths, initial_log_lines=None))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Mouse-double-click / Enter on a focused DataTable row.

        DataTable emits this in addition to firing the row's
        ``Enter`` binding, so the Screen-level ``open_selected``
        binding might never fire when focus is in the table. Hook the
        event explicitly so the open path works for both keyboard
        and mouse operators."""
        # The action already resolves the cursor row, so we don't need
        # to thread the event payload through.
        event.stop()
        self.action_open_selected()

    def action_new_run(self) -> None:
        # Spawning a new run from the TUI is a three-step modal flow:
        # prompt → workspace → backend preset. Advanced configuration
        # (model overrides, max_outer/inner) is left to ``ai run`` from
        # the shell because cramming it into a TUI form would compete
        # with argparse's ergonomics rather than complement them.
        #
        # The backend step is the only place this front-end edits
        # backend selection: the picker exposes the two recommended
        # layouts (Cursor with Opus+GPT, or Claude Code + Codex) and
        # nothing else. Operators who want env-driven backend
        # resolution use ``ai run`` from the shell. Either way, the
        # final cfg is built by :func:`actions.default_run_config` so
        # ``ai run`` and the TUI's ``n`` verb produce byte-identical
        # ``RunConfig``s for the same inputs.
        def on_prompt(text: Optional[str]) -> None:
            if not text or not text.strip():
                return

            def on_workspace(ws: Optional[str]) -> None:
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
                        # Surface the resolved backend layout so the
                        # operator can confirm the picker worked —
                        # silent acceptance is too easy to miss when
                        # presets and env vars interact.
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

                self.app.push_screen(_BackendChoiceModal(), on_backend)

            self.app.push_screen(
                _PromptModal(
                    title="New run · workspace",
                    hint="Path to the source workspace (the agent's cwd).",
                    placeholder=str(Path.cwd()),
                    initial=str(Path.cwd()),
                ),
                on_workspace,
            )

        self.app.push_screen(
            _PromptModal(
                title="New run · prompt",
                hint="The task description for the agent.",
                placeholder="Implement feature X carefully.",
            ),
            on_prompt,
        )

    def _ensure_runner_alive(self, paths: RunPaths) -> bool:
        """Mirror the CLI's mutation liveness gate before a control-file write.

        ``cli._drop_mutation`` refuses to drop ``guidance.txt`` /
        ``rewind.json`` / ``prompt.txt`` for a runner whose meta marks
        it ``killed`` / ``crashed`` / ``exited`` or whose pid is no
        longer alive. Without the same check, the TUI would silently
        leave stale control files behind for runs the CLI rejects —
        the protocol is filesystem-as-state, so the two front-ends
        must apply the same gate. We fetch fresh meta on each call so
        a runner that exited *while the operator was typing in the
        modal* is still rejected at the moment of submit."""
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

        def on_text(text: Optional[str]) -> None:
            if text is None or not text.strip():
                return
            # Re-check liveness *after* the modal closes so a runner
            # that exited while the operator was typing still gets
            # rejected, matching ``_drop_mutation`` in the CLI.
            if not self._ensure_runner_alive(paths):
                return
            try:
                actions.write_guidance(paths, text.strip())
            except OSError as exc:
                self.notify(f"send failed: {exc}", severity="error")
                return
            self.notify(f"guidance queued for {paths.run_id}",
                        severity="information")

        self.app.push_screen(
            _PromptModal(
                title=f"Send guidance · {paths.run_id}",
                hint="Text steered into the next review prompt.",
                placeholder="Focus on the failing test in foo_test.py",
            ),
            on_text,
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

        self.app.push_screen(
            _ConfirmModal(
                title=f"Kill run {paths.run_id}?",
                body="Sends SIGTERM (then SIGKILL after 5 s) to the runner.",
            ),
            on_confirm,
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
                from .run_dir import read_json
                from .runner import spec_to_cfg

                spec = read_json(paths.spec)
                cfg = spec_to_cfg(spec)
            except (OSError, KeyError, ValueError) as exc:
                self.notify(f"restart failed: cannot read spec: {exc}",
                            severity="error")
                return
            # Kill the old runner first so it can't race the new one.
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
                    f"restarted as {result.run_id}", severity="information",
                )
                self.refresh_rows()
            else:
                self.notify(
                    f"restart failed: {result.message}", severity="error",
                )

        self.app.push_screen(
            _ConfirmModal(
                title=f"Restart run {run_id}?",
                body="Old runner is killed; a fresh one spawns from spec.json.",
            ),
            on_confirm,
        )

    def action_rewind(self) -> None:
        paths = self._selected_paths()
        if paths is None:
            self.notify("(no run selected)", severity="warning")
            return

        def on_text(text: Optional[str]) -> None:
            if text is None or not text.strip():
                return
            # Same liveness gate as the CLI's ``_drop_mutation``: a
            # ``rewind.json`` written for a dead runner just sits there
            # forever. Reject before writing so the on-disk state stays
            # consistent with what ``ai rewind`` would have done.
            if not self._ensure_runner_alive(paths):
                return
            try:
                actions.write_rewind_from_to_string(paths, text.strip())
            except (OSError, ValueError) as exc:
                self.notify(f"rewind failed: {exc}", severity="error")
                return
            self.notify(f"rewind queued for {paths.run_id}",
                        severity="information")

        self.app.push_screen(
            _PromptModal(
                title=f"Rewind · {paths.run_id}",
                hint="Format: outer=N,inner=M[,phase=review|fix|after_impl]",
                placeholder="outer=1,inner=1,phase=review",
            ),
            on_text,
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

        self.app.push_screen(
            _ConfirmModal(
                title=f"Apply worktree changes for {paths.run_id}?",
                body="Applies the run's diff to the source workspace.",
            ),
            on_confirm,
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

        self.app.push_screen(
            _ConfirmModal(
                title=f"Revert applied changes for {paths.run_id}?",
                body="Reverses a previous apply against the source workspace.",
            ),
            on_confirm,
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
        self.app.push_screen(_DiffViewer(title=f"Diff · {paths.run_id}", body=body))

    def action_quit(self) -> None:
        self.app.exit()


class _DiffViewer(ModalScreen[None]):
    """Plain-text scrollable viewer for the diff/status preview."""

    DEFAULT_CSS = """
    _DiffViewer {
        align: center middle;
    }
    _DiffViewer > Vertical {
        width: 90%;
        height: 80%;
        background: $panel;
        border: thick $accent;
    }
    _DiffViewer Label {
        padding: 0 2;
        text-style: bold;
    }
    _DiffViewer VerticalScroll {
        height: 1fr;
        padding: 1 2;
    }
    _DiffViewer Horizontal {
        height: auto;
        align: right middle;
        padding: 0 2 1 2;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
    ]

    def __init__(self, *, title: str, body: str) -> None:
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._title)
            with VerticalScroll():
                yield Static(self._body, expand=True)
            with Horizontal():
                yield Button("Close", id="close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close":
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


# ── RunDetailScreen ─────────────────────────────────────────────────────────


class _WrapAwareRichLog(RichLog):
    """A :class:`RichLog` that re-flows its contents on a width change.

    Vanilla ``RichLog`` is built for performance: each ``write`` call
    renders the content into a list of pre-wrapped :class:`Strip`
    objects at *the width that was current at write time*, and from
    then on those strips are baked. ``Strip.from_lines`` is never
    re-run on the historical content. That trade-off is great for
    steady-state scrolling — every visible row is just an indexed
    fetch — but it has two visible failure modes when the terminal
    is resized:

    1. **Stale wrap.** Strips written at the old width still show up
       wrapped at that old width even after the viewport changes.
       The transcript looks "frozen" in its previous geometry.
    2. **Spurious horizontal scrollbar.** When the terminal shrinks
       horizontally, the existing strips become wider than
       ``scrollable_content_region.width``. Textual responds by
       enabling a horizontal scrollbar (the "blue bar at the bottom"
       the operator reports), because as far as the widget knows
       the *content* is too wide to fit.

    Fix: keep a parallel bounded :class:`deque` of the raw text lines
    that have been written through :meth:`write_line`, and on a real
    width change ``clear`` the widget and replay the deque so each
    historical line gets re-rendered at the new width.

    Why not override ``write`` itself instead of adding a sibling?
    Because :meth:`RichLog.on_resize` re-issues every entry from
    ``_deferred_renders`` via ``self.write`` once the widget's size
    is known for the first time. Mirroring inside ``write`` would
    double-count those entries (the screen had already mirrored
    them before deferral). A separate ``write_line`` keeps the raw
    mirror unambiguously the screen's responsibility, while
    ``write`` and the deferred-render replay path stay untouched.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Bounded mirror of every raw text line the screen has handed
        # us. Same ring size as the widget's strip cap so the buffer
        # we keep around for re-wrap is no fatter than what the
        # widget is already willing to display: under heavy-write
        # conditions the oldest entries fall off both ends in
        # lockstep and we don't surprise the operator with content
        # the widget has already aged out.
        cap = self.max_lines if self.max_lines is not None else 10_000
        self._raw_lines: Deque[str] = deque(maxlen=cap)
        # Last viewport width we successfully re-flowed at. Compared
        # against fresh ``Resize`` events to debounce away the
        # ``height changed but width didn't`` events Textual fires
        # liberally during layout settling.
        self._last_wrap_width: int = 0

    def write_line(
        self, line: str, *, scroll_end: Optional[bool] = None,
    ) -> None:
        """Append *line* to the log, mirroring it for resize replay.

        Callers MUST go through this helper rather than
        :meth:`RichLog.write` for any content they want preserved
        across a re-flow. Direct ``write`` calls (and the deferred-
        render replay inside Textual's :meth:`RichLog.on_resize`)
        intentionally bypass the mirror so we don't double-count.

        ``scroll_end`` mirrors :meth:`RichLog.write`'s parameter and
        defaults to ``None`` so the widget's own ``auto_scroll`` flag
        decides — that's what the seed path (``_seed_initial_log``)
        relies on to land at the tail when deferred renders are
        replayed once the widget learns its size. The runtime burst
        path passes ``scroll_end=False`` explicitly to suppress the
        per-line scroll request and folds them into one
        ``scroll_end`` call after the burst.
        """
        self._raw_lines.append(line)
        self.write(line, scroll_end=scroll_end)

    def on_resize(self, event: events.Resize) -> None:
        # Let the base class drain any deferred renders first so the
        # widget reaches a consistent ``_size_known`` state before
        # we touch it.
        super().on_resize(event)

        new_width = event.size.width
        if new_width <= 0:
            return
        if new_width == self._last_wrap_width:
            return

        self._last_wrap_width = new_width

        if not self._raw_lines:
            return

        # Defer the re-flow until after the next refresh. This
        # matters because Textual's layout pass — the one that
        # finally sets ``scrollable_content_region`` to the new
        # viewport width — runs AFTER ``on_resize`` event handlers,
        # not during them. Re-writing inline at this point would
        # measure the renderable against the stale (pre-resize)
        # content region, which on the very first resize is
        # ``Region(0, 0, 0, 0)`` and wraps every line at Rich's
        # default 80-column fallback. ``call_after_refresh`` parks
        # the work on the queue immediately *after* the next layout
        # / refresh cycle, where ``scrollable_content_region`` has
        # the correct width and the wrap actually re-flows the
        # transcript at the new geometry. The visible cost is one
        # extra frame between the resize event and the reflowed
        # paint — over a high-ping link this is unmeasurable next
        # to the SIGWINCH round trip itself.
        self.call_after_refresh(self._reflow_raw_lines)

    def _reflow_raw_lines(self) -> None:
        """Re-render every mirrored raw line at the current width.

        Called from :meth:`on_resize` via ``call_after_refresh`` so
        that ``scrollable_content_region`` reports the new viewport
        width by the time we re-issue the writes."""
        if not self._raw_lines:
            return

        # Capture the follow state before we wipe the widget so an
        # operator who was parked at the tail stays at the tail
        # post-reflow (otherwise ``clear`` would leave us at scroll
        # position 0 and the next tick would interpret that as
        # "scrolled away").
        was_at_end = self.scroll_y >= self.max_scroll_y - 1
        snapshot = list(self._raw_lines)

        # ``clear`` only touches the widget's own state — strips,
        # line cache, deferred renders, virtual size — and leaves
        # ``self._raw_lines`` alone. Replay through ``self.write``
        # (not ``write_line``) so the mirror isn't re-appended on
        # top of itself.
        self.clear()
        with self.app.batch_update():
            for raw in snapshot:
                self.write(raw, scroll_end=False)

        if was_at_end:
            self.scroll_end(animate=False, immediate=True, x_axis=False)


class RunDetailScreen(Screen):
    """Per-run detail view: minimal status bar + full agent transcript.

    Layout (top → bottom):

    1. **Status bar** — a single line built by
       :func:`_minimal_status_line` summarizing the reconciled run
       state (status / phase / outer-inner / verdict / paused). It
       refreshes every ``refresh_seconds`` so an operator parked in
       the log still sees the run advance.
    2. **Agent log panel** — a ``RichLog`` driven by
       :class:`LogTailer`, expanded to fill the rest of the screen.
       Bytes that arrive between ticks are appended; if the operator
       scrolls up, ``auto_scroll`` is left off so they aren't yanked
       back to the tail. Pressing ``f`` toggles follow.

    The verbose status block, the prompt preview, and the structured
    events stream that the non-TTY ``ai show`` view emits are
    intentionally omitted: this screen exists to watch the agent's
    raw transcript scroll by. Operators who need the structured data
    can use ``ai show <run_id> --once`` (combined block) or
    ``ai events <run_id>`` from a separate terminal.

    The ``--refresh`` CLI flag tunes the status-bar timer; the agent
    log polls at a fixed 0.2 s because that's the rate at which a
    chatty agent's transcript becomes worth re-rendering."""

    DEFAULT_CSS = """
    RunDetailScreen #status-bar {
        height: 1;
        background: $primary;
        color: $text;
        padding: 0 1;
    }
    RunDetailScreen #log-panel {
        height: 1fr;
        border: none;
        /* Scrollbars disabled. Textual's scrollbar widget repaints
           the thumb on every scroll position change and *also*
           swaps to ``$scrollbar-color-hover`` whenever the mouse
           pointer crosses it — so on a high-ping SSH link, every
           wheel notch and every stray mouse move turns into an
           extra ANSI burst that competes with the actual content
           diff for terminal bandwidth. The ``g`` / ``G`` / ``j`` /
           ``k`` / ``f`` bindings already give keyboard operators a
           full set of scroll affordances; mouse-wheel scrolling
           still works on the panel even without a visible thumb.

           ``scrollbar-size-horizontal: 0`` is belt-and-suspenders
           against the "blue bar at the bottom on resize" symptom:
           even though :class:`_WrapAwareRichLog` re-flows the
           transcript on a width change, the replay isn't atomic
           with the resize event — for the brief window between
           Textual recomputing the viewport and our re-flow
           landing, the still-old strips can be wider than the
           new ``scrollable_content_region.width``, which is what
           was painting the horizontal scrollbar. The log viewer
           wraps; it is never meaningfully horizontally scrolled,
           so the bar has no operator value either way. */
        scrollbar-size-vertical: 0;
        scrollbar-size-horizontal: 0;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "back", "Back", show=False),
        Binding("j", "scroll_log_down", "↓", show=False),
        Binding("k", "scroll_log_up", "↑", show=False),
        Binding("g", "scroll_log_top", "Top"),
        Binding("G", "scroll_log_bottom", "Bottom"),
        Binding("f", "toggle_follow", "Follow"),
    ]

    def __init__(
        self,
        paths: RunPaths,
        *,
        refresh_seconds: float = 0.5,
        initial_log_lines: Optional[int] = 30,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.refresh_seconds = max(0.1, float(refresh_seconds))
        # ``None`` is the "seed the entire existing log" sentinel used
        # by the run-list Enter action (the operator explicitly asked
        # for the full raw transcript, not a bounded tail). An int
        # caps the seed at that many trailing lines, which is what the
        # CLI's ``ai show --lines`` path needs.
        self.initial_log_lines: Optional[int] = (
            None if initial_log_lines is None else max(1, int(initial_log_lines))
        )
        self._tailer = LogTailer(paths.agent_log)
        # ``_follow`` reflects the *current* desired auto-scroll state;
        # it is recomputed each refresh from the widget's actual scroll
        # position so mouse-wheel / PageUp / Home scrolling correctly
        # pins the viewport. ``_user_forced_follow_off`` is the manual
        # override toggled by pressing ``f`` — once the operator
        # explicitly disables follow we stop auto-syncing from the
        # viewport state, so a stray "near-the-bottom" tick doesn't
        # silently re-enable it.
        self._follow = True
        self._user_forced_follow_off = False
        self._status_widget: Optional[Static] = None
        self._log_widget: Optional[_WrapAwareRichLog] = None
        # Last string we pushed into the status bar. Compared against
        # the freshly-rendered text on every tick so the bar's
        # ``Static.update`` is skipped when nothing changed: the
        # render cycle that ``update`` triggers is otherwise paid
        # twice a second forever, and over a high-latency SSH link
        # that's ~120 redundant frame diffs per minute that compete
        # with the agent transcript for terminal bandwidth.
        self._last_status_text: Optional[str] = None

    def compose(self) -> ComposeResult:
        # No Header/Footer here: the operator asked for a minimal
        # "show me the logs" screen and a chrome-free layout maximizes
        # the row budget for the transcript on a 24-row terminal.
        # Bindings are still active globally so ``q``/``f``/``g``/``G``
        # work without a visible footer cheatsheet.
        self._status_widget = Static(
            "(loading status...)", id="status-bar", expand=True, markup=False,
        )
        yield self._status_widget
        # ``wrap=True`` so a long agent line (a 500-char tool-call
        # payload, a wide diff hunk, a stack trace with embedded paths)
        # folds onto the next visible row instead of being clipped at
        # the terminal's right edge. Without wrapping, operators on a
        # narrow terminal see truncated content with no horizontal
        # scroll affordance — they'd have to widen the window or drop
        # to ``ai show <run_id> --logs`` to read the rest. The detail
        # screen exists to *watch the agent work*, so readability of
        # the raw transcript trumps the per-line one-row budget that
        # the non-TTY ``ai show`` view (``_truncate_visible``) cares
        # about.
        #
        # ``max_lines`` bounds the in-memory ring of pre-wrapped
        # ``Strip`` objects RichLog keeps for the transcript. Without
        # the cap, a long-lived run accumulates one Strip per logical
        # line for every byte the agent ever wrote, and Textual has
        # to walk that list each render — over a ~100 ms SSH link to
        # a VPS this turns scrolling into a visibly laggy gesture.
        # The cap matches :data:`_FULL_LOG_SEED_CAP` so the seed and
        # the steady-state buffer share the same shape: oldest lines
        # roll off the top once the cap is reached, which is the
        # behaviour an operator already expects from a tail-style
        # log viewer.
        #
        # ``min_width=0`` overrides RichLog's 78-column floor so a
        # short line ("inner_started", "review_finished verdict=…")
        # stops being padded out to a 78-cell ``Strip``. With wrap
        # enabled and a viewport that's typically wider than the
        # content, the floor was inflating every short transcript
        # row by 4-5× the cells it actually needed: more cells per
        # strip = more bytes Textual has to diff per scroll = more
        # bandwidth burned on the SSH link. The widget still uses
        # ``shrink=True`` (RichLog's default) to fold long lines
        # into the viewport width, so wrapping behaviour is
        # unaffected.
        #
        # ``auto_scroll`` stays on so the seed path's deferred
        # writes (the screen mounts before the first ``Resize`` event,
        # so every ``write`` lands in :attr:`RichLog._deferred_renders`
        # and is replayed later) snap the viewport to the tail when
        # they're finally rendered. The runtime burst loop in
        # :meth:`_refresh_log` overrides this with ``scroll_end=False``
        # per ``write`` and does a single explicit ``scroll_end`` at
        # the end of the burst — that fold collapses the previous
        # "one scroll request per appended line" pattern into a
        # single repaint, which is the actual SSH-bandwidth win. The
        # widget-level ``auto_scroll`` is also kept in sync with
        # :attr:`_follow` each tick so an operator who scrolled away
        # sees ``auto_scroll`` flip off (asserted by the smoke
        # tests).
        log = _WrapAwareRichLog(
            id="log-panel",
            wrap=True,
            highlight=False,
            markup=False,
            auto_scroll=True,
            max_lines=_FULL_LOG_SEED_CAP,
            min_width=0,
        )
        self._log_widget = log
        yield log

    def on_mount(self) -> None:
        # Seed the log panel with the existing transcript (full file
        # when ``initial_log_lines is None``, bounded tail otherwise)
        # so the operator has context when the screen opens. After the
        # first call the ``LogTailer`` offset is at EOF, so subsequent
        # polls only surface new bytes.
        self._seed_initial_log()
        self._refresh_status()
        # Two periodic timers feed the screen — the status bar and
        # the agent log. Each timer tick wakes the event loop, runs
        # the handler, and (often) schedules a widget refresh whose
        # diff has to ship across the SSH link before the operator's
        # next keystroke / scroll wheel event can be acted on. Above
        # ~3 Hz combined, those timers start visibly competing with
        # user input on a high-latency link.
        #
        # We hold the status timer at the operator-facing
        # ``--refresh`` interval (default 0.5 s, override via the
        # CLI flag) but pace the log poll deliberately slower than
        # the previous 0.2 s default. 0.4 s is still well below the
        # threshold at which an agent transcript stops feeling
        # "live" — a chatty agent emits at most a few lines per
        # second, and bursts are still drained in a single tick via
        # :meth:`_refresh_log`'s ``app.batch_update`` block — but it
        # halves the timer-driven render budget the screen pays
        # while the operator is trying to scroll.
        self.set_interval(self.refresh_seconds, self._refresh_status)
        self.set_interval(0.4, self._refresh_log)

    # ── refresh handlers ──

    def _refresh_status(self) -> None:
        if self._status_widget is None:
            return
        try:
            text = _strip_ansi(_minimal_status_line(self.paths))
        except Exception as exc:
            text = f"(status unavailable: {exc})"
        # Skip the ``Static.update`` no-op when the rendered string is
        # identical to the last one we pushed. ``update`` always
        # triggers a refresh / repaint cycle, even when the new
        # ``Text`` it builds compares equal to the current widget
        # content, so a stable run (status / phase / outer-inner /
        # verdict / paused all unchanged) would otherwise burn one
        # redraw per tick forever.
        if text == self._last_status_text:
            return
        self._last_status_text = text
        self._status_widget.update(text)

    def _refresh_log(self) -> None:
        if self._log_widget is None:
            return
        new_lines = self._tailer.read_new_lines()
        if not new_lines:
            return
        # Sync ``_follow`` from the widget's actual viewport position
        # *before* appending. ``RichLog``'s built-in scrolling
        # (mouse wheel, PageUp/PageDown, Home/End, arrow keys via the
        # widget's own bindings) doesn't route through our custom
        # ``j``/``k``/``g``/``G`` actions, so the only source of truth
        # for "is the operator currently at the bottom?" is the
        # widget's scroll geometry. Doing the sync here means a
        # mouse-scrolled-up viewer never gets yanked back to the tail
        # by an incoming append, while a viewer parked at the tail
        # keeps streaming.
        #
        # The "at the bottom" check uses a one-row tolerance instead
        # of the strict ``is_vertical_scroll_end`` equality: when a
        # burst of writes arrives, the layout settles a frame later
        # than the scroll position, so a viewer that *was* parked at
        # EOF can momentarily report ``scroll_y == max_scroll_y - 1``
        # and silently disengage follow. The tolerance restores the
        # "stay glued to the tail across bursts" property without
        # making the check any more permissive in practice — a
        # genuine scroll-up moves the offset by tens of rows, not one.
        #
        # The explicit ``f`` toggle still matters: it forces follow
        # back on (and snaps to end) so an operator who has scrolled
        # up can resume tailing without manually scrolling to the
        # bottom row.
        if self._user_forced_follow_off:
            self._follow = False
        else:
            self._follow = (
                self._log_widget.scroll_y >= self._log_widget.max_scroll_y - 1
            )
        # Mirror ``_follow`` onto the widget's own ``auto_scroll`` so
        # the visible state matches what the smoke tests assert and
        # so any code path that bypasses our burst loop (a future
        # action / write) inherits the right default.
        self._log_widget.auto_scroll = self._follow

        # Coalesce the whole burst into one repaint via
        # ``App.batch_update`` and pass ``scroll_end=False`` to each
        # ``write`` so RichLog skips its built-in per-line scroll
        # request. Without this, a 50-line burst from a chatty agent
        # would queue 50 ``scroll_end`` calls (one per ``write``) and
        # 50 widget-refresh notifications inside one tick — each of
        # those translates into terminal bytes Textual has to ship
        # over the SSH link before the operator's next scroll wheel
        # event can be acted on, which is the visible source of the
        # "scrolling lags during heavy output" complaint.
        with self.app.batch_update():
            for line in new_lines:
                # ``write_line`` mirrors the raw text into the
                # widget's resize-replay deque alongside the actual
                # ``write``. A direct ``write`` would land in the
                # widget's pre-wrapped strip list but would NOT be
                # available for re-flowing on a future terminal
                # resize, so the transcript would visibly
                # de-synchronise from the new viewport width.
                self._log_widget.write_line(line, scroll_end=False)
        if self._follow:
            # One snap-to-tail for the whole tick. ``animate=False`` +
            # ``immediate=True`` means the scroll lands in this frame
            # instead of being smeared across the next handful of
            # animation frames — each interpolated frame would
            # otherwise be a separate viewport repaint over SSH.
            self._log_widget.scroll_end(
                animate=False, immediate=True, x_axis=False,
            )

    def _seed_initial_log(self) -> None:
        # Two seeding modes share one EOF-park step at the end:
        #
        # * ``initial_log_lines is None`` — render the trailing
        #   :data:`_FULL_LOG_SEED_CAP` lines of the agent log. This is
        #   the "press Enter on a run" path: the operator asked for
        #   the full raw transcript, but we cap the seed because
        #   streaming a multi-MiB log line-by-line into the ``RichLog``
        #   widget on mount produces one ``write`` (and one render)
        #   per line. Over a high-latency SSH link to a VPS that turns
        #   the screen-open into a visibly slow paint and leaves the
        #   buffer too fat for snappy scrolling. The cap matches the
        #   widget's ``max_lines`` ring-buffer so the seed and the
        #   steady-state buffer share the same shape.
        # * ``initial_log_lines`` is an int — render only that many
        #   trailing lines. This is the ``ai show --lines N`` path:
        #   the operator deliberately bounded the screen budget.
        #
        # Either way :func:`tail_text_file` reads at most a handful of
        # MiB off the tail of the file rather than the whole thing, so
        # the open is fast even for an agent log that has been growing
        # for hours.
        #
        # In both branches we then park the tailer's offset *directly*
        # at EOF — calling ``read_new_lines`` to "burn" historical
        # bytes would be wrong for logs larger than the per-tick cap
        # (~4 MiB), because the second tick would surface historical
        # bytes instead of new appends.
        if self._log_widget is None:
            return

        if self.paths.agent_log.exists():
            seed_lines = (
                _FULL_LOG_SEED_CAP
                if self.initial_log_lines is None
                else self.initial_log_lines
            )
            # Coalesce the whole seed burst into a single repaint so
            # the operator never sees the screen draw in line by line
            # over SSH — Textual would otherwise queue one widget
            # refresh per ``write`` and stream a partial diff for
            # each. ``app.batch_update`` suspends those refreshes for
            # the duration of the ``with`` block, and the writes that
            # land before the widget's first Resize event are still
            # buffered in :attr:`RichLog._deferred_renders` exactly
            # as before — ``batch_update`` is a no-op for that path,
            # which is what makes it safe to wrap unconditionally.
            with self.app.batch_update():
                for line in tail_text_file(
                    self.paths.agent_log, lines=seed_lines,
                ):
                    # Same reason as in ``_refresh_log``: route every
                    # seeded line through ``write_line`` so it lands
                    # in the resize-replay mirror. Without this a
                    # post-seed terminal resize would leave the
                    # initial transcript wrapped at the old width
                    # while only newly-arriving lines re-flowed.
                    self._log_widget.write_line(line)
            # Snap to the tail once the seed is in. With the widget's
            # ``auto_scroll=False`` we don't get this for free from
            # the writes themselves, and an operator opening the
            # detail screen expects to land at "what's happening
            # right now" rather than at line 1 of the transcript.
            self._log_widget.scroll_end(
                animate=False, immediate=True, x_axis=False,
            )
        # Always advance to EOF — even if the file does not exist yet,
        # ``seek_to_end`` is a no-op that leaves the offset at 0 ready
        # for the first append.
        self._tailer.seek_to_end()

    # ── key bindings ──

    def action_quit(self) -> None:
        self.app.exit()

    def action_back(self) -> None:
        # Pop back to the run list (if there is one) — otherwise treat
        # ``Esc`` as quit. The bare ``ai`` entry pushes a list; the
        # standalone ``ai show`` pushes only the detail screen.
        if len(self.app.screen_stack) > 1:
            self.app.pop_screen()
        else:
            self.app.exit()

    def action_scroll_log_down(self) -> None:
        if self._log_widget is not None:
            self._log_widget.scroll_down()
            # Manual scrolling no longer needs to flip ``_follow``
            # explicitly: ``_refresh_log`` re-derives it from the
            # widget's ``is_vertical_scroll_end`` on the next tick,
            # so this binding stays consistent with mouse-wheel
            # scrolling without diverging from it.

    def action_scroll_log_up(self) -> None:
        if self._log_widget is not None:
            self._log_widget.scroll_up()

    def action_scroll_log_top(self) -> None:
        if self._log_widget is not None:
            self._log_widget.scroll_home()

    def action_scroll_log_bottom(self) -> None:
        if self._log_widget is not None:
            self._log_widget.scroll_end()
            # Snapping to the bottom is also an implicit "resume
            # follow" gesture, so clear the explicit-off override.
            self._user_forced_follow_off = False
            self._follow = True

    def action_toggle_follow(self) -> None:
        # ``f`` flips the explicit override. Turning follow ON snaps
        # to the tail so the next tick keeps streaming; turning it OFF
        # latches the override so an operator parked at EOF doesn't
        # have follow silently re-enabled by ``_refresh_log``'s
        # ``is_vertical_scroll_end`` sync.
        if self._follow:
            self._user_forced_follow_off = True
            self._follow = False
            if self._log_widget is not None:
                self._log_widget.auto_scroll = False
        else:
            self._user_forced_follow_off = False
            self._follow = True
            if self._log_widget is not None:
                self._log_widget.scroll_end()
                self._log_widget.auto_scroll = True
        self.notify(
            f"follow: {'on' if self._follow else 'off'}",
            severity="information",
        )


# ── App wrappers ────────────────────────────────────────────────────────────


# Animation level applied to both apps.
#
# Textual ships three levels — ``"full"`` (the default; smooth
# scrolls, focus-ring fades, dialog slide-ins), ``"basic"`` (only
# state-changing animations), and ``"none"`` (no tweening at all).
# We pin to ``"none"`` because the operator console is overwhelmingly
# driven over SSH to a remote VPS, and every interpolated animation
# frame is a separate viewport diff Textual has to ship across the
# wire before the next keystroke or scroll wheel event can be acted
# on. Locally the animations are pleasant chrome; over a ~100 ms
# round-trip they are the difference between "scroll feels glued to
# the wheel" and "scroll feels like it's mid-Atlantic". The env var
# ``TEXTUAL_ANIMATIONS`` (read by the constructor) still wins if an
# operator wants the full experience back; we just lower the default.
_TUI_ANIMATION_LEVEL = "none"


class RunListApp(App):
    """Top-level app for the bare ``ai`` command (no subcommand)."""

    TITLE = "auto-iterator"
    SUB_TITLE = "operator console"

    # Disable the built-in ctrl+p command palette. We don't expose
    # any commands through it, and leaving it on means every
    # keystroke pays for a palette-binding check plus the palette
    # adds a focus-trap widget Textual has to track in its hover /
    # focus walks. Free win.
    ENABLE_COMMAND_PALETTE = False

    def __init__(self, runs_dir: Path) -> None:
        super().__init__()
        self.runs_dir = runs_dir
        self.animation_level = _TUI_ANIMATION_LEVEL

    def on_mount(self) -> None:
        self.push_screen(RunListScreen(self.runs_dir))


class RunDetailApp(App):
    """Top-level app for ``ai show <run_id>`` (TTY default).

    Pushes only the detail screen; ``Esc`` becomes a quit instead of
    pop because there's no parent screen to fall back to."""

    TITLE = "auto-iterator"

    ENABLE_COMMAND_PALETTE = False

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
        self.animation_level = _TUI_ANIMATION_LEVEL

    def on_mount(self) -> None:
        self.push_screen(
            RunDetailScreen(
                self.paths,
                refresh_seconds=self.refresh_seconds,
                initial_log_lines=self.initial_log_lines,
            )
        )


# ── Entry points used by ``cli`` ────────────────────────────────────────────


def run_list_app(runs_dir: Path) -> int:
    """Launch the run-list TUI. Returns a CLI exit code."""
    app = RunListApp(runs_dir)
    app.run()
    return 0


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
    app = RunDetailApp(
        paths,
        refresh_seconds=refresh_seconds,
        initial_log_lines=initial_log_lines,
    )
    app.run()
    return 0


__all__ = [
    "RunDetailApp",
    "RunDetailScreen",
    "RunListApp",
    "RunListScreen",
    "run_detail_app",
    "run_list_app",
]
