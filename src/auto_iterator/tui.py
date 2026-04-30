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

from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
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
from .display import LogTailer
from .ls import RunRow, list_runs, reconcile_status
from .meta import read_meta
from .run_dir import RunPaths, read_json


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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
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

        We deliberately re-render every cell rather than tracking
        only changed rows: ``list_runs`` is already stable order
        (by ``started_at`` desc), and a full repaint of a few dozen
        rows is invisibly fast. Tracking deltas would just add bugs
        without saving anything operators can perceive."""
        if self._table is None:
            return
        try:
            rows = list_runs(self.runs_dir)
        except OSError:
            rows = []
        self._rows_by_key = {r.run_id: r for r in rows}
        # Preserve cursor on the same run_id across refreshes when
        # possible so a row that scrolled doesn't yank focus.
        prev_run_id: Optional[str] = None
        try:
            cursor_row = self._table.cursor_row
            if 0 <= cursor_row < self._table.row_count:
                prev_run_id = self._table.get_row_at(cursor_row)[0]
        except Exception:
            prev_run_id = None
        self._table.clear()
        for row in rows:
            self._table.add_row(*_row_cells(row), key=row.run_id)
        if prev_run_id is not None and prev_run_id in self._rows_by_key:
            try:
                idx = list(self._rows_by_key.keys()).index(prev_run_id)
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
        self._log_widget: Optional[RichLog] = None

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
        log = RichLog(
            id="log-panel",
            wrap=True,
            highlight=False,
            markup=False,
            auto_scroll=True,
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
        self.set_interval(self.refresh_seconds, self._refresh_status)
        self.set_interval(0.2, self._refresh_log)

    # ── refresh handlers ──

    def _refresh_status(self) -> None:
        if self._status_widget is None:
            return
        try:
            text = _minimal_status_line(self.paths)
        except Exception as exc:
            self._status_widget.update(f"(status unavailable: {exc})")
            return
        self._status_widget.update(_strip_ansi(text))

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
        # widget's ``is_vertical_scroll_end`` flag. Doing the sync
        # here means a mouse-scrolled-up viewer never gets yanked
        # back to the tail by an incoming append, while a viewer
        # parked at the tail keeps streaming.
        #
        # The explicit ``f`` toggle still matters: it forces follow
        # back on (and snaps to end) so an operator who has scrolled
        # up can resume tailing without manually scrolling to the
        # bottom row.
        if self._user_forced_follow_off:
            # Operator pressed ``f`` to disable follow; respect that
            # even when they happen to be parked at EOF.
            self._follow = False
        else:
            self._follow = bool(self._log_widget.is_vertical_scroll_end)
        self._log_widget.auto_scroll = self._follow
        for line in new_lines:
            self._log_widget.write(line)

    def _seed_initial_log(self) -> None:
        # Two seeding modes share one EOF-park step at the end:
        #
        # * ``initial_log_lines is None`` — stream the whole agent log
        #   into the widget, line by line. This is the "press Enter on
        #   a run" path: the operator asked for the full raw
        #   transcript and we honor it. We iterate the file handle
        #   instead of ``read_text()`` so memory stays bounded by the
        #   single longest line rather than the whole file.
        # * ``initial_log_lines`` is an int — render only that many
        #   trailing lines via :func:`tail_text_file`. This is the
        #   ``ai show --lines N`` path: the operator deliberately
        #   bounded the screen budget.
        #
        # In both branches we then park the tailer's offset *directly*
        # at EOF — calling ``read_new_lines`` to "burn" historical
        # bytes would be wrong for logs larger than the per-tick cap
        # (~4 MiB), because the second tick would surface historical
        # bytes instead of new appends.
        if self._log_widget is None:
            return

        if self.paths.agent_log.exists():
            if self.initial_log_lines is None:
                try:
                    with self.paths.agent_log.open(
                        "r", encoding="utf-8", errors="replace"
                    ) as fh:
                        for raw in fh:
                            self._log_widget.write(raw.rstrip("\r\n"))
                except OSError:
                    # File vanished mid-read (rotated / cleaned up):
                    # fall through to the EOF park so subsequent ticks
                    # behave as if the screen opened on an empty log.
                    pass
            else:
                from .display import tail_text_file

                for line in tail_text_file(
                    self.paths.agent_log, lines=self.initial_log_lines,
                ):
                    self._log_widget.write(line)
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


class RunListApp(App):
    """Top-level app for the bare ``ai`` command (no subcommand)."""

    TITLE = "auto-iterator"
    SUB_TITLE = "operator console"

    def __init__(self, runs_dir: Path) -> None:
        super().__init__()
        self.runs_dir = runs_dir

    def on_mount(self) -> None:
        self.push_screen(RunListScreen(self.runs_dir))


class RunDetailApp(App):
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
