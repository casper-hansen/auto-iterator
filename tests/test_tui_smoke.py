"""Smoke tests for the pyratatui TUI.

The TUI's screens and modals are pure-Python state machines (see the
docstring at the top of :mod:`auto_iterator.tui`), so these tests
drive them directly: no terminal, no event loop, no `pyratatui` native
binding loaded. The state machine is the contract — the renderer is a
stateless function over it — and that's exactly what we pin here.

Migrating away from Textual let us drop the async Pilot harness; every
test in this module is a synchronous function. ``pytest-asyncio`` is no
longer a dev dependency.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_iterator import actions  # noqa: E402
from auto_iterator.events import EventLog, RunState  # noqa: E402
from auto_iterator.ls import RunRow  # noqa: E402
from auto_iterator.meta import write_meta  # noqa: E402
from auto_iterator.run_dir import (  # noqa: E402
    RunPaths,
    create_run_dir,
    new_run_id,
    now_iso,
)
from auto_iterator.tui import (  # noqa: E402
    KeyEvent,
    RunDetailApp,
    RunDetailScreen,
    RunListApp,
    RunListScreen,
    _BackendChoiceModal,
    _compute_log_window,
    _ConfirmModal,
    _DiffViewer,
    _PromptModal,
    _rendered_rows_for,
    _render_run_detail,
    _render_run_list,
    _row_cells,
    _strip_ansi,
    _total_rendered_rows,
    _wrap_aware_tail,
)


# ── Fixtures and helpers ────────────────────────────────────────────────────


def _make_run_row(run_id: str = "20260430T101010Z-aaa") -> RunRow:
    return RunRow(
        run_id=run_id,
        status="running",
        phase="review",
        outer=1,
        inner=2,
        last_verdict="needs_fixes",
        exit_code=None,
        approved=False,
        started_at="2026-04-30T10:00:00+00:00",
        updated_at="2026-04-30T10:01:00+00:00",
        workspace="/tmp/ws",
        prompt_preview="Implement feature X carefully.",
        pid=999_999_999,
    )


def _seed_run_dir(runs_dir: Path) -> RunPaths:
    """Create a real run dir with meta + state + a couple of events.

    ``pid`` is set to the current pytest process so the liveness gate
    used by mutation actions (``send`` / ``rewind`` — see
    :func:`auto_iterator.actions.runner_is_alive`) treats the runner
    as alive. Tests that want to assert the dead-runner branch
    explicitly flip ``meta.status`` to ``exited`` / ``killed`` /
    ``crashed`` after seeding."""
    paths = create_run_dir(runs_dir, new_run_id())
    write_meta(paths, {
        "run_id": paths.run_id,
        "pid": os.getpid(),
        "status": "running",
        "workspace": "/tmp/ws",
        "started_at": now_iso(),
        "heartbeat_at": now_iso(),
    })
    state = RunState(
        run_id=paths.run_id,
        prompt="Implement feature X carefully.",
        workspace="/tmp/ws",
    )
    state.outer = 1
    state.inner = 1
    state.phase = "review"
    log = EventLog(paths, state)
    log.emit("run_started", workspace="/tmp/ws")
    log.emit("inner_started", outer=1, inner=1)
    return paths


def _press(target, code: str, **mods) -> None:
    """Drive a single keystroke against *target*.

    ``target`` may be an ``_AppBase`` (uses ``dispatch_key``), a
    screen, or a modal — everything that exposes ``handle_key`` is
    accepted. Tests use this in lieu of Textual's
    ``await pilot.press(...)`` so the synchronous flavour stays
    short."""
    ev = KeyEvent(code=code, **mods)
    if hasattr(target, "dispatch_key"):
        target.dispatch_key(ev)
    elif hasattr(target, "handle_key"):
        target.handle_key(ev)
        if hasattr(target, "_reconcile"):
            target._reconcile()
    else:
        raise TypeError(f"cannot press a key against {type(target).__name__}")


# ── RunListScreen smoke tests ───────────────────────────────────────────────


def test_run_list_renders_one_row_per_run():
    """``list_runs`` mocked → screen shows one entry per run.

    Pyratatui has no widget tree to query; the contract under test is
    that ``RunListScreen.refresh_rows`` populates ``self.rows`` with
    one ``RunRow`` per ``list_runs`` result. The renderer trivially
    walks that list, so a populated ``self.rows`` is the property the
    operator-visible table inherits."""
    rows = [
        _make_run_row("20260430T101010Z-aaa"),
        _make_run_row("20260430T101111Z-bbb"),
        _make_run_row("20260430T101212Z-ccc"),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=rows,
        ):
            app = RunListApp(Path(tmp))
        assert isinstance(app.screen, RunListScreen)
        screen = app.screen
        assert len(screen.rows) == 3
        assert {r.run_id for r in screen.rows} == {r.run_id for r in rows}


def test_pressing_enter_pushes_detail_screen():
    """Selecting a row opens :class:`RunDetailScreen`."""
    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        paths = _seed_run_dir(runs_dir)
        row = _make_run_row(paths.run_id)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[row],
        ):
            app = RunListApp(runs_dir)
            _press(app, "Enter")

        assert isinstance(app.screen, RunDetailScreen)
        assert app.screen.paths.run_id == paths.run_id


def test_pressing_enter_seeds_full_agent_log():
    """The press-Enter path seeds the *entire* existing transcript.

    Reviewer pin: the run-list pushes ``RunDetailScreen(paths,
    initial_log_lines=None)`` so older log lines stay scrollable
    rather than being permanently absent. We seed with 200 lines (well
    above the previous 30-line cap) and assert that every single one
    is rendered into the buffer."""
    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        paths = _seed_run_dir(runs_dir)
        seed_lines = [f"agent line {i:04d}" for i in range(200)]
        paths.agent_log.write_text(
            "\n".join(seed_lines) + "\n", encoding="utf-8",
        )

        row = _make_run_row(paths.run_id)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[row],
        ):
            app = RunListApp(runs_dir)
            _press(app, "Enter")

        assert isinstance(app.screen, RunDetailScreen)
        assert app.screen.initial_log_lines is None, (
            "press-Enter must request the full transcript, "
            "not the bounded tail"
        )
        rendered = list(app.screen._lines)
        assert "agent line 0000" in rendered, (
            "the head of the log must be visible after Enter"
        )
        assert "agent line 0100" in rendered
        assert "agent line 0199" in rendered, (
            "the tail of the log must be visible after Enter"
        )


def test_run_detail_long_lines_kept_verbatim_for_wrap():
    """Long lines are stored verbatim in the buffer.

    Pyratatui's ``Paragraph`` widget re-wraps every frame, so the only
    thing the screen has to get right is "store the raw line as
    written by the agent". Wrapping happens in Rust at render time
    against the current viewport width — there is no
    pre-baked-strip-list to invalidate the way :class:`RichLog` had,
    so this contract is much narrower than the Textual era."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run_dir(Path(tmp))
        long_line = "x" * 500
        paths.agent_log.write_text(long_line + "\n", encoding="utf-8")

        screen = RunDetailScreen(paths, refresh_seconds=0.1, initial_log_lines=5)
        screen.on_mount()

        assert long_line in screen._lines, (
            "the agent-output buffer must hold the raw line so the "
            "renderer can re-wrap it at any viewport width"
        )


def test_run_detail_buffer_survives_simulated_resize():
    """Resizing the terminal is a no-op against the buffer.

    The Textual era had to maintain a parallel raw-line mirror inside
    a ``_WrapAwareRichLog`` because RichLog pre-wrapped each ``write``
    into ``Strip`` objects and never re-flowed them. Pyratatui's
    ``Paragraph`` re-renders from the source string every frame, so
    the test reduces to "the line we wrote is still in the buffer
    after we changed the viewport" — which is trivially true.

    We do still assert the contract because this was the user-reported
    "blue bar at the bottom on resize" symptom: we want to be sure no
    future refactor reintroduces a stale per-line cache."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run_dir(Path(tmp))
        line_a = "x" * 240
        paths.agent_log.write_text(line_a + "\n", encoding="utf-8")

        screen = RunDetailScreen(paths, refresh_seconds=0.5, initial_log_lines=10)
        screen.on_mount()

        # Pretend a SIGWINCH narrowed the viewport. The renderer
        # simply paints a smaller area on the next tick — the buffer
        # is unchanged.
        screen._viewport_height = 30
        before = list(screen._lines)
        screen._viewport_height = 5
        after = list(screen._lines)
        assert before == after, (
            "viewport changes must not mutate the raw-line buffer"
        )
        assert line_a in after


# ── Modal flow tests ────────────────────────────────────────────────────────


def test_send_modal_writes_guidance_file():
    """Pressing ``s`` → modal → submit → ``control/guidance.txt`` written."""
    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        paths = _seed_run_dir(runs_dir)
        row = _make_run_row(paths.run_id)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[row],
        ):
            app = RunListApp(runs_dir)
            _press(app, "s")
            assert isinstance(app.screen.modals[-1], _PromptModal), (
                f"expected send modal, got "
                f"{type(app.screen.modals[-1]).__name__ if app.screen.modals else 'no modal'}"
            )
            modal = app.screen.modals[-1]
            modal.submit("Focus on the failing assertion in foo_test")
            app.dispatch_key(KeyEvent("Enter"))  # drain the done modal

        guidance_file = paths.control_file("guidance.txt")
        assert guidance_file.exists(), "guidance.txt must be written"
        content = guidance_file.read_text(encoding="utf-8")
        assert "Focus on the failing assertion in foo_test" in content
        # File shape is ``<ISO8601>\t<text>\n`` so a tab is present.
        assert "\t" in content


def test_quit_exits_without_signalling_any_pid(monkeypatch):
    """``q`` exits the TUI cleanly. ``os.kill`` must never be called.

    The TUI process must not own any runner lifecycles. We assert
    this by intercepting :func:`os.kill` in both ``actions`` and
    ``run_dir`` (which is what ``actions.signal_runner`` and
    ``pid_alive`` consult); the press-q path should never touch
    either."""
    kill_calls: list[tuple] = []

    def fake_kill(*args, **kwargs):
        kill_calls.append((args, kwargs))
        raise ProcessLookupError

    monkeypatch.setattr("auto_iterator.run_dir.os.kill", fake_kill)
    monkeypatch.setattr("auto_iterator.actions.os.kill", fake_kill)
    monkeypatch.setattr(os, "kill", fake_kill)

    rows = [_make_run_row()]
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=rows,
        ):
            app = RunListApp(Path(tmp))
            _press(app, "q")

        assert app._exit is True

    assert kill_calls == [], (
        f"TUI quit must not call os.kill, but it did: {kill_calls!r}"
    )


def test_pause_writes_pause_file():
    """``p`` on a selected row drops ``control/pause`` immediately."""
    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        paths = _seed_run_dir(runs_dir)
        row = _make_run_row(paths.run_id)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[row],
        ):
            app = RunListApp(runs_dir)
            _press(app, "p")

        assert paths.control_file("pause").exists()


def test_resume_clears_pause_file():
    """``r`` removes ``control/pause`` and tolerates the missing file."""
    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        paths = _seed_run_dir(runs_dir)
        actions.write_pause(paths)
        assert paths.control_file("pause").exists()
        row = _make_run_row(paths.run_id)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[row],
        ):
            app = RunListApp(runs_dir)
            _press(app, "r")

        assert not paths.control_file("pause").exists()


def test_rewind_modal_writes_rewind_file():
    """``w`` opens the rewind modal; submitting drops ``rewind.json``."""
    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        paths = _seed_run_dir(runs_dir)
        row = _make_run_row(paths.run_id)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[row],
        ):
            app = RunListApp(runs_dir)
            _press(app, "w")
            assert isinstance(app.screen.modals[-1], _PromptModal)
            modal = app.screen.modals[-1]
            modal.submit("outer=2,inner=3,phase=fix")
            app.dispatch_key(KeyEvent("Enter"))

        rewind_file = paths.control_file("rewind.json")
        assert rewind_file.exists()
        payload = json.loads(rewind_file.read_text(encoding="utf-8"))
        assert payload == {"outer": 2, "inner": 3, "phase": "fix"}


def test_kill_confirm_routes_through_signal_runner(monkeypatch):
    """``k`` then ``y`` must route through :func:`actions.signal_runner`.

    Pins the spec'd "kill always goes through actions.signal_runner so
    the CLI and TUI exercise the same path" rule."""
    signaled: list = []

    def fake_signal(paths, meta):
        signaled.append((paths.run_id, meta.get("pid")))
        return True

    monkeypatch.setattr(
        "auto_iterator.tui.actions.signal_runner", fake_signal,
    )

    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        paths = _seed_run_dir(runs_dir)
        row = _make_run_row(paths.run_id)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[row],
        ):
            app = RunListApp(runs_dir)
            _press(app, "k")
            assert isinstance(app.screen.modals[-1], _ConfirmModal)
            _press(app, "y")

        assert len(signaled) == 1
        assert signaled[0][0] == paths.run_id


def test_run_detail_streams_log_lines_incrementally():
    """``RunDetailScreen`` reads only new bytes per refresh (no full reload).

    Pin the spec'd "agent-log viewer never reads the whole file on
    refresh" property by inspecting the embedded :class:`LogTailer`'s
    offset before and after an append: the offset must only advance
    by the appended size, never re-read the whole file."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run_dir(Path(tmp))
        # Seed with 100 KiB of content so we can detect a re-read.
        seed = ("seed line\n" * 10240).encode("utf-8")
        paths.agent_log.write_bytes(seed)

        screen = RunDetailScreen(
            paths, refresh_seconds=0.1, initial_log_lines=5,
        )
        screen.on_mount()

        initial_offset = screen._tailer.offset
        assert initial_offset == paths.agent_log.stat().st_size, (
            "initial seed must advance the tailer offset to EOF"
        )

        # Append 1 KiB of new content.
        new_chunk = ("delta line\n" * 100).encode("utf-8")
        with paths.agent_log.open("ab") as fh:
            fh.write(new_chunk)

        screen._refresh_log()

        assert screen._tailer.offset == paths.agent_log.stat().st_size
        advance = screen._tailer.offset - initial_offset
        assert advance == len(new_chunk), (
            f"tailer advanced by {advance} bytes; expected "
            f"{len(new_chunk)}"
        )


def test_run_detail_seed_skips_to_eof_on_huge_log():
    """A multi-MiB pre-existing log must seed at EOF, not part-way.

    Reviewer pin: the previous implementation called
    ``LogTailer.read_new_lines()`` to "burn" the file after rendering
    the bounded tail, which only advances the offset by the per-tick
    cap (≈4 MiB). On larger files the next tick would surface old
    bytes as if they were new, polluting the agent-output panel with
    historical content.

    With ``seek_to_end`` the offset lands at exactly ``st_size`` on
    mount regardless of how big the file is."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run_dir(Path(tmp))
        # 8 MiB — over the per-tick read cap.
        payload = (b"x" * 4095 + b"\n") * 2048
        paths.agent_log.write_bytes(payload)
        size = paths.agent_log.stat().st_size

        screen = RunDetailScreen(
            paths, refresh_seconds=0.1, initial_log_lines=5,
        )
        screen.on_mount()

        assert screen._tailer.offset == size, (
            "seed must park the tailer offset at EOF for huge logs; "
            f"offset={screen._tailer.offset}, size={size}"
        )
        screen._refresh_log()
        assert screen._tailer.offset == size


def test_log_follow_pins_when_user_scrolls_away():
    """Scrolling up disengages follow: incoming appends do not yank
    the operator back to EOF.

    The Textual era had to derive ``_follow`` from
    ``RichLog.is_vertical_scroll_end`` because mouse-wheel scrolling
    bypassed the custom ``j``/``k``/``g``/``G`` actions. Pyratatui has
    no separate widget-internal scroll position to keep in sync — all
    scroll changes go through the screen's own actions, which is the
    source of truth.
    """
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run_dir(Path(tmp))
        paths.agent_log.write_bytes(("seed line\n" * 200).encode("utf-8"))

        screen = RunDetailScreen(
            paths, refresh_seconds=0.1, initial_log_lines=200,
        )
        screen.on_mount()

        assert screen._follow is True, (
            "test setup: the screen must start in follow mode"
        )

        # Scroll all the way to the top — equivalent to mouse-wheel
        # scrolling away from the tail.
        screen.action_scroll_log_top()
        assert screen._follow is False, (
            "scrolling to the head of the buffer must disengage follow"
        )

        with paths.agent_log.open("ab") as fh:
            fh.write(("delta line\n" * 50).encode("utf-8"))
        screen._refresh_log()

        assert screen._follow is False, (
            "appending while scrolled up must not silently re-enable "
            "follow"
        )
        # The visible window must NOT include the most recently
        # appended "delta" lines — that's the operator-facing pin
        # behavior.
        screen._viewport_height = 10
        visible = screen.visible_lines()
        assert all("delta" not in line for line in visible), (
            "scrolled-away viewport must not yank back to the new tail"
        )


def test_log_follow_resumes_when_user_returns_to_bottom():
    """Pressing ``G`` re-enables auto-follow — the inverse of the pin
    behavior. Without this, an operator who scrolled up to inspect
    history would have to press ``f`` to resume tailing."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run_dir(Path(tmp))
        paths.agent_log.write_bytes(("seed line\n" * 200).encode("utf-8"))

        screen = RunDetailScreen(
            paths, refresh_seconds=0.1, initial_log_lines=200,
        )
        screen.on_mount()

        screen.action_scroll_log_top()
        with paths.agent_log.open("ab") as fh:
            fh.write(b"one\n")
        screen._refresh_log()
        assert screen._follow is False

        # Operator scrolls back to the bottom (G).
        screen.action_scroll_log_bottom()
        assert screen._follow is True

        with paths.agent_log.open("ab") as fh:
            fh.write(b"two\n")
        screen._refresh_log()
        assert screen._follow is True
        # And the visible window now shows the latest line.
        assert any(
            "two" in line for line in list(screen._lines)[-3:]
        )


def test_toggle_follow_latches_off_until_next_explicit_action():
    """Pressing ``f`` while following latches the override OFF so a
    geometry change can't silently re-enable follow."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run_dir(Path(tmp))
        paths.agent_log.write_bytes(("seed line\n" * 50).encode("utf-8"))

        screen = RunDetailScreen(
            paths, refresh_seconds=0.1, initial_log_lines=50,
        )
        screen.on_mount()
        assert screen._follow is True
        screen.action_toggle_follow()
        assert screen._follow is False
        assert screen._user_forced_follow_off is True
        # A new append doesn't silently re-engage follow.
        with paths.agent_log.open("ab") as fh:
            fh.write(b"delta\n")
        screen._refresh_log()
        assert screen._follow is False

        # Pressing ``f`` again clears the latch.
        screen.action_toggle_follow()
        assert screen._follow is True
        assert screen._user_forced_follow_off is False


def test_send_rejects_dead_runner():
    """``s`` must refuse to drop ``guidance.txt`` for a run whose
    runner is gone, mirroring the CLI's ``_drop_mutation`` gate.

    Reviewer pin: the TUI used to write the guidance file
    unconditionally, leaving stale control files for runs the CLI
    would have rejected. The two front-ends must agree on the
    liveness gate so the on-disk state is consistent."""
    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        paths = _seed_run_dir(runs_dir)
        from auto_iterator.meta import update_meta

        update_meta(paths, status="exited")

        row = _make_run_row(paths.run_id)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[row],
        ):
            app = RunListApp(runs_dir)
            _press(app, "s")
            assert isinstance(app.screen.modals[-1], _PromptModal)
            modal = app.screen.modals[-1]
            modal.submit("Should be rejected")
            app.dispatch_key(KeyEvent("Enter"))

        guidance_file = paths.control_file("guidance.txt")
        assert not guidance_file.exists(), (
            "TUI must reject guidance writes for dead runners "
            "(matching the CLI's _drop_mutation gate)"
        )


def test_rewind_rejects_dead_runner():
    """``w`` must refuse to drop ``rewind.json`` for a dead runner.

    Same liveness gate as ``send`` — keeps the protocol consistent
    across both control-file mutators."""
    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        paths = _seed_run_dir(runs_dir)
        from auto_iterator.meta import update_meta

        update_meta(paths, status="killed")

        row = _make_run_row(paths.run_id)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[row],
        ):
            app = RunListApp(runs_dir)
            _press(app, "w")
            assert isinstance(app.screen.modals[-1], _PromptModal)
            modal = app.screen.modals[-1]
            modal.submit("outer=1,inner=1,phase=review")
            app.dispatch_key(KeyEvent("Enter"))

        rewind_file = paths.control_file("rewind.json")
        assert not rewind_file.exists(), (
            "TUI must reject rewind writes for dead runners"
        )


def test_new_run_cursor_preset_pins_backend(monkeypatch):
    """Picking the Cursor preset (key ``1``) overrides any env-var
    backend selection so the canonical "Opus impl + GPT reviewer"
    layout is what actually runs.

    Pins the "user can see which backend will run" contract end-to-end:
    the preset is opinionated, not a hint. The hostile env exercised
    here is the exact one a previous review found bypassed the
    contract — ``AGENT_CMD`` would survive into the spawned cfg and
    ``AGENT_REVIEWER_BACKEND`` would silently turn the "Cursor" pick
    into a mixed Cursor/Codex run. Both must be ignored now.
    """
    from auto_iterator.backends import BACKENDS

    monkeypatch.setenv("AGENT_BACKEND", "claude-code")
    monkeypatch.setenv("AGENT_CMD", "claude-fake-binary")
    monkeypatch.setenv("AGENT_REVIEWER_BACKEND", "codex")
    monkeypatch.setenv("AGENT_REVIEWER_CMD", "codex-fake-binary")
    monkeypatch.setenv("AGENT_IMPL_BACKEND", "codex")
    monkeypatch.setenv("AGENT_FIX_BACKEND", "codex")

    captured: dict = {}

    def fake_spawn(runs_dir, cfg, **kwargs):
        captured["cfg"] = cfg
        from auto_iterator.actions import ActionResult

        return ActionResult(ok=True, run_id="cursor-preset-run")

    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[],
        ), mock.patch(
            "auto_iterator.tui.actions.spawn_runner_detached",
            side_effect=fake_spawn,
        ):
            app = RunListApp(runs_dir)
            _press(app, "n")
            assert isinstance(app.screen.modals[-1], _PromptModal)
            app.screen.modals[-1].submit("Task")
            app.dispatch_key(KeyEvent("Enter"))
            assert isinstance(app.screen.modals[-1], _PromptModal)
            app.screen.modals[-1].submit(str(runs_dir))
            app.dispatch_key(KeyEvent("Enter"))
            assert isinstance(app.screen.modals[-1], _BackendChoiceModal)
            _press(app, "1")

    assert "cfg" in captured
    cfg = captured["cfg"]
    assert cfg.backend == "cursor", (
        f"Cursor preset must pin backend=cursor regardless of env; got {cfg.backend!r}"
    )
    assert cfg.backend_for("impl") == "cursor"
    assert cfg.backend_for("fix") == "cursor"
    assert cfg.backend_for("reviewer") == "cursor"
    assert cfg.has_mixed_backends is False
    cursor_default = BACKENDS["cursor"].default_cmd
    assert cfg.agent_cmd == cursor_default, (
        f"Cursor preset must ignore $AGENT_CMD; "
        f"got agent_cmd={cfg.agent_cmd!r}"
    )
    assert cfg.impl_agent_cmd is None
    assert cfg.fix_agent_cmd is None
    assert cfg.reviewer_agent_cmd is None


def test_new_run_claude_codex_preset_picks_mixed_backends(monkeypatch):
    """Picking the Claude+Codex preset (key ``2``) yields the canonical
    mixed setup: Claude Code for impl/fix, Codex as the reviewer.

    Equivalent to ``ai run --backend claude-code --reviewer-backend
    codex`` from the shell — same RunConfig either way.

    Exercises a hostile env that previously bypassed the preset: a
    stale ``AGENT_IMPL_BACKEND=codex`` would silently flip the
    implementer to Codex, ``AGENT_REVIEWER_CMD=agent`` would smuggle
    Cursor's binary into the Codex reviewer phase, and ``AGENT_CMD``
    would override the global Claude Code binary. The preset must pin
    every one of those fields back to its canonical layout.
    """
    from auto_iterator.backends import BACKENDS

    monkeypatch.setenv("AGENT_BACKEND", "cursor")
    monkeypatch.setenv("AGENT_CMD", "agent")
    monkeypatch.setenv("AGENT_IMPL_BACKEND", "codex")
    monkeypatch.setenv("AGENT_FIX_BACKEND", "cursor")
    monkeypatch.setenv("AGENT_REVIEWER_CMD", "agent")
    monkeypatch.setenv("AGENT_IMPL_CMD", "agent")

    captured: dict = {}

    def fake_spawn(runs_dir, cfg, **kwargs):
        captured["cfg"] = cfg
        from auto_iterator.actions import ActionResult

        return ActionResult(ok=True, run_id="mixed-preset-run")

    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[],
        ), mock.patch(
            "auto_iterator.tui.actions.spawn_runner_detached",
            side_effect=fake_spawn,
        ):
            app = RunListApp(runs_dir)
            _press(app, "n")
            app.screen.modals[-1].submit("Task")
            app.dispatch_key(KeyEvent("Enter"))
            app.screen.modals[-1].submit(str(runs_dir))
            app.dispatch_key(KeyEvent("Enter"))
            assert isinstance(app.screen.modals[-1], _BackendChoiceModal)
            _press(app, "2")

    assert "cfg" in captured
    cfg = captured["cfg"]
    assert cfg.backend == "claude-code"
    assert cfg.backend_for("impl") == "claude-code"
    assert cfg.backend_for("fix") == "claude-code"
    assert cfg.backend_for("reviewer") == "codex"
    assert cfg.has_mixed_backends is True
    claude_default = BACKENDS["claude-code"].default_cmd
    codex_default = BACKENDS["codex"].default_cmd
    assert cfg.agent_cmd == claude_default, (
        f"Claude+Codex preset must ignore $AGENT_CMD; "
        f"got agent_cmd={cfg.agent_cmd!r}"
    )
    assert cfg.reviewer_agent_cmd == codex_default, (
        f"Reviewer phase must use Codex's default_cmd, "
        f"not $AGENT_REVIEWER_CMD; got {cfg.reviewer_agent_cmd!r}"
    )
    assert cfg.impl_agent_cmd is None
    assert cfg.fix_agent_cmd is None


def test_new_run_backend_modal_cancel_skips_spawn():
    """Pressing Esc on the backend picker must abort the new-run flow
    without spawning a runner. Pin the cancel-path symmetry with the
    earlier prompt/workspace modals so a half-typed flow can be undone
    at any step."""
    spawn_calls: list = []

    def fake_spawn(*args, **kwargs):
        spawn_calls.append((args, kwargs))
        from auto_iterator.actions import ActionResult

        return ActionResult(ok=True, run_id="should-not-happen")

    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[],
        ), mock.patch(
            "auto_iterator.tui.actions.spawn_runner_detached",
            side_effect=fake_spawn,
        ):
            app = RunListApp(runs_dir)
            _press(app, "n")
            app.screen.modals[-1].submit("Task")
            app.dispatch_key(KeyEvent("Enter"))
            app.screen.modals[-1].submit(str(runs_dir))
            app.dispatch_key(KeyEvent("Enter"))
            assert isinstance(app.screen.modals[-1], _BackendChoiceModal)
            _press(app, "Escape")

    assert spawn_calls == [], (
        "Cancelling the backend picker must abort the new-run flow"
    )


def test_diff_modal_close_pops_overlay():
    """``Esc`` / ``q`` on the diff viewer dismisses it.

    The diff viewer is read-only; the only operator-facing interaction
    is "close it again". We pin both shortcuts so an operator who
    treats Esc as the universal back-button is matched, and so an
    operator who already has a finger on ``q`` (the screen-level
    quit) doesn't accidentally exit the whole TUI."""
    diff = _DiffViewer(title="Diff", body="line1\nline2\nline3")
    diff.handle_key(KeyEvent("Escape"))
    assert diff.done is True

    diff = _DiffViewer(title="Diff", body="x")
    diff.handle_key(KeyEvent("q"))
    assert diff.done is True


def test_run_list_renders_status_bar_for_detail_screen_after_open():
    """After Enter, the detail screen's ``status_text`` is non-empty
    once :meth:`_refresh_status` has run. Mirrors what the operator
    sees in the top bar of the per-run view."""
    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        paths = _seed_run_dir(runs_dir)
        row = _make_run_row(paths.run_id)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[row],
        ):
            app = RunListApp(runs_dir)
            _press(app, "Enter")
        screen = app.screen
        assert isinstance(screen, RunDetailScreen)
        assert screen.status_text != "(loading status...)"
        # Run id always lands in the bar — that's the operator-facing
        # "which run am I looking at" anchor.
        assert paths.run_id in screen.status_text


# ── Modal-only unit checks ──────────────────────────────────────────────────


def test_prompt_modal_typing_assembles_value():
    """``handle_key`` accumulates printable keys into ``value``.

    The renderer paints whatever ``value`` says, so the entire input
    contract for a modal reduces to "the chars I typed appear in
    self.value in order"."""
    modal = _PromptModal(title="t", placeholder="ph", value="")
    for ch in "hello":
        modal.handle_key(KeyEvent(ch))
    assert modal.value == "hello"
    assert modal.cursor == 5
    # Backspace deletes the last char.
    modal.handle_key(KeyEvent("Backspace"))
    assert modal.value == "hell"
    # Enter submits.
    modal.handle_key(KeyEvent("Enter"))
    assert modal.done is True
    assert modal.submitted is True


def test_prompt_modal_escape_cancels_without_submit():
    """``Esc`` on a prompt modal cancels — neither flag flips submit."""
    modal = _PromptModal(title="t")
    modal.handle_key(KeyEvent("a"))
    modal.handle_key(KeyEvent("Escape"))
    assert modal.done is True
    assert modal.submitted is False


def test_confirm_modal_y_n_paths():
    """The confirm modal's ``y`` / ``n`` / Esc routes match the spec."""
    yes = _ConfirmModal(title="kill?")
    yes.handle_key(KeyEvent("y"))
    assert yes.done is True
    assert yes.confirmed is True

    no = _ConfirmModal(title="kill?")
    no.handle_key(KeyEvent("n"))
    assert no.done is True
    assert no.confirmed is False

    esc = _ConfirmModal(title="kill?")
    esc.handle_key(KeyEvent("Escape"))
    assert esc.done is True
    assert esc.confirmed is False


def test_backend_choice_modal_arrow_then_enter():
    """↑ / ↓ + Enter selects a preset just like the digit shortcuts."""
    modal = _BackendChoiceModal()
    assert modal.selected_idx == 0
    modal.handle_key(KeyEvent("Down"))
    assert modal.selected_idx == 1
    modal.handle_key(KeyEvent("Enter"))
    assert modal.done is True
    assert modal.chosen is not None
    assert modal.chosen.get("backend") == "claude-code"


# ── Plain (non-async) TUI helper tests ──────────────────────────────────────


def test_row_cells_shape():
    """One :class:`RunRow` becomes the seven Table columns."""
    cells = _row_cells(_make_run_row())
    assert len(cells) == 7
    assert cells[0] == "20260430T101010Z-aaa"
    assert cells[1] == "running"
    assert cells[2] == "review"
    assert cells[3] == "1/2"
    assert cells[4] == "needs_fixes"
    assert cells[5].startswith("2026-04-30T10:01:00")
    assert "Implement feature X carefully" in cells[6]


def test_strip_ansi_removes_escape_codes():
    """The status panel re-uses ``_status_section_lines`` which emits
    ANSI escapes; the TUI must render them as plain strings (styling
    flows through pyratatui's ``Style``/``Span`` instead), not raw
    bytes."""
    raw = "\x1b[1mRun foo\x1b[0m\nstatus  \x1b[32mrunning\x1b[0m"
    out = _strip_ansi(raw)
    assert "\x1b" not in out
    assert "Run foo" in out
    assert "running" in out


# ── App-level integration smoke ─────────────────────────────────────────────


def test_run_detail_app_smoke_constructs_without_terminal():
    """``RunDetailApp(...)`` must instantiate without a TTY.

    The CLI's ``cmd_show`` path calls ``RunDetailApp(...).run()``; the
    constructor mounts the screen and seeds the log eagerly. If
    construction touched a Terminal, every ``ai show <run_id> --json``
    test would fail with "no TTY" the moment we lazy-imported the
    module. The contract is "the live ``run()`` is the only thing
    that opens pyratatui" — so we assert the constructor + on_mount
    chain runs to completion without a Terminal."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run_dir(Path(tmp))
        paths.agent_log.write_text("seed\n", encoding="utf-8")
        app = RunDetailApp(paths, refresh_seconds=0.1, initial_log_lines=5)
        assert isinstance(app.screen, RunDetailScreen)
        assert "seed" in app.screen._lines


def test_run_list_cursor_navigation():
    """Up/Down move the table cursor without leaving the row range."""
    rows = [
        _make_run_row(f"20260430T1010{i:02d}Z-aaa") for i in range(3)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=rows,
        ):
            app = RunListApp(Path(tmp))
        screen = app.screen
        assert screen.cursor_row == 0
        _press(app, "Down")
        assert screen.cursor_row == 1
        _press(app, "Down")
        assert screen.cursor_row == 2
        _press(app, "Down")  # already at the last row, must clamp
        assert screen.cursor_row == 2
        _press(app, "Up")
        assert screen.cursor_row == 1
        _press(app, "End")
        assert screen.cursor_row == 2
        _press(app, "Home")
        assert screen.cursor_row == 0


# ── Render-path smoke tests (catch widget API regressions) ──────────────────
#
# Reviewer pin: the state-machine tests above exercise the screen
# objects but never *paint* a frame, so a pyratatui constructor
# signature drift (e.g. ``Table(rows, widths)`` getting flipped to
# ``Table(rows).column_widths(widths)``) used to slip through with all
# tests green. The live ``_run_app_loop`` swallows render exceptions to
# stay alive, which means the operator just gets an empty TUI rather
# than a stack trace.
#
# These tests drive the render functions against a stub ``frame`` that
# tracks the widget calls but otherwise hands a real ``pyratatui.Rect``
# to the renderer. The widgets themselves are constructed against the
# *real* pyratatui module — so any ctor/builder signature drift fails
# this test fast, the way the reviewer reproduced it manually.


class _StubFrame:
    """Minimal stand-in for ``pyratatui.Frame`` in the renderer tests.

    The real frame is only valid inside ``Terminal.draw``; the render
    functions only need three things from it: ``area`` (a Rect),
    ``render_widget(widget, area)``, and
    ``render_stateful_table(table, area, state)``. We stash whatever
    the renderer hands us so the test can assert on it later."""

    def __init__(self, area):
        self.area = area
        self.widgets: list = []
        self.tables: list = []

    def render_widget(self, widget, area):
        self.widgets.append((widget, area))

    def render_stateful_table(self, table, area, state):
        self.tables.append((table, area, state))


def _make_stub_frame(width: int = 120, height: int = 40):
    import pyratatui as pr

    return _StubFrame(pr.Rect(0, 0, width, height))


def test_render_run_list_constructs_widgets_against_real_pyratatui():
    """The full run-list render path must succeed against the real
    pyratatui module.

    Pin the constructor signatures: pyratatui 0.2.x ``Table`` only
    accepts ``Table(rows)`` and the column widths / header must be
    set via the ``.column_widths(...).header(...)`` builder chain.
    Passing them positionally raises ``TypeError`` at the call site,
    which the live ``term.draw`` handler then swallows — so the
    operator-visible symptom is "the run list never paints" rather
    than a clean traceback. This test forces the render function to
    actually construct every widget against the live module so a
    future API drift fails here instead.
    """
    import pyratatui as pr

    rows = [
        _make_run_row("20260430T101010Z-aaa"),
        _make_run_row("20260430T101111Z-bbb"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=rows,
        ):
            app = RunListApp(Path(tmp))

        frame = _make_stub_frame()
        # Must not raise — this is the contract that the reviewer's
        # manual probe broke.
        _render_run_list(app.screen, frame, pr)

        # And the renderer must have actually queued the table.
        assert frame.tables, (
            "_render_run_list must call frame.render_stateful_table"
        )
        # Title bar + footer were rendered as plain widgets.
        assert len(frame.widgets) >= 2


def test_render_run_list_with_modal_overlay_succeeds():
    """A modal on top of the run-list also exercises the modal
    overlay's widget construction. This is the second path that was
    skipped by the previous test suite — modals build their own
    Layout / Paragraph / Block tree."""
    import pyratatui as pr

    rows = [_make_run_row()]
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=rows,
        ):
            app = RunListApp(Path(tmp))
            _press(app, "n")  # opens the prompt modal

        frame = _make_stub_frame()
        _render_run_list(app.screen, frame, pr)

        # The modal painted at least one widget (the framed
        # paragraph) plus a Clear() to wipe the underlying area.
        assert len(frame.widgets) >= 2


def test_render_run_detail_constructs_widgets_against_real_pyratatui():
    """Same contract as the run-list, applied to the per-run detail
    screen. The detail screen's renderer is simpler (no Table) but
    still uses the Layout/Constraint/Paragraph/Block chain — and a
    pyratatui upgrade that broke any of those would silently empty
    the agent-output panel for the live operator."""
    import pyratatui as pr

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run_dir(Path(tmp))
        paths.agent_log.write_text(
            "agent line a\nagent line b\n", encoding="utf-8",
        )
        app = RunDetailApp(
            paths, refresh_seconds=0.1, initial_log_lines=5,
        )

        frame = _make_stub_frame()
        _render_run_detail(app.screen, frame, pr)

        # Status bar, agent-log Paragraph, footer — at least three
        # widgets land in the stub frame's queue.
        assert len(frame.widgets) >= 3


def test_render_run_detail_paragraph_has_wrap_enabled():
    """Reviewer pin: the per-run detail Paragraph must have
    ``.wrap(True)`` enabled so long agent lines fold instead of
    being clipped at the terminal's right edge.

    The Textual era's ``RichLog`` was constructed with ``wrap=True``;
    the pyratatui port initially forgot to thread that through and a
    reviewer reproduced the regression on a 50-column PTY where a
    500-char line rendered only its first visible columns. We pin
    the contract by introspecting the Paragraph's ``repr`` (which
    pyratatui emits as ``Paragraph(lines=..., wrap=true)``) so any
    future renderer change that drops the call fails this test
    instead of being silently swallowed by the live render loop."""
    import pyratatui as pr

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run_dir(Path(tmp))
        long_line = "x" * 500
        paths.agent_log.write_text(long_line + "\n", encoding="utf-8")
        app = RunDetailApp(
            paths, refresh_seconds=0.1, initial_log_lines=5,
        )

        frame = _make_stub_frame(width=50, height=24)
        _render_run_detail(app.screen, frame, pr)

        # The agent-log Paragraph is the second of the three widgets
        # the renderer pushes (status bar → log → footer). Find it
        # by area: it's the row whose constraint was ``fill(1)``,
        # i.e. the tallest of the three.
        agent_widget, _agent_area = max(
            frame.widgets, key=lambda pair: pair[1].height,
        )
        rendered = repr(agent_widget)
        assert "wrap=true" in rendered, (
            "the agent-log Paragraph must enable wrap so long lines "
            "fold instead of being clipped at the panel border; "
            f"got repr={rendered!r}"
        )


def test_wrap_aware_tail_picks_suffix_that_fits_rendered_rows():
    """Pin the wrap-aware follow-mode tail logic.

    With ``.wrap(True)`` on, a single long buffer line consumes
    multiple rendered rows. Naively slicing the last ``height``
    *logical* lines would push the latest content past the panel
    border. ``_wrap_aware_tail`` shrinks the suffix until it fits
    in ``height`` *rendered* rows so the tail (newest content)
    stays painted. This is the operator-visible contract behind
    the reviewer's "follow mode must not lose the latest output
    when long lines wrap" complaint."""
    short = "abc"
    long = "x" * 100  # at width=20 → 5 rendered rows

    # Width 20, height 5: the long line alone fills the panel.
    assert _wrap_aware_tail([short, short, long], width=20, height=5) == [
        long,
    ]

    # Width 20, height 6: long line (5 rows) + one short line (1
    # row) fit exactly.
    assert _wrap_aware_tail(
        [short, short, short, long], width=20, height=6,
    ) == [short, long]

    # Width 100, height 3: nothing wraps, so tail behaviour matches
    # logical-line slicing.
    assert _wrap_aware_tail(
        ["a", "b", "c", "d"], width=100, height=3,
    ) == ["b", "c", "d"]


def test_wrap_aware_tail_handles_degenerate_inputs():
    """Width/height of zero or empty buffers must not divide-by-zero
    or hang. The renderer reports a tiny area if the operator
    aggressively shrinks the terminal; the helper must fail safely."""
    assert _wrap_aware_tail([], width=10, height=10) == []
    assert _wrap_aware_tail(["a", "b"], width=0, height=2) == ["a", "b"]
    assert _wrap_aware_tail(["a", "b"], width=10, height=0) == ["b"]


def test_render_run_detail_follow_mode_keeps_latest_with_wrapped_lines():
    """End-to-end pin: a buffer with a long line followed by a fresh
    short line, rendered into a narrow viewport, must produce a
    Paragraph whose body ends with the latest line.

    Without the wrap-aware tail logic the long line would consume
    the whole viewport and the final short line would be clipped
    below the panel border."""
    import pyratatui as pr

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run_dir(Path(tmp))
        long_line = "L" * 500
        paths.agent_log.write_text(
            long_line + "\nlatest output line\n", encoding="utf-8",
        )
        app = RunDetailApp(
            paths, refresh_seconds=0.1, initial_log_lines=10,
        )
        # Sanity: follow mode is the default and this test exercises
        # it. If a future change flips that, the test should fail
        # loudly rather than silently exercise the paused branch.
        assert app.screen._follow is True

        frame = _make_stub_frame(width=50, height=12)
        _render_run_detail(app.screen, frame, pr)

        agent_widget, _agent_area = max(
            frame.widgets, key=lambda pair: pair[1].height,
        )
        rendered = repr(agent_widget)
        # ``Paragraph.__repr__`` only exposes ``lines`` count, not
        # the body, so we re-derive what the renderer would have
        # composed by calling the helper directly with the same
        # geometry. This mirrors the contract being asserted.
        log_height = max(1, frame.area.height - 4)  # status+footer+border
        log_width = max(1, frame.area.width - 2)
        body_lines = _wrap_aware_tail(
            app.screen._lines, log_width, log_height,
        )
        assert body_lines and body_lines[-1] == "latest output line", (
            "follow mode must keep the newest line at the bottom of "
            f"the visible window; got tail={body_lines!r}"
        )
        # And the rendered Paragraph still has wrap on.
        assert "wrap=true" in rendered


# ── Wrap-aware scroll model tests ───────────────────────────────────────────
#
# Reviewer pin: with ``.wrap(True)`` enabled, the scroll model must
# count *rendered rows*, not *logical lines*. Otherwise a single
# 500-char agent line in a 50-column viewport becomes unreachable past
# the first screenful: the previous logical-line clamp short-circuited
# at ``len(_lines) <= 1``, so ``j``/``k``/``PageUp`` did nothing and
# pyratatui clipped the rest of the wrapped output at the panel
# border. These tests pin the new rendered-row semantics.


def test_long_single_line_is_scrollable_by_rendered_rows():
    """A buffer of exactly one 500-char line in a 50-col viewport
    must be scrollable past the first screenful.

    With logical-line semantics the actions short-circuited at
    ``len(_lines) <= 1`` and ``_scroll_offset`` could never grow,
    leaving the wrapped tail clipped below the panel border. The fix
    counts rendered rows: 500 chars at width 50 ≈ 10 rendered rows,
    so PageUp on a 5-row viewport must reach a non-zero offset."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run_dir(Path(tmp))
        long_line = "x" * 500
        paths.agent_log.write_text(long_line + "\n", encoding="utf-8")

        screen = RunDetailScreen(
            paths, refresh_seconds=0.1, initial_log_lines=5,
        )
        screen.on_mount()
        screen._viewport_width = 50
        screen._viewport_height = 5

        # Sanity: there is exactly one logical line. The bug would
        # short-circuit here.
        assert len(screen._lines) == 1

        # Total rendered rows of one 500-char line at width 50 is 10.
        assert screen._total_rendered_rows() == 10

        # PageUp on a 5-row viewport: step = max(1, 5-1) = 4.
        screen.action_page_up()
        assert screen._scroll_offset == 4, (
            "PageUp must advance the scroll offset by ~viewport "
            "rows even when there's only one logical line"
        )
        assert screen._follow is False, (
            "any off-tail scroll must disengage follow"
        )

        # Single-row scroll-up still works after that.
        screen.action_scroll_log_up()
        assert screen._scroll_offset == 5

        # Top jumps to total_rendered - 1 = 9. follow stays off.
        screen.action_scroll_log_top()
        assert screen._scroll_offset == 9
        assert screen._follow is False

        # Bottom resets and re-engages follow.
        screen.action_scroll_log_bottom()
        assert screen._scroll_offset == 0
        assert screen._follow is True


def test_long_line_render_uses_paragraph_scroll_to_reveal_later_rows():
    """The renderer must hand the screen's rendered-row offset to
    ``Paragraph.scroll(y, x)`` so a wrapped line's tail is reachable.

    We can't introspect the Paragraph's scroll state from
    ``__repr__``, so we drive ``_compute_log_window`` directly with
    the same geometry the renderer would compute and pin its
    ``scroll_y`` return — that's the value the renderer plumbs into
    ``Paragraph.scroll``. Without rendered-row scrolling, this would
    always be 0 and the rest of the wrapped line would be clipped.
    """
    long_line = "x" * 500
    lines = [long_line]

    # Width 50 → 10 rendered rows. Height 5, scroll_offset 4 (PageUp
    # from tail): visible rows are [1, 6).
    visible, scroll_y = _compute_log_window(
        lines, width=50, height=5, scroll_offset=4, follow=False,
    )
    assert visible == [long_line]
    assert scroll_y == 1, (
        "operator scrolled 4 rendered rows back from tail; the "
        "Paragraph must scroll past row 0 to reveal the later "
        "wrapped rows. scroll_y was 0 instead, which is the bug."
    )

    # All the way to the top: scroll_offset = total - 1 = 9. Visible
    # window starts at row 0.
    visible, scroll_y = _compute_log_window(
        lines, width=50, height=5, scroll_offset=9, follow=False,
    )
    assert visible == [long_line]
    assert scroll_y == 0, (
        "scrolling to the top must position the Paragraph at row 0"
    )

    # In follow mode the latest rendered rows are visible: scroll_y
    # is 5 (total 10 - height 5).
    visible, scroll_y = _compute_log_window(
        lines, width=50, height=5, scroll_offset=0, follow=True,
    )
    assert visible == [long_line]
    assert scroll_y == 5, (
        "follow mode must position the Paragraph so the last "
        "rendered row of the buffer is at the panel bottom"
    )


def test_compute_log_window_handles_mixed_short_and_long_lines():
    """A buffer mixing short and long lines must compose correctly:
    the short lines occupy 1 rendered row each and the long lines
    occupy multiple, and the visible slice plus ``scroll_y`` together
    cover exactly the requested rendered-row window.

    This is the real shape of an agent transcript — short status
    lines interleaved with multi-line tool output. The previous
    logical-line slicing under-counted the long lines, so a paused
    operator scrolled by 50 logical lines ended up looking at very
    different content than they expected."""
    short = "abc"
    long = "y" * 100  # at width 50 → 2 rendered rows
    lines = [short, short, long, short, long, short]
    # Rendered rows per logical line:
    #   short=1, short=1, long=2, short=1, long=2, short=1 → total 8.

    assert _total_rendered_rows(lines, 50) == 8

    # follow=True, height=4 → last 4 rendered rows: rows 4..7.
    # Logical lines: long (rows 4-5) + short (row 6) + ... wait,
    # cumulative: short 0..1, short 1..2, long 2..4, short 4..5,
    # long 5..7, short 7..8. Visible rows 4..8 cover indices 3,4,5
    # → [short, long, short]. scroll_y = 4 - cum_at_first(idx 3) =
    # 4 - 4 = 0.
    visible, scroll_y = _compute_log_window(
        lines, width=50, height=4, scroll_offset=0, follow=True,
    )
    assert visible == [short, long, short]
    assert scroll_y == 0

    # paused, scroll all the way back: target_top = 0.
    visible, scroll_y = _compute_log_window(
        lines, width=50, height=4, scroll_offset=8, follow=False,
    )
    assert visible[:3] == [short, short, long]
    assert scroll_y == 0


def test_render_run_detail_long_line_paged_up_scrolls_paragraph():
    """End-to-end pin: rendering after PageUp on a long-line buffer
    must move the visible window away from the tail.

    With logical-line scroll semantics (the bug), ``action_page_up``
    short-circuited at ``len(_lines) <= 1`` so the rendered Paragraph
    stayed pinned to follow-mode tail no matter how many times the
    operator pressed PageUp. Here we force enough wrapped rows that
    PageUp lands at an intermediate offset, then assert the
    renderer's ``scroll_y`` shifted off the follow position."""
    import pyratatui as pr

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run_dir(Path(tmp))
        # 2000 chars at width 48 (50 - 2 border) → ~42 rendered
        # rows; way more than the 10-row viewport, so PageUp must
        # land somewhere short of the head.
        long_line = "z" * 2000
        paths.agent_log.write_text(long_line + "\n", encoding="utf-8")
        app = RunDetailApp(
            paths, refresh_seconds=0.1, initial_log_lines=5,
        )

        # First paint sets the viewport geometry on the screen so
        # subsequent action clamps know the width — exactly what the
        # live loop does.
        frame = _make_stub_frame(width=50, height=12)
        _render_run_detail(app.screen, frame, pr)

        log_height = max(1, frame.area.height - 4)
        log_width = max(1, frame.area.width - 2)

        # Capture what follow mode would scroll to (the tail of the
        # wrapped output), then scroll up by a page.
        _, follow_scroll_y = _compute_log_window(
            list(app.screen._lines), log_width, log_height,
            0, True,
        )
        assert follow_scroll_y > 0, (
            "follow mode on a long wrapped line must scroll the "
            "Paragraph to its tail rendered row — otherwise the "
            "scroll path is dead code"
        )

        app.screen.action_page_up()
        assert app.screen._scroll_offset > 0, (
            "PageUp must advance scroll_offset on a wrapped long "
            "line — this is the bug the reviewer reproduced"
        )
        assert app.screen._follow is False

        # Re-render and re-derive the renderer's scroll_y.
        frame2 = _make_stub_frame(width=50, height=12)
        _render_run_detail(app.screen, frame2, pr)
        body, paused_scroll_y = _compute_log_window(
            list(app.screen._lines), log_width, log_height,
            app.screen._scroll_offset, app.screen._follow,
        )
        assert body == [long_line]
        # Critically: paused scroll position is *different* from the
        # follow-tail position. With the logical-line bug, both would
        # be identical because PageUp couldn't change anything.
        assert paused_scroll_y != follow_scroll_y, (
            "after PageUp, the renderer must position the Paragraph "
            "at a different rendered-row offset than follow mode; "
            f"got paused={paused_scroll_y} follow={follow_scroll_y}"
        )
        assert paused_scroll_y < follow_scroll_y, (
            "PageUp must move the visible window *up* (towards the "
            "head), so the rendered-row offset must shrink"
        )


def test_rendered_rows_for_floors_at_one():
    """``_rendered_rows_for`` must never return 0 — even an empty
    line takes up one rendered row. Otherwise the cumulative-rows
    walk in :func:`_compute_log_window` could place two logical
    lines at the same row offset and corrupt the visible-window
    derivation."""
    assert _rendered_rows_for("", 80) == 1
    assert _rendered_rows_for("a", 80) == 1
    assert _rendered_rows_for("a" * 80, 80) == 1
    assert _rendered_rows_for("a" * 81, 80) == 2
    assert _rendered_rows_for("a" * 240, 80) == 3
    # Degenerate width: must not divide by zero.
    assert _rendered_rows_for("anything", 0) == 1


# ── Ctrl-C / Ctrl-D unconditional-exit contract ────────────────────────────


def test_ctrl_c_exits_run_list_app_immediately():
    """``Ctrl-C`` must exit the run-list app even though the binding
    map doesn't list it.

    pyratatui delivers Ctrl-C as a regular ``KeyEvent(code="c",
    ctrl=True)`` once the terminal is in raw mode (the kernel does
    *not* synthesize SIGINT for raw-mode terminals). The App layer
    must short-circuit that into an exit so the operator can always
    bail out, matching the UX of ``claude code`` and ``codex``."""
    rows = [_make_run_row()]
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=rows,
        ):
            app = RunListApp(Path(tmp))
            assert app._exit is False
            _press(app, "c", ctrl=True)
            assert app._exit is True, (
                "Ctrl-C in the run-list app must exit unconditionally"
            )


def test_ctrl_c_exits_run_detail_app_immediately():
    """Same contract on the per-run detail screen — the binding map
    there also omits Ctrl-C, so the App-level short-circuit is the
    only thing keeping the operator from getting stuck in the
    alternate screen with no way out."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run_dir(Path(tmp))
        app = RunDetailApp(paths, refresh_seconds=0.1)
        assert app._exit is False
        _press(app, "c", ctrl=True)
        assert app._exit is True, (
            "Ctrl-C in the run-detail app must exit unconditionally"
        )


def test_ctrl_c_exits_even_with_a_modal_open():
    """A modal on top of the run-list must not swallow Ctrl-C.

    The reviewer's reproduction: an operator stuck inside a prompt
    modal (e.g. they hit ``n`` to start a new run, then changed
    their mind) needs to be able to abort the whole TUI with
    Ctrl-C, not just cancel the modal. The App-layer short-circuit
    fires *before* the modal's ``handle_key``."""
    rows = [_make_run_row()]
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=rows,
        ):
            app = RunListApp(Path(tmp))
            _press(app, "n")  # opens the new-run prompt modal
            scr = app.screen
            assert scr.modals, (
                "precondition: ``n`` must open a modal, otherwise this "
                "test isn't exercising the modal-overrides-Ctrl-C path"
            )
            _press(app, "c", ctrl=True)
            assert app._exit is True, (
                "Ctrl-C must exit the app even when a modal is open"
            )


def test_ctrl_d_exits_run_list_app_immediately():
    """``Ctrl-D`` is bound the same way as Ctrl-C — both are the
    canonical "get me out of here" shortcut and the operator should
    not have to remember which one ends the alternate-screen
    session."""
    rows = [_make_run_row()]
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=rows,
        ):
            app = RunListApp(Path(tmp))
            _press(app, "d", ctrl=True)
            assert app._exit is True


def test_plain_c_does_not_exit_the_app():
    """Sanity check: a bare ``c`` (no ctrl) is *not* a binding on the
    run list, but it also must not leak through the Ctrl-C
    short-circuit. Otherwise typing ``c`` inside a future modal
    would unexpectedly tear down the whole app."""
    rows = [_make_run_row()]
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=rows,
        ):
            app = RunListApp(Path(tmp))
            _press(app, "c")  # no ctrl=True
            assert app._exit is False, (
                "Plain 'c' must not exit; only ctrl-c does"
            )


# ── refresh_rows must pin selection by run_id, not by index ─────────────────


def test_refresh_rows_pins_cursor_to_same_run_id_when_newer_run_appears():
    """A new run inserted above the selection must not silently
    retarget the cursor to a different ``run_id``.

    Reviewer's reproduction: ``list_runs`` is sorted newest-first, so
    when a fresh run appears it goes to row 0 and shifts every other
    run down by one. The previous numeric-clamp ``refresh_rows``
    would leave ``cursor_row`` pointing at the same *index*, which
    now refers to a different run — and any subsequent destructive
    verb (``k`` / ``R`` / ``a`` / ``v``) would hit the wrong run.
    """
    initial = [
        _make_run_row("20260430T101010Z-aaa"),
        _make_run_row("20260430T101111Z-bbb"),
        _make_run_row("20260430T101212Z-ccc"),
    ]
    after_insert = [
        _make_run_row("20260430T101313Z-ddd"),  # newer, lands at row 0
    ] + initial

    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=initial,
        ):
            app = RunListApp(Path(tmp))
        scr = app.screen
        scr.cursor_row = 1
        selected_run_id = scr.rows[scr.cursor_row].run_id
        assert selected_run_id == "20260430T101111Z-bbb"

        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=after_insert,
        ):
            scr.refresh_rows()

        assert scr.rows[scr.cursor_row].run_id == selected_run_id, (
            "cursor must follow the same run_id across the structural "
            "change, not stay clamped to its previous numeric index"
        )
        assert scr.cursor_row == 2


def test_refresh_rows_falls_back_to_index_clamp_when_selected_run_disappears():
    """If the previously-selected run is gone (e.g. removed by
    ``worktree-remove``), there's no run_id to pin to. We fall back
    to clamping the old numeric index into the new range — that
    lands the cursor on the neighbour just below the deleted row,
    which matches the UX of any list editor.
    """
    initial = [
        _make_run_row("20260430T101010Z-aaa"),
        _make_run_row("20260430T101111Z-bbb"),
        _make_run_row("20260430T101212Z-ccc"),
    ]
    after_delete = [
        _make_run_row("20260430T101010Z-aaa"),
        _make_run_row("20260430T101212Z-ccc"),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=initial,
        ):
            app = RunListApp(Path(tmp))
        scr = app.screen
        scr.cursor_row = 1  # bbb

        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=after_delete,
        ):
            scr.refresh_rows()

        assert scr.cursor_row == 1
        assert scr.rows[scr.cursor_row].run_id == "20260430T101212Z-ccc"


def test_refresh_rows_clamps_cursor_when_runs_shrink_past_old_index():
    """Old index past the new tail must clamp to the new last row,
    rather than leaving the cursor dangling in undefined territory."""
    initial = [
        _make_run_row("20260430T101010Z-aaa"),
        _make_run_row("20260430T101111Z-bbb"),
        _make_run_row("20260430T101212Z-ccc"),
    ]
    after_delete = [
        _make_run_row("20260430T101010Z-aaa"),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=initial,
        ):
            app = RunListApp(Path(tmp))
        scr = app.screen
        scr.cursor_row = 2  # ccc

        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=after_delete,
        ):
            scr.refresh_rows()

        assert scr.cursor_row == 0
        assert scr.rows[scr.cursor_row].run_id == "20260430T101010Z-aaa"


def test_refresh_rows_resets_cursor_when_all_runs_disappear():
    """Empty list → cursor 0 so the next non-empty refresh starts
    from the top instead of inheriting a stale numeric offset."""
    initial = [
        _make_run_row("20260430T101010Z-aaa"),
        _make_run_row("20260430T101111Z-bbb"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=initial,
        ):
            app = RunListApp(Path(tmp))
        scr = app.screen
        scr.cursor_row = 1

        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[],
        ):
            scr.refresh_rows()

        assert scr.rows == []
        assert scr.cursor_row == 0


def test_refresh_rows_preserves_run_id_when_runs_reorder():
    """If ``list_runs`` returns the same set in a different order
    (e.g. an older run got a fresh heartbeat and bubbled up by
    ``updated_at``), the cursor should follow its run by ``run_id``
    rather than stay glued to the original index."""
    initial = [
        _make_run_row("20260430T101010Z-aaa"),
        _make_run_row("20260430T101111Z-bbb"),
        _make_run_row("20260430T101212Z-ccc"),
    ]
    reordered = [
        _make_run_row("20260430T101212Z-ccc"),
        _make_run_row("20260430T101010Z-aaa"),
        _make_run_row("20260430T101111Z-bbb"),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=initial,
        ):
            app = RunListApp(Path(tmp))
        scr = app.screen
        scr.cursor_row = 0  # aaa

        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=reordered,
        ):
            scr.refresh_rows()

        assert scr.rows[scr.cursor_row].run_id == "20260430T101010Z-aaa"
        assert scr.cursor_row == 1
