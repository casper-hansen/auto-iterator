"""Smoke tests for the Textual TUI built on the ``Pilot`` async harness.

These tests do not exercise the full end-to-end UX (we don't render a
real terminal); they pin the contracts the task spec calls out:

* The run-list screen renders one row per :class:`RunRow`.
* Pressing ``Enter`` on a row pushes the per-run detail screen.
* Pressing ``s`` opens the send-guidance modal; submitting it writes
  ``<run_dir>/control/guidance.txt`` with the typed text.
* Pressing ``q`` exits the app cleanly without signalling any pid.

The harness uses Textual's :class:`Pilot` API which simulates key
presses against a headless app. A ``run_test`` context tears the app
down on exit, so even a hung test never blocks the suite.
"""

from __future__ import annotations

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


# ── RunListScreen smoke tests ───────────────────────────────────────────────


async def test_run_list_renders_one_row_per_run():
    """``list_runs`` mocked → DataTable shows one row per run."""
    from auto_iterator.tui import RunListApp

    rows = [
        _make_run_row("20260430T101010Z-aaa"),
        _make_run_row("20260430T101111Z-bbb"),
        _make_run_row("20260430T101212Z-ccc"),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        app = RunListApp(Path(tmp))
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=rows,
        ):
            async with app.run_test() as pilot:
                # ``set_interval`` schedules the periodic refresh; the
                # delay gives the initial ``refresh_rows`` from
                # ``on_mount`` time to populate the table.
                await pilot.pause(0.05)
                table = app.screen.query_one("DataTable")
                assert table.row_count == 3
                # All three run IDs land in the first column.
                row_keys = {
                    table.get_row_at(i)[0] for i in range(table.row_count)
                }
                assert row_keys == {r.run_id for r in rows}


async def test_pressing_enter_pushes_detail_screen():
    """Selecting a row opens :class:`RunDetailScreen`."""
    from auto_iterator.tui import RunDetailScreen, RunListApp

    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        paths = _seed_run_dir(runs_dir)
        row = _make_run_row(paths.run_id)
        app = RunListApp(runs_dir)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[row],
        ):
            async with app.run_test() as pilot:
                # Small delay so ``refresh_rows`` populates the table
                # before the keypress.
                await pilot.pause(0.05)
                await pilot.press("enter")
                await pilot.pause(0.05)
                # Top of the screen stack is the detail screen.
                assert isinstance(app.screen, RunDetailScreen)
                assert app.screen.paths.run_id == paths.run_id


async def test_pressing_enter_seeds_full_agent_log():
    """The press-Enter path seeds the *entire* existing transcript.

    Reviewer pin: previously the run-list pushed
    ``RunDetailScreen(paths)`` with the default ``initial_log_lines=30``,
    so older log lines were permanently absent from the screen even
    though the original task asked for "the full raw logs in one
    screen". The fix is to push with ``initial_log_lines=None`` so
    ``_seed_initial_log`` streams the whole file before parking the
    tailer at EOF.

    We seed with 200 lines (well above the previous 30-line cap) and
    assert that every single one is rendered into the ``RichLog``."""
    from auto_iterator.tui import RunDetailScreen, RunListApp

    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        paths = _seed_run_dir(runs_dir)
        # 200 unique lines so we can detect any line being dropped.
        seed_lines = [f"agent line {i:04d}" for i in range(200)]
        paths.agent_log.write_text("\n".join(seed_lines) + "\n", encoding="utf-8")

        row = _make_run_row(paths.run_id)
        app = RunListApp(runs_dir)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[row],
        ):
            async with app.run_test() as pilot:
                await pilot.pause(0.05)
                await pilot.press("enter")
                # ``on_mount`` runs synchronously after push; a tiny
                # pause lets the seed render before we inspect.
                await pilot.pause(0.1)
                assert isinstance(app.screen, RunDetailScreen)
                # Sentinel: the screen knows it should render the full
                # log (not a bounded tail).
                assert app.screen.initial_log_lines is None, (
                    "press-Enter must request the full transcript, "
                    "not the bounded tail"
                )

                # Every seeded line landed in the RichLog. We compare
                # rendered Strip text rather than relying on widget
                # internals so the assertion is robust across Textual
                # versions.
                log_widget = app.screen._log_widget
                assert log_widget is not None
                rendered = "\n".join(
                    str(line) for line in log_widget.lines
                )
                # First, last, and a middle line — proving the seed
                # spans the whole file, not just a 30-line tail.
                assert "agent line 0000" in rendered, (
                    "the head of the log must be visible after Enter"
                )
                assert "agent line 0100" in rendered
                assert "agent line 0199" in rendered, (
                    "the tail of the log must be visible after Enter"
                )


async def test_run_detail_log_panel_wraps_long_lines():
    """``RunDetailScreen``'s agent-output panel must wrap long lines.

    Reviewer pin: the ``RichLog`` widget was instantiated with
    ``wrap=False``, so a single long agent line (e.g. a 500-character
    tool-call payload, a wide diff hunk) was clipped at the terminal's
    right edge with no horizontal-scroll affordance. The detail screen
    is the "watch the agent work" view, so readability of the raw
    transcript trumps the per-line one-row budget the non-TTY
    ``ai show`` view (``_truncate_visible``) cares about.

    We assert the widget property directly because it's the only
    public knob that controls wrapping; rendering inspection would
    couple to Textual's internal Strip layout."""
    from textual.widgets import RichLog

    from auto_iterator.tui import RunDetailApp

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run_dir(Path(tmp))
        # A line wider than any sane terminal so the wrap is visibly
        # meaningful — not just a 0/1 toggle on the widget.
        paths.agent_log.write_text(
            ("x" * 500) + "\n", encoding="utf-8",
        )

        app = RunDetailApp(paths, refresh_seconds=0.1, initial_log_lines=5)
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            log_widget = app.screen.query_one("#log-panel", RichLog)
            assert log_widget.wrap is True, (
                "the agent-output panel must wrap long lines so "
                "operators on narrow terminals can read the full "
                "transcript without horizontal clipping"
            )


async def test_run_detail_reflows_transcript_on_terminal_resize():
    """Resizing the terminal must re-flow already-rendered transcript
    lines at the new viewport width.

    Reviewer pin: vanilla :class:`RichLog` pre-wraps each ``write``
    into ``Strip`` objects at the width that was current at write
    time, then never re-wraps. Resizing the terminal narrower left
    the historical transcript stuck at the old (wider) wrap geometry
    — Textual exposed a horizontal scrollbar for the over-wide
    strips ("blue bar at the bottom"), and short lines that should
    have re-flowed onto multiple rows kept showing as single
    too-wide rows. The fix is :class:`_WrapAwareRichLog` mirroring
    the raw text and replaying it on a width change.

    We seed a long line, capture the post-mount strip count, then
    resize the terminal narrower and assert (a) the strip count
    *grew* — proving the re-flow actually re-wrapped — and (b) the
    widget's ``virtual_size.width`` is bounded by the new viewport
    width, so Textual won't paint the horizontal scrollbar."""
    from auto_iterator.tui import RunDetailApp, _WrapAwareRichLog

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run_dir(Path(tmp))
        # A 240-char line is comfortably wider than the new (60-col)
        # viewport — we expect it to wrap to ~4 strips after resize,
        # regardless of the framework-default measurement quirks
        # that influence the *initial* strip count under
        # ``run_test``.
        paths.agent_log.write_text(("x" * 240) + "\n", encoding="utf-8")

        app = RunDetailApp(paths, refresh_seconds=0.5, initial_log_lines=10)
        async with app.run_test(size=(250, 30)) as pilot:
            await pilot.pause(0.3)
            log_widget = app.screen.query_one(
                "#log-panel", _WrapAwareRichLog,
            )
            assert log_widget._raw_lines, (
                "seed must populate the resize-replay mirror; "
                "without it, post-resize content would vanish"
            )
            initial_strip_count = len(log_widget.lines)
            initial_virtual_width = log_widget.virtual_size.width

            # Resize the terminal narrower than the seeded line.
            # ``Pilot.resize_terminal`` drives the same SIGWINCH
            # path a real terminal would, dispatching ``Resize``
            # through Textual's screen tree — the path our
            # ``_WrapAwareRichLog.on_resize`` is wired into.
            await pilot.resize_terminal(60, 30)
            # Two ticks: one for the resize event to land and our
            # ``call_after_refresh`` to be queued, a second for the
            # queued ``_reflow_raw_lines`` to actually run after
            # layout settles.
            await pilot.pause(0.1)
            await pilot.pause(0.1)

            reflowed_strip_count = len(log_widget.lines)
            assert reflowed_strip_count > initial_strip_count, (
                "the 240-char line must re-flow into more strips "
                "after the terminal narrowed from 250 -> 60 cols; "
                f"strips went from {initial_strip_count} to "
                f"{reflowed_strip_count} — wrap did not rerun on "
                "resize"
            )
            # The widget must NOT think it has content wider than
            # the new viewport — that's the underlying condition
            # that paints the horizontal scrollbar (the "blue bar
            # at the bottom" the operator reported on resize).
            assert log_widget.virtual_size.width <= 60, (
                "after resize to width 60, "
                f"virtual_size.width is {log_widget.virtual_size.width}"
                "; the widget still thinks it has wider content "
                "than the viewport, which is what triggers the "
                "blue horizontal scrollbar"
            )
            assert log_widget.virtual_size.width < initial_virtual_width, (
                "virtual_size.width must shrink along with the "
                "viewport; otherwise the historical strips are "
                f"still at the old geometry ({initial_virtual_width} "
                f"-> {log_widget.virtual_size.width})"
            )


async def test_send_modal_writes_guidance_file():
    """Pressing ``s`` → modal → submit → ``control/guidance.txt`` written."""
    from auto_iterator.tui import RunListApp, _PromptModal

    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        paths = _seed_run_dir(runs_dir)
        row = _make_run_row(paths.run_id)
        app = RunListApp(runs_dir)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[row],
        ):
            async with app.run_test() as pilot:
                # Use a small delay so ``on_mount`` (and the
                # ``refresh_rows`` it triggers) finishes populating
                # the DataTable before we drive any keypresses.
                await pilot.pause(0.05)
                # Open the send-guidance modal.
                await pilot.press("s")
                await pilot.pause(0.05)
                assert isinstance(app.screen, _PromptModal), (
                    f"expected send modal, got {type(app.screen).__name__}"
                )
                from textual.widgets import Input

                entry = app.screen.query_one("#entry", Input)
                entry.value = "Focus on the failing assertion in foo_test"
                await pilot.press("enter")
                await pilot.pause(0.05)

            # Read the file *while* the temp dir still exists — the
            # ``tempfile.TemporaryDirectory`` cleanup at the end of the
            # outer ``with`` block removes the path tree before any
            # post-block assertions could see the file.
            guidance_file = paths.control_file("guidance.txt")
            assert guidance_file.exists(), "guidance.txt must be written"
            content = guidance_file.read_text(encoding="utf-8")
            assert "Focus on the failing assertion in foo_test" in content
            # File shape is ``<ISO8601>\t<text>\n`` so a tab is present.
            assert "\t" in content


async def test_quit_exits_without_signalling_any_pid(monkeypatch):
    """``q`` exits the TUI cleanly. ``os.kill`` must never be called.

    The TUI process must not own any runner lifecycles. We assert
    this by intercepting :func:`os.kill` in both ``actions`` and
    ``run_dir`` (which is what ``actions.signal_runner`` and
    ``pid_alive`` consult); the press-q path should never touch
    either."""
    kill_calls: list[tuple] = []

    def fake_kill(*args, **kwargs):
        kill_calls.append((args, kwargs))
        # Translate to ProcessLookupError so any path that did call us
        # would surface as a clean "no such process" rather than killing
        # the test runner.
        raise ProcessLookupError

    # Patch os.kill at the module-level imports the codepath uses.
    monkeypatch.setattr("auto_iterator.run_dir.os.kill", fake_kill)
    monkeypatch.setattr("auto_iterator.actions.os.kill", fake_kill)
    # Belt-and-braces: also patch the bare module import.
    monkeypatch.setattr(os, "kill", fake_kill)

    from auto_iterator.tui import RunListApp

    rows = [_make_run_row()]
    with tempfile.TemporaryDirectory() as tmp:
        app = RunListApp(Path(tmp))
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=rows,
        ):
            async with app.run_test() as pilot:
                await pilot.pause(0.05)
                await pilot.press("q")
                await pilot.pause(0.05)
            # After the harness exits we can sanity-check the app
            # state — ``run_test`` only returns once the app has
            # genuinely exited.
            assert app._exit is True

    assert kill_calls == [], (
        f"TUI quit must not call os.kill, but it did: {kill_calls!r}"
    )


async def test_pause_writes_pause_file():
    """``p`` on a selected row drops ``control/pause`` immediately."""
    from auto_iterator.tui import RunListApp

    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        paths = _seed_run_dir(runs_dir)
        row = _make_run_row(paths.run_id)
        app = RunListApp(runs_dir)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[row],
        ):
            async with app.run_test() as pilot:
                # Small delay so ``refresh_rows`` populates the table
                # before the keypress; otherwise ``action_pause``
                # sees an empty table and bails early without writing.
                await pilot.pause(0.05)
                await pilot.press("p")
                await pilot.pause(0.05)

            # Assert *inside* the temp-dir context — once the outer
            # ``with`` block exits, the directory tree (including the
            # control file we just wrote) is gone.
            assert paths.control_file("pause").exists()


async def test_resume_clears_pause_file():
    """``r`` removes ``control/pause`` and tolerates the missing file."""
    from auto_iterator.tui import RunListApp

    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        paths = _seed_run_dir(runs_dir)
        actions.write_pause(paths)
        assert paths.control_file("pause").exists()
        row = _make_run_row(paths.run_id)
        app = RunListApp(runs_dir)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[row],
        ):
            async with app.run_test() as pilot:
                await pilot.pause(0.05)
                await pilot.press("r")
                await pilot.pause(0.05)

            # Inside the temp-dir context: the assertion would
            # otherwise race the directory cleanup.
            assert not paths.control_file("pause").exists()


async def test_rewind_modal_writes_rewind_file():
    """``w`` opens the rewind modal; submitting drops ``rewind.json``."""
    from auto_iterator.tui import RunListApp, _PromptModal

    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        paths = _seed_run_dir(runs_dir)
        row = _make_run_row(paths.run_id)
        app = RunListApp(runs_dir)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[row],
        ):
            async with app.run_test() as pilot:
                await pilot.pause(0.05)
                await pilot.press("w")
                await pilot.pause(0.05)
                assert isinstance(app.screen, _PromptModal)
                from textual.widgets import Input

                entry = app.screen.query_one("#entry", Input)
                entry.value = "outer=2,inner=3,phase=fix"
                await pilot.press("enter")
                await pilot.pause(0.05)

            # Inside the temp-dir context: the file is gone once we
            # leave it.
            rewind_file = paths.control_file("rewind.json")
            assert rewind_file.exists()
            import json

            payload = json.loads(rewind_file.read_text(encoding="utf-8"))
            assert payload == {"outer": 2, "inner": 3, "phase": "fix"}


async def test_run_detail_streams_log_lines_incrementally():
    """``RunDetailScreen`` reads only new bytes per tick (no full reload).

    Pin the spec'd "agent-log viewer never reads the whole file on
    refresh" property by inspecting the embedded :class:`LogTailer`'s
    offset before and after an append: the offset must only advance
    by the appended size, never re-read the whole file."""
    from auto_iterator.tui import RunDetailApp

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run_dir(Path(tmp))
        # Seed with 100 KiB of content so we can detect a re-read.
        seed = ("seed line\n" * 10240).encode("utf-8")
        paths.agent_log.write_bytes(seed)

        app = RunDetailApp(paths, refresh_seconds=0.1, initial_log_lines=5)
        async with app.run_test() as pilot:
            await pilot.pause()
            # The screen seeded the initial tail; the tailer should now
            # be at EOF so further appends are bounded.
            screen = app.screen
            initial_offset = screen._tailer.offset
            assert initial_offset == paths.agent_log.stat().st_size, (
                "initial seed must advance the tailer offset to EOF"
            )

            # Append 1 KiB of new content.
            new_chunk = ("delta line\n" * 100).encode("utf-8")
            with paths.agent_log.open("ab") as fh:
                fh.write(new_chunk)

            # Trigger one refresh cycle by waiting longer than the
            # log poll interval. The screen paces the poll at 0.4 s
            # (set in ``on_mount``); 0.5 s gives a comfortable margin
            # without making the test noticeably slower.
            await pilot.pause(0.5)

            # The tailer's offset advanced by exactly the appended
            # bytes — proving we never re-read the whole file.
            assert screen._tailer.offset == paths.agent_log.stat().st_size
            advance = screen._tailer.offset - initial_offset
            assert advance == len(new_chunk), (
                f"tailer advanced by {advance} bytes; expected "
                f"{len(new_chunk)}"
            )


async def test_run_detail_seed_skips_to_eof_on_huge_log():
    """A multi-MiB pre-existing log must seed at EOF, not part-way.

    Reviewer pin: the previous implementation called
    ``LogTailer.read_new_lines()`` to "burn" the file after rendering
    the bounded tail, which only advances the offset by the per-tick
    cap (≈4 MiB). On larger files the next tick would surface old
    bytes as if they were new, polluting the agent-output panel with
    historical content.

    With ``seek_to_end`` the offset lands at exactly ``st_size`` on
    mount regardless of how big the file is."""
    from auto_iterator.tui import RunDetailApp

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run_dir(Path(tmp))
        # 8 MiB — over the per-tick read cap.
        payload = (b"x" * 4095 + b"\n") * 2048
        paths.agent_log.write_bytes(payload)
        size = paths.agent_log.stat().st_size

        app = RunDetailApp(paths, refresh_seconds=0.1, initial_log_lines=5)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert screen._tailer.offset == size, (
                "seed must park the tailer offset at EOF for huge logs; "
                f"offset={screen._tailer.offset}, size={size}"
            )
            # No new bytes → no advance on subsequent ticks.
            await pilot.pause(0.5)
            assert screen._tailer.offset == size


async def test_log_follow_pins_when_user_scrolls_away():
    """Scrolling up via the widget's own scroll API must pin the
    viewport: incoming appends do not yank the operator back to EOF.

    Reviewer pin: ``_follow`` was previously only flipped by the
    custom ``j``/``k``/``g``/``G`` actions, so mouse-wheel /
    PageUp / Home scrolling left ``_follow`` at ``True`` and the next
    refresh auto-scrolled back to the bottom. The fix is to derive
    ``_follow`` from the widget's actual ``is_vertical_scroll_end``
    state on every tick."""
    from auto_iterator.tui import RunDetailApp

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run_dir(Path(tmp))
        # Enough lines that the widget actually has somewhere to
        # scroll *up* into when we ask it to.
        paths.agent_log.write_bytes(("seed line\n" * 200).encode("utf-8"))

        app = RunDetailApp(paths, refresh_seconds=0.1, initial_log_lines=200)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            log_widget = screen._log_widget
            assert log_widget is not None
            # Make sure the widget thinks it's at the end before we
            # scroll away — the seed populated it.
            assert log_widget.is_vertical_scroll_end

            # Scroll up via the widget API (simulates a mouse-wheel
            # scroll the operator might perform). ``animate=False``
            # makes the change synchronous so the next tick sees the
            # new offset deterministically.
            log_widget.scroll_home(animate=False)
            await pilot.pause(0.05)
            assert not log_widget.is_vertical_scroll_end, (
                "test setup: the widget must be off the tail after "
                "scroll_home so the follow-pin assertion is meaningful"
            )

            # Append new content.
            with paths.agent_log.open("ab") as fh:
                fh.write(("delta line\n" * 50).encode("utf-8"))
            await pilot.pause(0.5)

            # ``_follow`` is re-derived per tick: scrolled-away
            # viewer → follow off → no auto-scroll.
            assert screen._follow is False
            assert log_widget.auto_scroll is False
            assert not log_widget.is_vertical_scroll_end, (
                "appending while scrolled up must not snap the viewport "
                "back to the bottom"
            )


async def test_log_follow_resumes_when_user_returns_to_bottom():
    """Scrolling back to EOF re-enables auto-follow — the inverse of
    the pin behavior. Without this, an operator who scrolled up to
    inspect history would have to press ``f`` to resume tailing."""
    from auto_iterator.tui import RunDetailApp

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run_dir(Path(tmp))
        paths.agent_log.write_bytes(("seed line\n" * 200).encode("utf-8"))

        app = RunDetailApp(paths, refresh_seconds=0.1, initial_log_lines=200)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            log_widget = screen._log_widget
            assert log_widget is not None
            log_widget.scroll_home(animate=False)
            await pilot.pause(0.05)

            # Drive a tick with an append so ``_follow`` flips off.
            with paths.agent_log.open("ab") as fh:
                fh.write(b"one\n")
            await pilot.pause(0.5)
            assert screen._follow is False

            # Now the operator scrolls back to the bottom (mouse-end /
            # PageDown / End).
            log_widget.scroll_end(animate=False)
            await pilot.pause(0.05)
            assert log_widget.is_vertical_scroll_end

            # The next tick re-derives ``_follow`` from the viewport
            # and resumes tailing.
            with paths.agent_log.open("ab") as fh:
                fh.write(b"two\n")
            await pilot.pause(0.5)
            assert screen._follow is True
            assert log_widget.auto_scroll is True


async def test_send_rejects_dead_runner(monkeypatch):
    """``s`` must refuse to drop ``guidance.txt`` for a run whose
    runner is gone, mirroring the CLI's ``_drop_mutation`` gate.

    Reviewer pin: the TUI used to write the guidance file
    unconditionally, leaving stale control files for runs the CLI
    would have rejected. The two front-ends must agree on the
    liveness gate so the on-disk state is consistent."""
    from auto_iterator.tui import RunListApp, _PromptModal

    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        paths = _seed_run_dir(runs_dir)
        # Mark the run as exited so ``runner_is_alive`` returns False.
        from auto_iterator.meta import update_meta

        update_meta(paths, status="exited")

        row = _make_run_row(paths.run_id)
        app = RunListApp(runs_dir)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[row],
        ):
            async with app.run_test() as pilot:
                await pilot.pause(0.05)
                await pilot.press("s")
                await pilot.pause(0.05)
                assert isinstance(app.screen, _PromptModal)
                from textual.widgets import Input

                entry = app.screen.query_one("#entry", Input)
                entry.value = "Should be rejected"
                await pilot.press("enter")
                await pilot.pause(0.05)

            # Guidance file must NOT exist — the TUI should have
            # refused to write it because the runner is exited.
            guidance_file = paths.control_file("guidance.txt")
            assert not guidance_file.exists(), (
                "TUI must reject guidance writes for dead runners "
                "(matching the CLI's _drop_mutation gate)"
            )


async def test_rewind_rejects_dead_runner(monkeypatch):
    """``w`` must refuse to drop ``rewind.json`` for a dead runner.

    Same liveness gate as ``send`` — keeps the protocol consistent
    across both control-file mutators."""
    from auto_iterator.tui import RunListApp, _PromptModal

    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        paths = _seed_run_dir(runs_dir)
        from auto_iterator.meta import update_meta

        update_meta(paths, status="killed")

        row = _make_run_row(paths.run_id)
        app = RunListApp(runs_dir)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[row],
        ):
            async with app.run_test() as pilot:
                await pilot.pause(0.05)
                await pilot.press("w")
                await pilot.pause(0.05)
                assert isinstance(app.screen, _PromptModal)
                from textual.widgets import Input

                entry = app.screen.query_one("#entry", Input)
                entry.value = "outer=1,inner=1,phase=review"
                await pilot.press("enter")
                await pilot.pause(0.05)

            rewind_file = paths.control_file("rewind.json")
            assert not rewind_file.exists(), (
                "TUI must reject rewind writes for dead runners"
            )


async def test_new_run_cursor_preset_pins_backend(monkeypatch):
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
    from auto_iterator.tui import (
        RunListApp,
        _BackendChoiceModal,
        _PromptModal,
    )
    from auto_iterator.backends import BACKENDS

    # Hostile env: shell points at claude-code, exports a per-phase
    # reviewer override, and overrides the global agent_cmd. The
    # preset must beat all of them.
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
        app = RunListApp(runs_dir)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[],
        ), mock.patch(
            "auto_iterator.tui.actions.spawn_runner_detached",
            side_effect=fake_spawn,
        ):
            async with app.run_test() as pilot:
                await pilot.pause(0.05)
                await pilot.press("n")
                await pilot.pause(0.05)
                assert isinstance(app.screen, _PromptModal)
                from textual.widgets import Input

                app.screen.query_one("#entry", Input).value = "Task"
                await pilot.press("enter")
                await pilot.pause(0.05)
                assert isinstance(app.screen, _PromptModal)
                app.screen.query_one("#entry", Input).value = str(runs_dir)
                await pilot.press("enter")
                await pilot.pause(0.05)
                assert isinstance(app.screen, _BackendChoiceModal)
                await pilot.press("1")
                await pilot.pause(0.05)

    assert "cfg" in captured
    cfg = captured["cfg"]
    assert cfg.backend == "cursor", (
        f"Cursor preset must pin backend=cursor regardless of env; got {cfg.backend!r}"
    )
    # Per-phase backends collapse to the global cursor backend.
    assert cfg.backend_for("impl") == "cursor"
    assert cfg.backend_for("fix") == "cursor"
    assert cfg.backend_for("reviewer") == "cursor"
    assert cfg.has_mixed_backends is False
    # The preset must also ignore $AGENT_CMD / $AGENT_REVIEWER_CMD —
    # otherwise the resolved binary is a lie relative to the modal.
    cursor_default = BACKENDS["cursor"].default_cmd
    assert cfg.agent_cmd == cursor_default, (
        f"Cursor preset must ignore $AGENT_CMD; "
        f"got agent_cmd={cfg.agent_cmd!r}"
    )
    assert cfg.impl_agent_cmd is None
    assert cfg.fix_agent_cmd is None
    assert cfg.reviewer_agent_cmd is None


async def test_new_run_claude_codex_preset_picks_mixed_backends(monkeypatch):
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
    from auto_iterator.tui import (
        RunListApp,
        _BackendChoiceModal,
        _PromptModal,
    )
    from auto_iterator.backends import BACKENDS

    # Hostile env from the prior review: every per-phase knob is set
    # to something the preset must ignore.
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
        app = RunListApp(runs_dir)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[],
        ), mock.patch(
            "auto_iterator.tui.actions.spawn_runner_detached",
            side_effect=fake_spawn,
        ):
            async with app.run_test() as pilot:
                await pilot.pause(0.05)
                await pilot.press("n")
                await pilot.pause(0.05)
                from textual.widgets import Input

                app.screen.query_one("#entry", Input).value = "Task"
                await pilot.press("enter")
                await pilot.pause(0.05)
                app.screen.query_one("#entry", Input).value = str(runs_dir)
                await pilot.press("enter")
                await pilot.pause(0.05)
                assert isinstance(app.screen, _BackendChoiceModal)
                await pilot.press("2")
                await pilot.pause(0.05)

    assert "cfg" in captured
    cfg = captured["cfg"]
    assert cfg.backend == "claude-code"
    assert cfg.backend_for("impl") == "claude-code"
    assert cfg.backend_for("fix") == "claude-code"
    assert cfg.backend_for("reviewer") == "codex"
    assert cfg.has_mixed_backends is True
    # Cmds must come from the resolved backend defaults — not env.
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


async def test_new_run_backend_modal_cancel_skips_spawn(monkeypatch):
    """Pressing Esc on the backend picker must abort the new-run flow
    without spawning a runner. Pin the cancel-path symmetry with the
    earlier prompt/workspace modals so a half-typed flow can be undone
    at any step."""
    from auto_iterator.tui import (
        RunListApp,
        _BackendChoiceModal,
        _PromptModal,
    )

    spawn_calls: list = []

    def fake_spawn(*args, **kwargs):
        spawn_calls.append((args, kwargs))
        from auto_iterator.actions import ActionResult

        return ActionResult(ok=True, run_id="should-not-happen")

    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        app = RunListApp(runs_dir)
        with mock.patch(
            "auto_iterator.tui.list_runs", return_value=[],
        ), mock.patch(
            "auto_iterator.tui.actions.spawn_runner_detached",
            side_effect=fake_spawn,
        ):
            async with app.run_test() as pilot:
                await pilot.pause(0.05)
                await pilot.press("n")
                await pilot.pause(0.05)
                from textual.widgets import Input

                assert isinstance(app.screen, _PromptModal)
                app.screen.query_one("#entry", Input).value = "Task"
                await pilot.press("enter")
                await pilot.pause(0.05)
                assert isinstance(app.screen, _PromptModal)
                app.screen.query_one("#entry", Input).value = str(runs_dir)
                await pilot.press("enter")
                await pilot.pause(0.05)
                assert isinstance(app.screen, _BackendChoiceModal)
                await pilot.press("escape")
                await pilot.pause(0.05)

    assert spawn_calls == [], (
        "Cancelling the backend picker must abort the new-run flow"
    )


# ── Plain (non-async) TUI helper tests ──────────────────────────────────────


def test_row_cells_shape():
    """One :class:`RunRow` becomes the seven DataTable columns."""
    from auto_iterator.tui import _row_cells

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
    ANSI escapes; the TUI must render them as Rich text, not raw bytes.
    """
    from auto_iterator.tui import _strip_ansi

    raw = "\x1b[1mRun foo\x1b[0m\nstatus  \x1b[32mrunning\x1b[0m"
    out = _strip_ansi(raw)
    assert "\x1b" not in out
    assert "Run foo" in out
    assert "running" in out
