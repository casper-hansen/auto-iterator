"""Dispatch contracts for ``ai show``.

The big task-spec promise is that the **stateless / scriptable** flags —
``--json``, ``--once``, ``--logs``, and the non-TTY auto-fallback — keep
working byte-identically and *do not* import :mod:`pyratatui`. Importing
pyratatui on every ``ai ls`` invocation would bolt the native-binding
startup cost onto operator commands that have no business with a TUI.

These tests cover the dispatch surface specifically:

* ``--json`` → calls ``state_json_text``, emits parseable JSON, never
  imports ``pyratatui``.
* ``--once`` → calls ``render_combined_view``, emits the snapshot, never
  imports ``pyratatui``.
* Non-TTY default → behaves like ``--once``, never imports ``pyratatui``.
* ``--logs`` → documented internal alias for ``--once``.

Each "no pyratatui import" check uses ``importlib.invalidate_caches``
plus :func:`sys.modules.pop` so the test is meaningful even if a
previous test in the same process had already imported ``pyratatui``.

Historical note: the project used to be built on Textual, which had
the same "don't pay the TUI startup cost on scriptable flags" rule.
After the migration to pyratatui the contract is identical — only the
module name changed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_iterator.cli import EXIT_OK, main  # noqa: E402
from auto_iterator.events import EventLog, RunState  # noqa: E402
from auto_iterator.meta import write_meta  # noqa: E402
from auto_iterator.run_dir import (  # noqa: E402
    RunPaths,
    create_run_dir,
    new_run_id,
    now_iso,
)


# ── helpers ────────────────────────────────────────────────────────────────


def _seed_run(runs_dir: Path) -> RunPaths:
    """Match ``tests/test_cli_selector.py``'s seeded run shape."""
    paths = create_run_dir(runs_dir, new_run_id())
    write_meta(paths, {
        "run_id": paths.run_id,
        "pid": 999_999_999,
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
    state.last_verdict = "needs_fixes"
    log = EventLog(paths, state)
    log.emit("run_started", workspace="/tmp/ws")
    log.emit("inner_started", outer=1, inner=1)
    paths.agent_log.write_text(
        "first agent line\nsecond agent line\n", encoding="utf-8",
    )
    return paths


def _drop_tui_modules_from_sys_modules() -> None:
    """Reset ``sys.modules`` so a follow-up import is meaningful.

    Drops both the legacy ``textual`` namespace (kept around because
    parts of the test process may still have it imported transitively
    via tooling) and the ``pyratatui`` namespace that the live TUI
    binds against. Sub-modules of each TUI framework are left in
    place because re-importing them after a full purge is brittle —
    what we want to assert is "the dispatch path doesn't *trigger* a
    fresh import" of the top-level package, which is what a parent
    process running ``ai show --json`` from a fresh interpreter
    observes.

    NOTE: We deliberately do **not** drop ``auto_iterator.tui`` here.
    ``auto_iterator.tui`` lazy-imports ``pyratatui`` only inside
    ``_run_app_loop``, so its mere presence in ``sys.modules`` does
    not violate the "scriptable flags don't pay the TUI cold-start
    cost" contract. Dropping it, however, would leak across into
    ``tests/test_tui_smoke.py``: that module imports ``RunListApp``
    and friends at *collection* time, so a re-import here would
    yield two distinct module objects (the original one bound to
    ``RunListApp`` and a fresh one that ``mock.patch`` would patch),
    making the smoke tests' ``mock.patch("auto_iterator.tui.list_runs")``
    silently miss the class's bound reference. The reviewer caught
    this regression with a co-run of the two files; keeping the
    module pinned is the simplest fix that keeps both contracts."""
    for name in list(sys.modules):
        if name == "pyratatui" or name.startswith("pyratatui."):
            sys.modules.pop(name, None)
        elif name == "textual" or name.startswith("textual."):
            sys.modules.pop(name, None)


# Backwards-compat alias: older test scaffolding may still call this.
_drop_textual_from_sys_modules = _drop_tui_modules_from_sys_modules


# ── --json path ────────────────────────────────────────────────────────────


def test_show_json_emits_parseable_json_without_importing_tui(
    capsys, monkeypatch,
) -> None:
    """``ai show --json`` emits parseable JSON and never imports the TUI lib.

    This is the scriptable contract — anything that pipes
    ``ai show … --json`` through ``jq`` must keep working, and the
    pyratatui native-binding cold-start cost must not land on it."""
    _drop_tui_modules_from_sys_modules()

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        rc = main(["--runs-dir", tmp, "show", paths.run_id, "--json"])
        out = capsys.readouterr().out
        assert rc == EXIT_OK
        # Parses → no truncation, no live ANSI.
        payload = json.loads(out)
        # state.json shape: at minimum the run id round-trips.
        assert payload.get("run_id") == paths.run_id

    assert "pyratatui" not in sys.modules, (
        "ai show --json must not import pyratatui; "
        f"sys.modules contains: {[k for k in sys.modules if 'pyratatui' in k]}"
    )
    assert "textual" not in sys.modules, (
        "ai show --json must not import textual either"
    )


# ── --once path ────────────────────────────────────────────────────────────


def test_show_once_uses_render_combined_view_without_importing_tui(
    capsys, monkeypatch,
) -> None:
    """``ai show --once`` is byte-identical to ``render_combined_view``.

    The "no pyratatui import" half is the cheap-startup contract. The
    "byte-identical" half is the stronger contract: shells that already
    parse the snapshot (``grep``, ``awk``, log-aggregators) must see
    exactly today's bytes."""
    _drop_tui_modules_from_sys_modules()

    # Capture the call args render_combined_view was invoked with so we
    # can replay them and compare bytes.
    captured: dict = {}
    from auto_iterator import display as _display

    real_render = _display.render_combined_view

    def spy_render(paths, *, event_lines, log_lines):
        captured["args"] = (paths, event_lines, log_lines)
        return real_render(paths, event_lines=event_lines, log_lines=log_lines)

    monkeypatch.setattr(_display, "render_combined_view", spy_render)

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        rc = main([
            "--runs-dir", tmp, "show", paths.run_id,
            "--once", "--event-lines", "5", "--log-lines", "10",
        ])
        out = capsys.readouterr().out
        assert rc == EXIT_OK
        assert "args" in captured, (
            "render_combined_view must be the renderer for --once"
        )

        # Replay with the same args; must be byte-identical.
        replay_paths, ev, ll = captured["args"]
        expected = real_render(replay_paths, event_lines=ev, log_lines=ll)
        assert out == expected, (
            "ai show --once must be byte-identical to render_combined_view"
        )

    assert "pyratatui" not in sys.modules, (
        "ai show --once must not import pyratatui; "
        f"sys.modules contains: {[k for k in sys.modules if 'pyratatui' in k]}"
    )


def test_show_logs_alias_behaves_like_once(capsys, monkeypatch) -> None:
    """``ai show --logs`` is the documented internal alias for ``--once``.

    The CLI keeps the alias around for muscle memory; verify it still
    routes to ``render_combined_view`` and stays TUI-library-free."""
    _drop_tui_modules_from_sys_modules()

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))

        from auto_iterator import display as _display

        real_render = _display.render_combined_view
        rc = main([
            "--runs-dir", tmp, "show", paths.run_id, "--logs",
        ])
        out_via_logs = capsys.readouterr().out
        assert rc == EXIT_OK

        # Compare to the real renderer with the default args ``cmd_show``
        # uses (event_lines=12, log_lines=30).
        expected = real_render(paths, event_lines=12, log_lines=30)
        assert out_via_logs == expected

    assert "pyratatui" not in sys.modules


# ── --stream path ──────────────────────────────────────────────────────────


def test_show_stream_calls_stream_log_without_importing_tui(
    capsys, monkeypatch,
) -> None:
    """``ai show --stream`` is the SSH-friendly tail mode.

    Two contracts pinned here:

    1. **No pyratatui.** ``--stream`` exists *because* the TUI is too
       chatty over high-latency links; importing it would defeat the
       cold-start argument.
    2. **Routes to ``stream_log``.** The CLI must hand off to
       :func:`auto_iterator.display.stream_log` with the operator's
       ``--log-lines`` / ``--refresh`` arguments rather than falling
       back to the one-shot ``render_combined_view``.

    We force ``stdout.isatty()`` to ``True`` so the dispatcher can't
    short-circuit into the non-TTY ``--once`` branch — the whole
    point of ``--stream`` is that it overrides the TTY default.
    """
    _drop_tui_modules_from_sys_modules()
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

    captured: dict = {}

    def fake_stream(paths, *, log_lines, poll_seconds):
        captured["paths"] = paths
        captured["log_lines"] = log_lines
        captured["poll_seconds"] = poll_seconds
        return EXIT_OK

    from auto_iterator import display as _display

    monkeypatch.setattr(_display, "stream_log", fake_stream, raising=True)

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        rc = main([
            "--runs-dir", tmp, "show", paths.run_id,
            "--stream", "--log-lines", "42", "--refresh", "0.25",
        ])
        assert rc == EXIT_OK

    assert captured.get("paths") is not None, (
        "ai show --stream must route through display.stream_log"
    )
    assert captured["log_lines"] == 42
    assert abs(captured["poll_seconds"] - 0.25) < 1e-6

    assert "pyratatui" not in sys.modules, (
        "ai show --stream must not import pyratatui (that's the "
        "cold-start contract --stream exists to preserve over "
        "high-latency SSH); sys.modules contains: "
        f"{[k for k in sys.modules if 'pyratatui' in k]}"
    )


def test_show_stream_overrides_non_tty_fallback(capsys, monkeypatch) -> None:
    """``--stream`` with stdout *not* a TTY still follows.

    The dispatcher's non-TTY heuristic exists so a bare ``ai show <id>``
    piped through ``grep`` produces a single snapshot. But an explicit
    ``--stream`` is the operator saying "follow this, like ``tail -f``"
    — and ``tail -f`` follows even when piped. So ``--stream`` must
    take precedence over the non-TTY → ``--once`` fallback."""
    _drop_tui_modules_from_sys_modules()
    # Pytest's capsys gives us a non-TTY stdout already; assert it.
    assert not sys.stdout.isatty()

    routed = {"stream": False, "once": False}

    def fake_stream(paths, *, log_lines, poll_seconds):
        routed["stream"] = True
        return EXIT_OK

    def fake_render(paths, *, event_lines, log_lines):
        routed["once"] = True
        return ""

    from auto_iterator import display as _display

    monkeypatch.setattr(_display, "stream_log", fake_stream, raising=True)
    monkeypatch.setattr(_display, "render_combined_view", fake_render,
                        raising=True)

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        rc = main([
            "--runs-dir", tmp, "show", paths.run_id, "--stream",
        ])
        assert rc == EXIT_OK

    assert routed["stream"] is True, (
        "ai show --stream on non-TTY must call stream_log, not "
        "fall back to render_combined_view (--once)"
    )
    assert routed["once"] is False


def test_show_once_beats_stream(capsys, monkeypatch) -> None:
    """If the operator passes both ``--once`` and ``--stream``, ``--once``
    wins. A single snapshot is the more conservative interpretation —
    we don't want to silently turn a one-shot into a long-running
    follow loop just because both flags landed in muscle memory."""
    _drop_tui_modules_from_sys_modules()

    routed = {"stream": False, "once": False}

    def fake_stream(paths, *, log_lines, poll_seconds):
        routed["stream"] = True
        return EXIT_OK

    def fake_render(paths, *, event_lines, log_lines):
        routed["once"] = True
        return ""

    from auto_iterator import display as _display

    monkeypatch.setattr(_display, "stream_log", fake_stream, raising=True)
    monkeypatch.setattr(_display, "render_combined_view", fake_render,
                        raising=True)

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        rc = main([
            "--runs-dir", tmp, "show", paths.run_id, "--once", "--stream",
        ])
        assert rc == EXIT_OK

    assert routed["once"] is True
    assert routed["stream"] is False


# ── non-TTY default ────────────────────────────────────────────────────────


def test_show_non_tty_default_falls_back_to_once(capsys, monkeypatch) -> None:
    """Non-TTY stdout makes ``ai show <run_id>`` (no flags) emit the snapshot.

    The dispatch heuristic is ``not sys.stdout.isatty()`` → take the
    one-shot path. ``capsys`` already exposes a non-TTY stdout, so we
    don't have to fake it; we just assert the output is the combined
    view's bytes and pyratatui is absent.
    """
    _drop_tui_modules_from_sys_modules()

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))

        from auto_iterator import display as _display

        rc = main(["--runs-dir", tmp, "show", paths.run_id])
        out = capsys.readouterr().out
        assert rc == EXIT_OK

        expected = _display.render_combined_view(
            paths, event_lines=12, log_lines=30,
        )
        assert out == expected

    assert "pyratatui" not in sys.modules, (
        "non-TTY ai show must not import pyratatui; "
        f"sys.modules contains: {[k for k in sys.modules if 'pyratatui' in k]}"
    )


# ── TTY default → streaming tail (no pyratatui) ────────────────────────────


def test_show_tty_default_routes_to_stream_log(capsys, monkeypatch) -> None:
    """When stdout is a TTY, ``ai show <run_id>`` defaults to streaming.

    The high-latency-SSH redesign deliberately moves scrolling from
    pyratatui's frame loop (where every ``j``/``k``/PageUp/wheel key
    is a server round-trip) to the local terminal's native
    scrollback (zero-latency, client-side). For that to actually
    matter, the streaming tail has to be the **default** path, not
    an opt-in flag — otherwise muscle memory and tutorials still
    push operators into the laggy in-process detail TUI.

    Two contracts pinned here:

    1. **Routes to ``stream_log``.** A bare ``ai show <run_id>`` in a
       TTY hands off to :func:`auto_iterator.display.stream_log`
       with the operator's ``--log-lines`` / ``--refresh`` arguments.
    2. **No pyratatui.** Since streaming is now the default, the
       cold-start cost story applies to the default path, too: an
       interactive ``ai show`` must not pull in the native binding.
    """
    _drop_tui_modules_from_sys_modules()
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

    captured: dict = {}

    def fake_stream(paths, *, log_lines, poll_seconds):
        captured["paths"] = paths
        captured["log_lines"] = log_lines
        captured["poll_seconds"] = poll_seconds
        return EXIT_OK

    from auto_iterator import display as _display

    monkeypatch.setattr(_display, "stream_log", fake_stream, raising=True)

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        rc = main([
            "--runs-dir", tmp, "show", paths.run_id,
            "--refresh", "0.25", "--log-lines", "42",
        ])
        assert rc == EXIT_OK

    assert captured.get("paths") is not None, (
        "TTY default must hand off to display.stream_log; the "
        "in-process pyratatui detail screen is the laggy path the "
        "redesign exists to avoid as the default."
    )
    assert abs(captured["poll_seconds"] - 0.25) < 1e-6
    assert captured["log_lines"] == 42

    assert "pyratatui" not in sys.modules, (
        "TTY-default ai show must not import pyratatui; the cold-"
        "start cost contract that --json / --once / --stream all "
        "honour now applies to the default path too. "
        f"sys.modules contains: {[k for k in sys.modules if 'pyratatui' in k]}"
    )


def test_show_tui_flag_opts_into_pyratatui_detail_screen(
    capsys, monkeypatch,
) -> None:
    """``ai show <run_id> --tui`` is the explicit escape hatch back into
    the pyratatui detail screen.

    Streaming is the default because most operators run auto-iterator
    over network links where the in-process TUI's per-keystroke
    round-trip is painful. But on a *local* terminal the in-process
    TUI is genuinely nice, so we keep it as an opt-in. This test
    pins that the flag exists and does in fact route to
    :func:`auto_iterator.tui.run_detail_app`.
    """
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

    sentinel_called: dict = {}

    def fake_run(paths, *, refresh_seconds, initial_log_lines):
        sentinel_called["paths"] = paths
        sentinel_called["refresh"] = refresh_seconds
        sentinel_called["log_lines"] = initial_log_lines
        return EXIT_OK

    import auto_iterator.tui as _tui

    monkeypatch.setattr(_tui, "run_detail_app", fake_run, raising=True)

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        rc = main([
            "--runs-dir", tmp, "show", paths.run_id, "--tui",
            "--refresh", "0.25", "--log-lines", "42",
        ])
        assert rc == EXIT_OK

    assert sentinel_called.get("paths") is not None, (
        "ai show --tui must route through tui.run_detail_app"
    )
    assert abs(sentinel_called["refresh"] - 0.25) < 1e-6
    assert sentinel_called["log_lines"] == 42


def test_show_tui_flag_errors_when_stdout_not_tty(
    capsys, monkeypatch,
) -> None:
    """``--tui`` requires an interactive TTY.

    Without one we'd open the alt-screen against a pipe and produce
    garbage; surface a friendly error instead. Mirrors how
    ``cmd_tui`` (bare ``ai``) errors out on non-TTY stdout."""
    # capsys gives a non-TTY stdout already; no monkeypatch needed.
    assert not sys.stdout.isatty()

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        rc = main([
            "--runs-dir", tmp, "show", paths.run_id, "--tui",
        ])
        err = capsys.readouterr().err
        assert rc != EXIT_OK
        assert "TTY" in err or "tty" in err


# ── bare ``ai`` (run-list TUI) → Enter routes to stream_log ─────────────────


def test_bare_ai_enter_on_run_routes_to_stream_log(
    capsys, monkeypatch,
) -> None:
    """Bare ``ai`` → Enter on a row drops into the streaming tail.

    The lag story this whole change addresses is that pushing an
    in-process pyratatui detail screen from the run-list re-routes
    every scroll keystroke through a network round-trip. The fix is
    a two-step handoff:

      1. ``RunListApp`` exits cleanly on Enter, surfacing the chosen
         run via ``app.streamed_run`` (tested in
         ``tests/test_tui_smoke.py``).
      2. ``cmd_tui`` reads ``streamed_run`` and dispatches into
         :func:`auto_iterator.display.stream_log`, which writes plain
         bytes to the regular screen buffer.

    This test pins step 2: monkey-patch ``run_list_app_with_selection``
    to short-circuit the live TUI loop and return a selection
    sentinel, then assert the CLI follows up with a ``stream_log``
    call against the same paths."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

    captured: dict = {}

    def fake_stream(paths, *, log_lines, poll_seconds):
        captured["paths"] = paths
        captured["log_lines"] = log_lines
        captured["poll_seconds"] = poll_seconds
        return EXIT_OK

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))

        def fake_list_app(_runs_dir):
            return EXIT_OK, paths

        import auto_iterator.tui as _tui
        monkeypatch.setattr(
            _tui, "run_list_app_with_selection", fake_list_app,
            raising=True,
        )
        from auto_iterator import display as _display
        monkeypatch.setattr(
            _display, "stream_log", fake_stream, raising=True,
        )

        rc = main(["--runs-dir", tmp])
        assert rc == EXIT_OK

    assert captured.get("paths") is not None, (
        "Enter on a run from the bare-ai run-list must dispatch "
        "into display.stream_log; otherwise scroll keystrokes go "
        "through pyratatui's frame loop and we re-introduce the "
        "high-latency-SSH lag the redesign exists to fix."
    )
    assert captured["paths"].run_id == paths.run_id
    # The handoff must request the *full* transcript (log_lines=None)
    # rather than a bounded tail. A bounded seed silently truncates
    # anything older than the cap, leaving the operator unable to
    # scroll back to the rest of the log — the local terminal's
    # scrollback can only show what was actually written to it.
    assert captured["log_lines"] is None, (
        "Enter on a run must seed the full transcript, not a tail; "
        "otherwise the operator cannot see the full log via native "
        "scrollback once the alt-screen TUI tears down."
    )


def test_bare_ai_quit_without_selection_does_not_call_stream_log(
    capsys, monkeypatch,
) -> None:
    """Quitting the run-list TUI without picking a run must NOT then
    drop into ``stream_log``.

    The handoff trigger is an explicit selection (``streamed_run !=
    None``); a plain ``q`` / Ctrl-C from the run list returns the
    operator to the shell. Otherwise a "just exploring runs" exit
    would silently start tailing whatever happened to be highlighted
    last."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

    routed = {"stream": False}

    def fake_stream(paths, *, log_lines, poll_seconds):
        routed["stream"] = True
        return EXIT_OK

    with tempfile.TemporaryDirectory() as tmp:
        _seed_run(Path(tmp))

        def fake_list_app(_runs_dir):
            return EXIT_OK, None  # operator quit without selecting

        import auto_iterator.tui as _tui
        monkeypatch.setattr(
            _tui, "run_list_app_with_selection", fake_list_app,
            raising=True,
        )
        from auto_iterator import display as _display
        monkeypatch.setattr(
            _display, "stream_log", fake_stream, raising=True,
        )

        rc = main(["--runs-dir", tmp])
        assert rc == EXIT_OK

    assert routed["stream"] is False, (
        "no selection ⇒ no follow-up stream_log call"
    )
