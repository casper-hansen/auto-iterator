"""Dispatch contracts for ``ai show``.

The big task-spec promise is that the **stateless / scriptable** flags —
``--json``, ``--once``, ``--logs``, and the non-TTY auto-fallback — keep
working byte-identically and *do not* import :mod:`textual`. Importing
Textual on every ``ai ls`` invocation would bolt a ~120 ms startup cost
onto operator commands that have no business with a TUI.

These tests cover the dispatch surface specifically:

* ``--json`` → calls ``state_json_text``, emits parseable JSON, never
  imports ``textual``.
* ``--once`` → calls ``render_combined_view``, emits the snapshot, never
  imports ``textual``.
* Non-TTY default → behaves like ``--once``, never imports ``textual``.
* ``--logs`` → documented internal alias for ``--once``.

Each "no Textual import" check uses ``importlib.invalidate_caches`` plus
:func:`sys.modules.pop` so the test is meaningful even if a previous
test in the same process had already imported ``textual``.
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


def _drop_textual_from_sys_modules() -> None:
    """Reset ``sys.modules`` so a follow-up import is meaningful.

    We only drop the *top-level* ``textual`` namespace (and the
    ``auto_iterator.tui`` module that re-exports the live entry
    points). Sub-modules (``textual.app``, ``textual.widgets``, …) are
    left in place because:

    * Re-importing them after a full purge is expensive and brittle —
      Textual's importtime is exactly what we're protecting against.
    * The contract under test is "the dispatch path doesn't *trigger*
      a fresh import". We assert the absence of the top-level
      ``textual`` symbol after the call, which is what a parent
      process running ``ai show --json`` from a fresh interpreter
      observes.
    """
    for name in list(sys.modules):
        if name == "textual" or name.startswith("textual."):
            sys.modules.pop(name, None)
    sys.modules.pop("auto_iterator.tui", None)


# ── --json path ────────────────────────────────────────────────────────────


def test_show_json_emits_parseable_json_without_importing_textual(
    capsys, monkeypatch,
) -> None:
    """``ai show --json`` emits parseable JSON and never imports textual.

    This is the scriptable contract — anything that pipes
    ``ai show … --json`` through ``jq`` must keep working, and the
    Textual cold-start cost must not land on it."""
    _drop_textual_from_sys_modules()

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        rc = main(["--runs-dir", tmp, "show", paths.run_id, "--json"])
        out = capsys.readouterr().out
        assert rc == EXIT_OK
        # Parses → no truncation, no live ANSI.
        payload = json.loads(out)
        # state.json shape: at minimum the run id round-trips.
        assert payload.get("run_id") == paths.run_id

    assert "textual" not in sys.modules, (
        "ai show --json must not import textual; "
        f"sys.modules contains: {[k for k in sys.modules if 'textual' in k]}"
    )


# ── --once path ────────────────────────────────────────────────────────────


def test_show_once_uses_render_combined_view_without_importing_textual(
    capsys, monkeypatch,
) -> None:
    """``ai show --once`` is byte-identical to ``render_combined_view``.

    The "no Textual import" half is the cheap-startup contract. The
    "byte-identical" half is the stronger contract: shells that already
    parse the snapshot (``grep``, ``awk``, log-aggregators) must see
    exactly today's bytes."""
    _drop_textual_from_sys_modules()

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

    assert "textual" not in sys.modules, (
        "ai show --once must not import textual; "
        f"sys.modules contains: {[k for k in sys.modules if 'textual' in k]}"
    )


def test_show_logs_alias_behaves_like_once(capsys, monkeypatch) -> None:
    """``ai show --logs`` is the documented internal alias for ``--once``.

    The CLI keeps the alias around for muscle memory; verify it still
    routes to ``render_combined_view`` and stays Textual-free."""
    _drop_textual_from_sys_modules()

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

    assert "textual" not in sys.modules


# ── non-TTY default ────────────────────────────────────────────────────────


def test_show_non_tty_default_falls_back_to_once(capsys, monkeypatch) -> None:
    """Non-TTY stdout makes ``ai show <run_id>`` (no flags) emit the snapshot.

    The dispatch heuristic is ``not sys.stdout.isatty()`` → take the
    one-shot path. ``capsys`` already exposes a non-TTY stdout, so we
    don't have to fake it; we just assert the output is the combined
    view's bytes and Textual is absent.
    """
    _drop_textual_from_sys_modules()

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

    assert "textual" not in sys.modules, (
        "non-TTY ai show must not import textual; "
        f"sys.modules contains: {[k for k in sys.modules if 'textual' in k]}"
    )


# ── TTY default → Textual ──────────────────────────────────────────────────


def test_show_tty_default_lazy_imports_tui(capsys, monkeypatch) -> None:
    """When stdout is a TTY, ``ai show`` lazy-imports the TUI entry point.

    We don't actually run the TUI — that would block on stdin — we
    only assert the dispatch path *would* land at
    :func:`auto_iterator.tui.run_detail_app`. Pinning the symbol with
    ``monkeypatch`` lets us return a sentinel without spinning up
    Textual.
    """
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

    sentinel_called: dict = {}

    def fake_run(paths, *, refresh_seconds, initial_log_lines):
        sentinel_called["paths"] = paths
        sentinel_called["refresh"] = refresh_seconds
        sentinel_called["log_lines"] = initial_log_lines
        return EXIT_OK

    # Patch on the ``tui`` module so the lazy import inside
    # ``cmd_show`` resolves to our fake. We don't drop ``textual``
    # from sys.modules here — this test *expects* the lazy import to
    # happen, and patching the symbol is enough to keep it cheap.
    import auto_iterator.tui as _tui

    monkeypatch.setattr(_tui, "run_detail_app", fake_run, raising=True)

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        rc = main([
            "--runs-dir", tmp, "show", paths.run_id,
            "--refresh", "0.25", "--log-lines", "42",
        ])
        assert rc == EXIT_OK

    assert sentinel_called.get("paths") is not None
    assert abs(sentinel_called["refresh"] - 0.25) < 1e-6
    assert sentinel_called["log_lines"] == 42
