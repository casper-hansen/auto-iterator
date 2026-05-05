"""Tests for the interactive run-selector wired into the ``ai`` CLI.

Covers the spec acceptance criteria without requiring a real terminal:

* Parser accepts omitted ``run_id`` for selector-enabled subcommands.
* In non-TTY mode, an omitted ``run_id`` exits with ``EXIT_USER_ERROR``.
* Explicit ``run_id`` still bypasses selector resolution.
* The selector resolution path picks a ``RunRow`` and sets
  ``args.run_id`` before the existing handler runs.
* ``ai show`` default (non-TTY) output is the combined status + events
  + agent-output view; it does *not* dump raw JSON.
* ``ai show --json`` remains parseable JSON.
* The no-runs case exits cleanly with an error code.
* The live renderer can be exercised with a fake stop-condition + sleep.

We avoid spawning subprocesses where possible — the selector code path
is patched out so the dispatcher logic can be exercised in-process.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_iterator import selector  # noqa: E402
from auto_iterator.cli import (  # noqa: E402
    EXIT_OK,
    EXIT_USER_ERROR,
    SELECTOR_COMMANDS,
    _build_parser,
    _normalize_send_args,
    _resolve_selector_run_id,
    main,
)
from auto_iterator.events import EventLog, RunState  # noqa: E402
from auto_iterator.ls import RunRow  # noqa: E402
from auto_iterator.meta import write_meta  # noqa: E402
from auto_iterator.run_dir import (  # noqa: E402
    create_run_dir,
    new_run_id,
    now_iso,
)


# ── Parser shape ──────────────────────────────────────────────────────────────


def test_parser_accepts_omitted_run_id_for_selector_commands() -> None:
    """Every selector-enabled subcommand must parse without ``run_id``.

    The selector lives in the dispatcher, not the parser; argparse
    must therefore accept ``ai show`` with no positional and we resolve
    the id later."""
    p = _build_parser()
    # Subcommands that need extra required flags / args are tested with
    # those filled in; the assertion is only about the missing run id.
    cases = {
        "restart": ["restart"],
        "kill": ["kill"],
        "show": ["show"],
        "send": ["send", "guidance text"],
        "rewind": ["rewind", "--to", "outer=1,inner=1"],
        "set-prompt": ["set-prompt", "--text", "new"],
        "pause": ["pause"],
        "resume": ["resume"],
        "worktree": ["worktree"],
        "diff": ["diff"],
        "apply": ["apply"],
        "revert": ["revert"],
        "worktree-remove": ["worktree-remove"],
    }
    assert set(cases) == set(SELECTOR_COMMANDS)
    for cmd, argv in cases.items():
        ns = p.parse_args(argv)
        # ``send`` carries a second positional (``text``) so its raw
        # parse stores the lone positional as ``run_id``; the dispatcher
        # normalizes it before selector resolution. Apply the same
        # normalization here so the assertion mirrors what handlers see.
        if cmd == "send":
            _normalize_send_args(ns)
        assert ns.run_id is None, f"{cmd}: expected run_id=None, got {ns.run_id!r}"
    print("  test_parser_accepts_omitted_run_id_for_selector_commands PASS")


def test_parser_explicit_run_id_still_works() -> None:
    p = _build_parser()
    ns = p.parse_args(["show", "abc123"])
    assert ns.run_id == "abc123"
    ns = p.parse_args(["kill", "abc123", "--force"])
    assert ns.run_id == "abc123"
    assert ns.force is True
    print("  test_parser_explicit_run_id_still_works PASS")


def test_send_parses_all_positional_orderings() -> None:
    """Reviewer regression: ``ai send RUN_ID --wait TEXT`` must parse.

    The four forms below all need to round-trip to the same
    ``(run_id, text, wait)`` tuple after :func:`_normalize_send_args`
    runs. The single-positional cases must leave ``run_id=None`` so the
    selector path takes over."""
    p = _build_parser()

    def parsed(argv):
        ns = p.parse_args(argv)
        rc = _normalize_send_args(ns)
        return rc, ns

    # 1. Both positionals, no flag.
    rc, ns = parsed(["send", "RID", "Focus on X"])
    assert rc is None
    assert (ns.run_id, ns.text, ns.wait) == ("RID", "Focus on X", False)

    # 2. Both positionals with --wait between them — the case the
    #    reviewer flagged as broken under the previous parser shape.
    rc, ns = parsed(["send", "RID", "--wait", "Focus on X"])
    assert rc is None
    assert (ns.run_id, ns.text, ns.wait) == ("RID", "Focus on X", True)

    # 3. --wait before any positional.
    rc, ns = parsed(["send", "--wait", "RID", "Focus on X"])
    assert rc is None
    assert (ns.run_id, ns.text, ns.wait) == ("RID", "Focus on X", True)

    # 4. Single positional → treat as text, leave run_id for selector.
    rc, ns = parsed(["send", "Focus on X"])
    assert rc is None
    assert (ns.run_id, ns.text, ns.wait) == (None, "Focus on X", False)

    # 5. Single positional + --wait, same disambiguation.
    rc, ns = parsed(["send", "--wait", "Focus on X"])
    assert rc is None
    assert (ns.run_id, ns.text, ns.wait) == (None, "Focus on X", True)

    print("  test_send_parses_all_positional_orderings PASS")


def test_send_missing_text_is_user_error(capsys) -> None:
    """No positionals at all → clean user error, no selector hang."""
    p = _build_parser()
    ns = p.parse_args(["send"])
    rc = _normalize_send_args(ns)
    assert rc == EXIT_USER_ERROR
    err = capsys.readouterr().err
    assert "requires guidance text" in err
    print("  test_send_missing_text_is_user_error PASS")


def test_show_combined_view_flags() -> None:
    """``ai show`` exposes the combined-view knobs and a one-shot escape."""
    p = _build_parser()
    ns = p.parse_args(["show", "rid"])
    # New live-view defaults: small recent-events window + a manageable
    # agent-output tail. Refresh interval is on the responsive end of
    # the spec'd 250–500ms range.
    assert ns.once is False
    assert ns.event_lines == 12
    assert ns.log_lines == 30
    assert ns.lines is None
    assert ns.refresh == 0.4
    # ``--once`` forces the scriptable one-shot path.
    ns_once = p.parse_args(["show", "rid", "--once"])
    assert ns_once.once is True
    # Section caps are independently tunable.
    ns_tuned = p.parse_args([
        "show", "rid",
        "--event-lines", "5", "--log-lines", "60", "--refresh", "1.0",
    ])
    assert ns_tuned.event_lines == 5
    assert ns_tuned.log_lines == 60
    assert ns_tuned.refresh == 1.0
    # ``--lines`` survives as a backwards-compat alias for ``--log-lines``
    # (and continues to be accepted alongside the deprecated ``--logs``).
    ns_legacy = p.parse_args(["show", "rid", "--logs", "--lines", "10"])
    assert ns_legacy.logs is True
    assert ns_legacy.lines == 10
    print("  test_show_combined_view_flags PASS")


# ── Selector resolution ───────────────────────────────────────────────────────


def _make_args(**fields):
    """Build a minimal argparse-like Namespace for dispatcher tests."""
    import argparse
    return argparse.Namespace(**fields)


def test_resolve_selector_run_id_explicit_bypasses_selector() -> None:
    """If the user passed run_id, the selector must not be invoked."""
    args = _make_args(cmd="show", run_id="explicit-id")
    with mock.patch.object(selector, "select_run") as sel_mock, \
            mock.patch.object(selector, "is_interactive", return_value=True):
        rc = _resolve_selector_run_id(args, Path("/tmp"))
    assert rc == EXIT_OK
    assert args.run_id == "explicit-id"
    assert args._selected is False
    sel_mock.assert_not_called()
    print("  test_resolve_selector_run_id_explicit_bypasses_selector PASS")


def test_resolve_selector_run_id_non_tty_exits_user_error(capsys) -> None:
    """Non-interactive callers must get a clean error, not a hung select."""
    args = _make_args(cmd="show", run_id=None)
    with mock.patch.object(selector, "is_interactive", return_value=False):
        rc = _resolve_selector_run_id(args, Path("/tmp"))
    assert rc == EXIT_USER_ERROR
    assert args.run_id is None
    err = capsys.readouterr().err
    assert "requires a run_id" in err
    print("  test_resolve_selector_run_id_non_tty_exits_user_error PASS")


def test_resolve_selector_run_id_no_runs(capsys) -> None:
    """A runs-dir with zero runs surfaces ``(no runs)`` and exits cleanly."""
    with tempfile.TemporaryDirectory() as tmp:
        args = _make_args(cmd="show", run_id=None)
        with mock.patch.object(selector, "is_interactive", return_value=True):
            rc = _resolve_selector_run_id(args, Path(tmp))
    assert rc == EXIT_USER_ERROR
    err = capsys.readouterr().err
    assert "(no runs)" in err
    print("  test_resolve_selector_run_id_no_runs PASS")


def test_resolve_selector_run_id_picks_runrow(capsys) -> None:
    """Interactive path: ``select_run`` returns a row → args.run_id is set."""
    with tempfile.TemporaryDirectory() as tmp:
        # Seed one real run dir so list_runs has something to return.
        paths = create_run_dir(Path(tmp), new_run_id())
        write_meta(paths, {
            "run_id": paths.run_id,
            "pid": 999_999_999,
            "status": "running",
            "workspace": "/tmp/ws",
            "started_at": now_iso(),
        })
        args = _make_args(cmd="show", run_id=None)

        # Simulate the user pressing Enter on the first row.
        chosen = RunRow(
            run_id=paths.run_id, status="crashed", phase="init",
            outer=0, inner=0, last_verdict="", exit_code=None,
            approved=False, started_at=now_iso(), updated_at=now_iso(),
            workspace="/tmp/ws", prompt_preview="", pid=None,
        )
        with mock.patch.object(selector, "is_interactive", return_value=True), \
                mock.patch(
                    "auto_iterator.selector.select_run", return_value=chosen,
                ):
            rc = _resolve_selector_run_id(args, Path(tmp))
    assert rc == EXIT_OK
    assert args.run_id == paths.run_id
    assert args._selected is True
    print("  test_resolve_selector_run_id_picks_runrow PASS")


def test_resolve_selector_run_id_cancel(capsys) -> None:
    """User cancel (Esc / Ctrl-C) returns ``EXIT_USER_ERROR`` cleanly."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = create_run_dir(Path(tmp), new_run_id())
        write_meta(paths, {
            "run_id": paths.run_id, "pid": 999_999_999,
            "status": "running", "workspace": "/tmp/ws",
            "started_at": now_iso(),
        })
        args = _make_args(cmd="show", run_id=None)
        with mock.patch.object(selector, "is_interactive", return_value=True), \
                mock.patch(
                    "auto_iterator.selector.select_run", return_value=None,
                ):
            rc = _resolve_selector_run_id(args, Path(tmp))
    assert rc == EXIT_USER_ERROR
    err = capsys.readouterr().err
    assert "cancelled" in err
    print("  test_resolve_selector_run_id_cancel PASS")


# ── End-to-end dispatch via main() ────────────────────────────────────────────


def _seed_run(runs_dir: Path) -> Path:
    """Create a run-dir with state.json + meta.json + a couple of events."""
    paths = create_run_dir(runs_dir, new_run_id())
    write_meta(paths, {
        "run_id": paths.run_id,
        "pid": 999_999_999,  # not alive → reconciled status will be "crashed"
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
    state.total_reviews = 2
    log = EventLog(paths, state)
    log.emit("run_started", workspace="/tmp/ws")
    log.emit("inner_started", outer=1, inner=1)
    log.emit("review_finished", outer=1, inner=1, verdict="needs_fixes")
    return paths


def test_show_default_is_combined_view(capsys) -> None:
    """``ai show RUN_ID`` (non-TTY) prints status + events + agent output.

    The default observation experience is now the combined view; it
    includes the labelled status block, a Recent events section, and
    an Agent output section. No top-level JSON, no live ANSI escapes
    (capsys's stdout is not a TTY)."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        paths.agent_log.write_text(
            "agent line one\nagent line two\n", encoding="utf-8",
        )
        rc = main(["--runs-dir", tmp, "show", paths.run_id])
    out = capsys.readouterr().out
    assert rc == EXIT_OK
    # Section anchors must all be present so operators have a single
    # screen to look at.
    assert out.startswith("Run ") or "Run " in out.splitlines()[0]
    assert "status" in out and "phase" in out and "outer/inner" in out
    assert "workspace" in out
    assert "Recent events" in out
    assert "Agent output" in out
    # Useful event payloads still surface through the combined view.
    assert "run_started" in out
    assert "review_finished" in out
    # Agent transcript tail is included verbatim.
    assert "agent line two" in out
    # No top-level JSON object.
    stripped = out.lstrip()
    assert not stripped.startswith("{"), "show default must not be raw JSON"
    print("  test_show_default_is_combined_view PASS")


def test_show_json_remains_valid_json(capsys) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        rc = main(["--runs-dir", tmp, "show", paths.run_id, "--json"])
    out = capsys.readouterr().out
    assert rc == EXIT_OK
    obj = json.loads(out)  # must parse; raises otherwise
    assert isinstance(obj, dict)
    assert obj.get("run_id") == paths.run_id
    print("  test_show_json_remains_valid_json PASS")


def test_show_once_includes_log_tail(capsys) -> None:
    """``ai show RUN_ID --once --log-lines N`` truncates the agent tail."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        paths.agent_log.write_text(
            "\n".join(f"line {i}" for i in range(1, 11)) + "\n",
            encoding="utf-8",
        )
        rc = main([
            "--runs-dir", tmp, "show", paths.run_id, "--once",
            "--log-lines", "5",
        ])
    out = capsys.readouterr().out
    assert rc == EXIT_OK
    # Last 5 lines of 1..10 land in the Agent output section.
    assert "line 10" in out
    assert "line 6" in out
    assert "line 5" not in out  # 1..5 truncated
    # Section header must be present so operators don't confuse log
    # output with event lines.
    assert "Agent output" in out
    print("  test_show_once_includes_log_tail PASS")


def test_show_legacy_logs_alias_renders_combined(capsys) -> None:
    """The hidden ``--logs`` flag still produces a useful one-shot view.

    Older scripts that invoke ``ai show RUN_ID --logs --lines N`` keep
    working — ``--logs`` now collapses to ``--once`` and ``--lines`` is
    honoured as the agent-output tail size. The output is the combined
    view, not a raw log dump, but the same agent transcript content
    still appears."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        paths.agent_log.write_text(
            "\n".join(f"line {i}" for i in range(1, 11)) + "\n",
            encoding="utf-8",
        )
        rc = main([
            "--runs-dir", tmp, "show", paths.run_id, "--logs",
            "--lines", "5",
        ])
    out = capsys.readouterr().out
    assert rc == EXIT_OK
    assert "line 10" in out and "line 6" in out
    assert "line 5" not in out
    assert "Agent output" in out
    print("  test_show_legacy_logs_alias_renders_combined PASS")


def test_show_handles_missing_agent_log(capsys) -> None:
    """Combined view survives an empty/missing logs/agent.log gracefully."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        # Don't create paths.agent_log — simulate a fresh run before
        # the runner has emitted anything.
        if paths.agent_log.exists():
            paths.agent_log.unlink()
        rc = main(["--runs-dir", tmp, "show", paths.run_id, "--once"])
    out = capsys.readouterr().out
    assert rc == EXIT_OK
    assert "Agent output" in out
    assert "agent has not produced output yet" in out
    print("  test_show_handles_missing_agent_log PASS")


def test_show_handles_empty_agent_log(capsys) -> None:
    """An empty agent.log surfaces as a friendly placeholder, not blank."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        paths.agent_log.write_text("", encoding="utf-8")
        rc = main(["--runs-dir", tmp, "show", paths.run_id, "--once"])
    out = capsys.readouterr().out
    assert rc == EXIT_OK
    assert "Agent output" in out
    assert "agent output is empty" in out
    print("  test_show_handles_empty_agent_log PASS")


def test_main_show_no_runs_exits_user_error(capsys) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(selector, "is_interactive", return_value=True):
            rc = main(["--runs-dir", tmp, "show"])
    err = capsys.readouterr().err
    assert rc == EXIT_USER_ERROR
    assert "(no runs)" in err
    print("  test_main_show_no_runs_exits_user_error PASS")


def test_main_show_non_tty_without_run_id(capsys) -> None:
    """Non-interactive ``ai show`` (no run_id) must error out, not hang."""
    with tempfile.TemporaryDirectory() as tmp:
        # Seed a run so the failure is *only* the TTY check, not "no runs".
        _seed_run(Path(tmp))
        with mock.patch.object(selector, "is_interactive", return_value=False):
            rc = main(["--runs-dir", tmp, "show"])
    err = capsys.readouterr().err
    assert rc == EXIT_USER_ERROR
    assert "requires a run_id" in err
    print("  test_main_show_non_tty_without_run_id PASS")


def test_main_selector_picks_run_and_runs_handler(capsys, monkeypatch) -> None:
    """Interactive show with no run_id: selector picks → handler runs.

    Pin stdout to non-TTY so ``cmd_show`` takes the deterministic
    one-shot path and we get the rendered combined view in stdout
    rather than entering the live renderer."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        chosen = RunRow(
            run_id=paths.run_id, status="crashed", phase="review",
            outer=1, inner=1, last_verdict="needs_fixes",
            exit_code=None, approved=False, started_at=now_iso(),
            updated_at=now_iso(), workspace="/tmp/ws",
            prompt_preview="Implement feature X carefully.", pid=None,
        )
        with mock.patch.object(selector, "is_interactive", return_value=True), \
                mock.patch(
                    "auto_iterator.selector.select_run", return_value=chosen,
                ):
            rc = main(["--runs-dir", tmp, "show"])
    out = capsys.readouterr().out
    assert rc == EXIT_OK
    assert paths.run_id in out
    assert "status" in out  # combined view, not raw JSON
    assert "Recent events" in out
    print("  test_main_selector_picks_run_and_runs_handler PASS")


def test_main_destructive_confirmation_skipped_for_explicit_id(
    capsys, monkeypatch,
) -> None:
    """Explicit run_id must never trigger the confirmation prompt.

    We patch ``input`` to fail loudly: if the dispatcher prompts
    despite the explicit id, the test fails."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        # Make pid_alive return False so kill exits as RUN_GONE rather
        # than actually trying to signal anything.
        monkeypatch.setattr(
            "auto_iterator.cli.pid_alive", lambda pid: False,
        )
        monkeypatch.setattr(
            "builtins.input",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("must not prompt for explicit id"),
            ),
        )
        rc = main(["--runs-dir", tmp, "kill", paths.run_id])
    # kill on a dead pid returns RUN_GONE, which is "we got past the
    # confirmation" — that's all this test cares about.
    assert rc in (EXIT_OK, 3), rc
    print("  test_main_destructive_confirmation_skipped_for_explicit_id PASS")


def test_main_destructive_confirmation_prompts_after_selector(
    capsys, monkeypatch,
) -> None:
    """Selector-chosen destructive run: confirm → ``n`` cancels cleanly."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        chosen = RunRow(
            run_id=paths.run_id, status="crashed", phase="review",
            outer=0, inner=0, last_verdict="", exit_code=None,
            approved=False, started_at=now_iso(), updated_at=now_iso(),
            workspace="/tmp/ws", prompt_preview="x", pid=None,
        )
        prompts = []

        def fake_input(prompt=""):
            prompts.append(prompt)
            return "n"

        monkeypatch.setattr("builtins.input", fake_input)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
        with mock.patch.object(selector, "is_interactive", return_value=True), \
                mock.patch(
                    "auto_iterator.selector.select_run", return_value=chosen,
                ):
            rc = main(["--runs-dir", tmp, "kill"])
    err = capsys.readouterr().err
    assert rc == EXIT_USER_ERROR
    assert prompts and "Confirm" in prompts[0]
    assert "cancelled" in err
    print("  test_main_destructive_confirmation_prompts_after_selector PASS")


def test_main_destructive_yes_skips_confirmation(capsys, monkeypatch) -> None:
    """``--yes`` after a selector pick suppresses the confirmation."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        chosen = RunRow(
            run_id=paths.run_id, status="crashed", phase="review",
            outer=0, inner=0, last_verdict="", exit_code=None,
            approved=False, started_at=now_iso(), updated_at=now_iso(),
            workspace="/tmp/ws", prompt_preview="x", pid=None,
        )
        monkeypatch.setattr(
            "builtins.input",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("must not prompt with --yes"),
            ),
        )
        monkeypatch.setattr(
            "auto_iterator.cli.pid_alive", lambda pid: False,
        )
        with mock.patch.object(selector, "is_interactive", return_value=True), \
                mock.patch(
                    "auto_iterator.selector.select_run", return_value=chosen,
                ):
            rc = main(["--runs-dir", tmp, "kill", "--yes"])
    # Pid is dead → kill returns EXIT_RUN_GONE (3). We just want past the
    # confirmation gate.
    assert rc in (EXIT_OK, 3), rc
    print("  test_main_destructive_yes_skips_confirmation PASS")


# ── Display-layer unit checks ────────────────────────────────────────────────


def test_display_render_status_view_shape() -> None:
    from auto_iterator.display import render_status_view

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        out = render_status_view(paths)
    # Each labelled field appears.
    for label in (
        "status", "phase", "outer/inner", "paused", "approved",
        "last verdict", "total reviews", "exit code", "pid",
        "workspace", "started", "updated",
    ):
        assert label in out, f"missing label '{label}' in status view"
    assert "Run " in out
    print("  test_display_render_status_view_shape PASS")


def test_display_render_event_compact() -> None:
    from auto_iterator.display import render_event

    line = render_event({
        "seq": 7, "type": "review_finished",
        "timestamp": "2026-04-29T10:11:12.345+00:00",
        "outer": 1, "inner": 2, "verdict": "needs_fixes",
    }, color=False)
    assert "10:11:12" in line
    assert "#7" in line
    assert "review_finished" in line
    assert "o/i=1/2" in line
    assert "verdict=needs_fixes" in line
    print("  test_display_render_event_compact PASS")


def test_display_render_combined_view_sections() -> None:
    """The single-string combined view stitches all three sections."""
    from auto_iterator.display import render_combined_view

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        paths.agent_log.write_text("first\nsecond\nthird\n", encoding="utf-8")
        out = render_combined_view(paths, event_lines=10, log_lines=10)
    # All three section headers must show up so the operator knows
    # what they're looking at.
    assert "Run " in out
    assert "Recent events" in out
    assert "Agent output" in out
    # Status fields, event lines and log lines all coexist.
    assert "outer/inner" in out
    assert "review_finished" in out
    assert "third" in out
    # Output ends with a newline so callers can ``sys.stdout.write`` it.
    assert out.endswith("\n")
    print("  test_display_render_combined_view_sections PASS")


def test_display_render_combined_view_missing_files(capsys) -> None:
    """The combined view degrades cleanly when state/events/logs are gone."""
    from auto_iterator.display import render_combined_view

    with tempfile.TemporaryDirectory() as tmp:
        # No write_meta / no events / no agent_log — only an empty run dir.
        paths = create_run_dir(Path(tmp), new_run_id())
        out = render_combined_view(paths, event_lines=5, log_lines=5)
    # We still produce a recognisable structure rather than crashing
    # or returning empty output.
    assert "Run " in out
    assert "Recent events" in out
    assert "no events yet" in out
    assert "Agent output" in out
    assert "agent has not produced output" in out
    print("  test_display_render_combined_view_missing_files PASS")


def test_display_tail_text_file_returns_last_lines(tmp_path) -> None:
    """``tail_text_file`` must read only the trailing N lines."""
    from auto_iterator.display import tail_text_file

    log = tmp_path / "agent.log"
    log.write_text(
        "\n".join(f"row {i}" for i in range(1, 101)) + "\n",
        encoding="utf-8",
    )
    last5 = tail_text_file(log, lines=5)
    assert last5 == [f"row {i}" for i in range(96, 101)]
    # Missing file → empty list, not exception.
    assert tail_text_file(tmp_path / "missing.log", lines=5) == []
    # Empty file → empty list.
    empty = tmp_path / "empty.log"
    empty.write_text("", encoding="utf-8")
    assert tail_text_file(empty, lines=5) == []
    print("  test_display_tail_text_file_returns_last_lines PASS")


def test_display_tail_text_file_long_single_line(tmp_path) -> None:
    """Regression: a long single line longer than the read window must
    still produce visible (truncated) output, not an empty list.

    The previous implementation called ``f.readline()`` after seeking
    near EOF to drop a partial leading line, but when the entire window
    was one unbroken line that consumed the whole window and left the
    caller with nothing — hiding the latest agent output. We now keep
    the bytes when the window contains no newline at all."""
    from auto_iterator.display import tail_text_file

    log = tmp_path / "agent.log"
    # 130 KiB single-line payload, no trailing newline. Larger than the
    # default 30 * 4096 = 120 KiB chunk window so we exercise the
    # "seeked, no newline visible" branch.
    payload = "x" * (130 * 1024)
    log.write_bytes(payload.encode("utf-8"))

    out = tail_text_file(log, lines=30)
    assert out, "long single line must not return empty output"
    # We expect exactly one line (the truncated tail of the giant line),
    # and it must end with the actual end-of-file content.
    assert len(out) == 1
    assert out[0].endswith("x" * 100)
    # The returned tail must come from the *end* of the window, so its
    # length is bounded by our chunk budget rather than the whole file.
    assert len(out[0]) <= 30 * 4096
    print("  test_display_tail_text_file_long_single_line PASS")


def test_display_tail_text_file_long_line_then_short_line(tmp_path) -> None:
    """A trailing newline-terminated short line after a long line still
    surfaces — and the partial leading line is correctly dropped."""
    from auto_iterator.display import tail_text_file

    log = tmp_path / "agent.log"
    # First a long line that the seek will cut in half, then a short
    # final line. We must see only the short final line, not part of
    # the long one.
    big = "a" * (200 * 1024)
    log.write_text(big + "\n" + "tail line\n", encoding="utf-8")

    out = tail_text_file(log, lines=5)
    assert out == ["tail line"]
    print("  test_display_tail_text_file_long_line_then_short_line PASS")


def test_display_run_live_show_iterates_with_stop_hook() -> None:
    """The live runner is exercisable with a fake stop + sleep.

    We never touch a real terminal: a custom ``out`` collects writes
    and ``sleep`` is a counter. ``should_continue`` returns False
    after a few iterations so the loop terminates deterministically.
    A fake ``get_size`` keeps the fit logic deterministic across
    different test environments.

    Asserts:
      * loop ran the expected number of times,
      * each tick wrote a fresh combined view (status + sections) to
        ``out``,
      * cursor + alternate-screen state were both restored before
        return,
      * a final post-exit snapshot landed on the regular screen.
    """
    import io
    from auto_iterator.display import run_live_show

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        paths.agent_log.write_text("hello world\n", encoding="utf-8")

        out = io.StringIO()
        sleeps: list[float] = []

        def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        # Run 3 iterations then stop. Iteration counter starts at 1.
        # Generous fake terminal so the fit logic doesn't cap caps.
        rc = run_live_show(
            paths,
            event_lines=5,
            log_lines=5,
            refresh_seconds=0.25,
            out=out,
            sleep=fake_sleep,
            should_continue=lambda i: i < 3,
            get_size=lambda: (120, 60),
        )

    text = out.getvalue()
    assert rc == 0
    # We asked for 3 iterations and each renders the combined view.
    assert text.count("Recent events") >= 3
    assert text.count("Agent output") >= 3
    # Cursor + alt-screen state always restored at the end.
    assert text.endswith("(exited live view)\n") or "Run " in text
    assert "\033[?25h" in text
    assert "\033[?1049l" in text
    assert "\033[?25l" in text  # cursor was hidden during the loop
    assert "\033[?1049h" in text  # alt screen was entered
    # Sleep was called between renders but not after the last one
    # (we stop *before* the final sleep). 2 sleeps for 3 iterations.
    assert sleeps == [0.25, 0.25]
    # Combined view content showed up.
    assert "hello world" in text
    print("  test_display_run_live_show_iterates_with_stop_hook PASS")


def test_display_run_live_show_ctrl_c_exits_cleanly() -> None:
    """KeyboardInterrupt during ``sleep`` exits 0 and restores state."""
    import io
    from auto_iterator.display import run_live_show

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        out = io.StringIO()

        calls = {"n": 0}

        def boom(_seconds: float) -> None:
            calls["n"] += 1
            raise KeyboardInterrupt

        rc = run_live_show(
            paths,
            event_lines=3,
            log_lines=3,
            refresh_seconds=0.1,
            out=out,
            sleep=boom,
            get_size=lambda: (120, 60),
        )

    text = out.getvalue()
    # Clean exit code despite the simulated Ctrl-C.
    assert rc == 0
    assert calls["n"] == 1  # we slept once before the interrupt
    # Terminal state is restored: cursor shown, alt screen left.
    assert "\033[?25h" in text
    assert "\033[?1049l" in text
    # The final post-exit snapshot lands on the regular screen so the
    # user has something to look at after returning.
    assert text.rstrip().endswith("(exited live view)")
    # The combined view rendered at least once before the interrupt.
    assert "Recent events" in text
    print("  test_display_run_live_show_ctrl_c_exits_cleanly PASS")


def test_display_fit_section_caps_24_row_terminal() -> None:
    """The reviewer's blocker: 12 events + 30 logs scroll status off a 24-row
    terminal. ``fit_section_caps`` must shrink them to fit, never letting
    the events or agent-output sections drop below 1 line each."""
    from auto_iterator.display import (
        _LIVE_VIEW_OVERHEAD,
        _status_section_lines,
        fit_section_caps,
        render_combined_view,
    )

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        paths.agent_log.write_text(
            "\n".join(f"line {i}" for i in range(1, 51)) + "\n",
            encoding="utf-8",
        )

        # 24-row terminal is the canonical small case the reviewer
        # called out. The fitted caps + status height + fixed overhead
        # must be <= 24 so the live renderer never scrolls the status
        # section off screen.
        rows = 24
        events, logs = fit_section_caps(
            paths,
            rows=rows,
            requested_event_lines=12,
            requested_log_lines=30,
        )
        assert events >= 1 and logs >= 1, (
            f"both sections must be visible: events={events} logs={logs}"
        )
        status_h = len(_status_section_lines(paths))
        # Section content + status block + fixed overhead must fit.
        # ``_LIVE_VIEW_OVERHEAD`` already accounts for both section
        # headers and the footer, so we don't add headers separately.
        total = status_h + _LIVE_VIEW_OVERHEAD + events + logs
        assert total <= rows, (
            f"combined view must fit in {rows} rows, "
            f"got {total} (status={status_h}, events={events}, logs={logs})"
        )

        # Sanity-check by counting the rendered output's actual line
        # count and adding the live runner's footer (1 blank + 1 line).
        view = render_combined_view(
            paths, event_lines=events, log_lines=logs,
        )
        # Trailing newline from render_combined_view doesn't count as
        # a visible row; subtract it.
        rendered_rows = view.count("\n") - (1 if view.endswith("\n") else 0)
        rendered_rows += 2  # footer (\n + footer line)
        assert rendered_rows <= rows, (
            f"rendered view is {rendered_rows} rows for a {rows}-row "
            "terminal — status would scroll off screen"
        )

        # Caps remain bounded by the user's request (never grow).
        assert events <= 12 and logs <= 30
    print("  test_display_fit_section_caps_24_row_terminal PASS")


def test_display_fit_section_caps_large_terminal() -> None:
    """On a tall terminal the user's requested caps must be honoured."""
    from auto_iterator.display import fit_section_caps

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        events, logs = fit_section_caps(
            paths,
            rows=80,
            requested_event_lines=12,
            requested_log_lines=30,
        )
    # Plenty of room: nothing was capped below the request.
    assert events == 12
    assert logs == 30
    print("  test_display_fit_section_caps_large_terminal PASS")


def test_display_fit_section_caps_tiny_terminal() -> None:
    """When the terminal is so small the status alone overflows, fall
    back to a tiny but still-rendered layout instead of zero lines."""
    from auto_iterator.display import fit_section_caps

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        events, logs = fit_section_caps(
            paths,
            rows=10,
            requested_event_lines=12,
            requested_log_lines=30,
        )
    assert events >= 1 and logs >= 1
    print("  test_display_fit_section_caps_tiny_terminal PASS")


def test_display_run_live_show_respects_terminal_height() -> None:
    """Reviewer regression: live view fits in a 24-row terminal.

    The pre-fix loop wrote ~60+ lines per tick regardless of terminal
    height, so the alt-screen scrolled status + events off and the
    user only saw agent output and the footer. This test pins the
    fake terminal to 24 rows, runs one iteration, and asserts every
    chunk that the renderer wrote (between two screen-clears or
    between a clear and the cursor-restore epilogue) is at most 24
    rows tall."""
    import io
    from auto_iterator.display import run_live_show

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        # Plenty of agent output so the renderer would happily flood
        # the screen if it ignored the terminal height.
        paths.agent_log.write_text(
            "\n".join(f"line {i}" for i in range(1, 101)) + "\n",
            encoding="utf-8",
        )
        out = io.StringIO()
        rc = run_live_show(
            paths,
            event_lines=12,
            log_lines=30,
            refresh_seconds=0.25,
            out=out,
            sleep=lambda _s: None,
            should_continue=lambda i: i < 1,
            get_size=lambda: (80, 24),
        )

    text = out.getvalue()
    assert rc == 0
    # Status anchor present despite the 24-row budget.
    assert "Run " in text
    assert "Recent events" in text
    assert "Agent output" in text

    # Slice the live-loop redraw out of the buffer: it sits between
    # the initial alt-screen-on / clear-home and the cursor-restore
    # epilogue. That slice must be at most ``rows`` lines tall, or
    # the alt screen would scroll the status block off.
    #
    # Strict cursor-position invariant: the redraw begins with the
    # cursor at row 1 (just after ``_CLEAR_HOME``) and each newline
    # advances the cursor by one row. To prevent a scroll on a
    # ``rows``-tall terminal, the cursor must land on row ``<=
    # rows`` after the redraw — i.e. ``newline_count <= rows - 1``.
    # Anything tighter (a final trailing ``\n`` on the footer, for
    # example) would push the cursor to row ``rows + 1`` and most
    # terminals turn that into a scroll, knocking the top status
    # row off screen tick after tick. So we assert ``< 24`` here,
    # not ``<= 24``.
    clear = "\033[H\033[2J"
    show_cursor = "\033[?25h"
    assert clear in text and show_cursor in text
    redraw = text.split(clear, 1)[1].split(show_cursor, 1)[0]
    redraw_rows = redraw.count("\n")
    assert redraw_rows < 24, (
        f"live redraw advanced cursor {redraw_rows + 1} rows on a "
        f"24-row terminal — the bottom-row newline triggers scroll "
        f"and knocks the top status row off screen: {redraw!r}"
    )

    # Lines well within the requested 30-line cap must not all appear
    # — the fit shrunk the agent-output section to fit. The very last
    # log line must always be visible (operators want the latest).
    assert "line 100" in text
    # The very first log line (1) must NOT appear in the live redraw,
    # because we capped the section. (The post-exit snapshot uses the
    # full requested cap, so it can show line 1 there — we look only
    # at the redraw slice.)
    assert "line 1\n" not in redraw, (
        "fit must drop the oldest agent-output lines on a 24-row "
        "terminal, but the redraw still contains 'line 1'"
    )
    print("  test_display_run_live_show_respects_terminal_height PASS")


def test_display_truncate_visible_unit() -> None:
    """``_truncate_visible`` clamps visible width while preserving ANSI codes.

    The helper exists because terminal wrapping silently doubles a
    section's row count when a single line is wider than the terminal,
    breaking the row budget computed by :func:`fit_section_caps`.
    These assertions pin the visible-width contract:

    * Already-fitting strings come through unchanged.
    * Plain truncation reserves one column for the ``…`` marker.
    * ANSI escapes inside the kept prefix survive intact.
    * A terminating reset is emitted so a colored prefix can't bleed.
    * Empty / non-positive ``max_cols`` returns the empty string.
    """
    from auto_iterator.display import _ANSI_RE, _truncate_visible

    # Short string: untouched.
    assert _truncate_visible("hi", 80) == "hi"

    # Plain truncation: visible width must collapse to exactly max_cols
    # and the visible ending must be the ellipsis indicator. We compare
    # on the ANSI-stripped projection so the assertion holds regardless
    # of whether ``NC`` is empty (colors disabled in a non-TTY test
    # harness) or a real escape sequence.
    out = _truncate_visible("x" * 100, 10)
    visible = _ANSI_RE.sub("", out)
    assert len(visible) == 10
    assert visible.endswith("…")
    assert visible.startswith("xxxxxxxxx")

    # ANSI codes inside kept prefix survive.
    colored = "\033[32mhello world\033[0m"
    out = _truncate_visible(colored, 5)
    assert "\033[32m" in out
    visible = _ANSI_RE.sub("", out)
    assert len(visible) == 5
    assert visible.startswith("hell")
    assert visible.endswith("…")

    # Defensive: non-positive cap returns empty.
    assert _truncate_visible("hello", 0) == ""
    assert _truncate_visible("hello", -3) == ""
    print("  test_display_truncate_visible_unit PASS")


def test_display_render_combined_view_truncates_to_cols() -> None:
    """``render_combined_view(cols=N)`` keeps every line within N columns."""
    from auto_iterator.display import _ANSI_RE, render_combined_view

    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        # A line that would wrap a 80-col terminal three times over.
        paths.agent_log.write_text("z" * 500 + "\n", encoding="utf-8")
        out = render_combined_view(
            paths, event_lines=5, log_lines=5, cols=80,
        )
    for line in out.split("\n"):
        visible = _ANSI_RE.sub("", line)
        assert len(visible) <= 80, (
            f"render_combined_view leaked a {len(visible)}-col line "
            f"despite cols=80: {line!r}"
        )
    # Without ``cols`` the same call must not truncate (piped output
    # stays lossless).
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        paths.agent_log.write_text("z" * 500 + "\n", encoding="utf-8")
        out_full = render_combined_view(paths, event_lines=5, log_lines=5)
    assert "z" * 500 in out_full, (
        "non-TTY output must not truncate long lines"
    )
    print("  test_display_render_combined_view_truncates_to_cols PASS")


def test_display_run_live_show_does_not_scroll_at_bottom_row() -> None:
    """Reviewer regression: a perfect-fit redraw must not emit a
    trailing newline that scrolls the alt screen.

    The previous renderer wrote the footer as ``"\\n" + footer_text +
    "\\n"`` and ``fit_section_caps`` reserved exactly two visible rows
    for the footer (one blank + the footer line). On a 24-row
    terminal where status + events + agent output exactly fill the
    remaining 22 rows, the footer text lands on row 24, but the
    final ``\\n`` advances the cursor to row 25 — which most
    terminals interpret as a scroll, knocking the top status row off
    screen on every refresh.

    The fix is to omit the trailing newline. The cursor parks on the
    bottom row at the end of the footer text, and the next tick's
    ``_CLEAR_HOME`` repositions it cleanly. This test pins the
    invariant by:

    1. Forcing a perfect-fit layout with a 24-row terminal and
       enough recent events / agent output to consume the budget.
    2. Slicing the redraw region between the first ``_CLEAR_HOME``
       and the cursor-restore epilogue.
    3. Asserting the redraw does NOT end with ``\\n`` so a future
       refactor that re-introduces a trailing newline trips here.
    4. Asserting the cursor-advance count (``newline_count``) is
       at most ``rows - 1`` so the last cursor row stays inside
       the visible area.
    """
    import io
    from auto_iterator.display import (
        _LIVE_VIEW_OVERHEAD,
        _status_section_lines,
        run_live_show,
    )

    rows = 24
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        # Seed plenty of agent output so the fit is genuinely
        # constrained — the whole budget should be consumed.
        paths.agent_log.write_text(
            "\n".join(f"line {i}" for i in range(1, 201)) + "\n",
            encoding="utf-8",
        )

        out = io.StringIO()
        rc = run_live_show(
            paths,
            event_lines=12,
            log_lines=30,
            refresh_seconds=0.25,
            out=out,
            sleep=lambda _s: None,
            should_continue=lambda i: i < 1,
            get_size=lambda: (80, rows),
        )

    text = out.getvalue()
    assert rc == 0

    clear = "\033[H\033[2J"
    show_cursor = "\033[?25h"
    assert clear in text and show_cursor in text
    redraw = text.split(clear, 1)[1].split(show_cursor, 1)[0]

    # Redraw slice must NOT terminate with a newline. A trailing
    # newline on a perfect-fit redraw is exactly the bug the
    # reviewer flagged.
    assert not redraw.endswith("\n"), (
        "live redraw must not end in '\\n' — that final newline "
        "advances the cursor past the bottom row of the alt screen "
        "and triggers a scroll, knocking the top status row off "
        f"screen: redraw tail={redraw[-40:]!r}"
    )

    # Cursor-position invariant: redraw starts at (1, 1) after
    # ``_CLEAR_HOME``; each ``\n`` advances the cursor by one row.
    # The final cursor row must be ``<= rows`` so no scroll is
    # triggered. Equivalently: ``redraw.count("\n") <= rows - 1``.
    advanced = redraw.count("\n")
    assert advanced <= rows - 1, (
        f"redraw advances cursor to row {advanced + 1} on a "
        f"{rows}-row terminal — bottom-row scroll bug regressed"
    )

    # Sanity check that the layout actually consumed the full
    # budget: if status + section caps + overhead is well under
    # ``rows`` the test isn't exercising the perfect-fit case.
    status_h = len(_status_section_lines(paths))
    used = status_h + _LIVE_VIEW_OVERHEAD + 12 + 30  # requested caps
    assert used > rows, (
        "test setup error: requested caps don't exceed terminal "
        "rows so the fit isn't actually constrained"
    )

    # Status anchor + all three section headers must still be
    # visible after the fit, proving we kept them on screen
    # rather than starving them to fit the footer.
    assert "Run " in redraw
    assert "Recent events" in redraw
    assert "Agent output" in redraw
    print("  test_display_run_live_show_does_not_scroll_at_bottom_row PASS")


def test_display_run_live_show_no_line_exceeds_columns() -> None:
    """Reviewer regression: long lines must not wrap a 24-row terminal.

    The prior fit fixed the *line count* but ignored *line width*. A
    single 500-character agent transcript line would wrap to 7 physical
    rows on an 80-col terminal, eating the status section even though
    the row budget thought everything fit. Pin the fake terminal to
    80x24, seed long agent output and a long workspace path, then
    assert every line in the redraw slice fits within 80 visible
    columns *and* that the redraw slice is at most 24 rows tall."""
    import io
    from auto_iterator.display import _ANSI_RE, run_live_show

    long_path = "/tmp/" + ("nested/" * 30) + "workspace_with_a_very_long_name"
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        # Patch a long workspace so the status section's right column
        # would also overflow if the renderer didn't truncate.
        from auto_iterator.meta import write_meta
        write_meta(paths, {
            "run_id": paths.run_id,
            "pid": 999_999_999,
            "status": "running",
            "workspace": long_path,
            "started_at": now_iso(),
            "heartbeat_at": now_iso(),
        })
        # 500-character single line — wider than 80 cols by 6x.
        long_line = "z" * 500
        paths.agent_log.write_text(
            "\n".join(long_line for _ in range(40)) + "\n",
            encoding="utf-8",
        )

        out = io.StringIO()
        rc = run_live_show(
            paths,
            event_lines=12,
            log_lines=30,
            refresh_seconds=0.25,
            out=out,
            sleep=lambda _s: None,
            should_continue=lambda i: i < 1,
            get_size=lambda: (80, 24),
        )

    text = out.getvalue()
    assert rc == 0
    clear = "\033[H\033[2J"
    show_cursor = "\033[?25h"
    assert clear in text and show_cursor in text
    redraw = text.split(clear, 1)[1].split(show_cursor, 1)[0]

    # Every redraw line must fit in 80 visible columns or the
    # alt-screen will wrap and double its physical row count.
    for raw in redraw.split("\n"):
        visible = _ANSI_RE.sub("", raw)
        assert len(visible) <= 80, (
            f"live redraw leaked a {len(visible)}-col line on a "
            f"80-col terminal: {raw!r}"
        )
    # And the row count itself stays inside the 24-row budget.
    assert redraw.count("\n") <= 24, (
        f"live redraw is {redraw.count(chr(10))} rows for a 24-row terminal"
    )
    # Combined view sections all rendered despite the truncation.
    assert "Run " in redraw
    assert "Recent events" in redraw
    assert "Agent output" in redraw
    print("  test_display_run_live_show_no_line_exceeds_columns PASS")


def test_main_show_keyboard_interrupt_exits_zero(capsys, monkeypatch) -> None:
    """Top-level ``ai show`` swallows KeyboardInterrupt with exit 0.

    Forces the TUI path by pinning ``sys.stdout.isatty`` to True, then
    monkeypatches the lazy import so the pyratatui app entry point
    raises :class:`KeyboardInterrupt`. The wrapper in
    :func:`auto_iterator.cli.main` is the last line of defence against
    a Ctrl-C in the live loop turning into a noisy traceback."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

    def fake_app(*_args, **_kwargs):
        raise KeyboardInterrupt

    # ``cmd_show`` lazy-imports ``auto_iterator.tui`` and calls
    # ``run_detail_app``; patch the symbol on the module so the lazy
    # import sees our fake.
    import auto_iterator.tui as _tui

    monkeypatch.setattr(_tui, "run_detail_app", fake_app, raising=True)
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        rc = main(["--runs-dir", tmp, "show", paths.run_id])
    assert rc == EXIT_OK
    print("  test_main_show_keyboard_interrupt_exits_zero PASS")


# ── Selector helper unit checks ───────────────────────────────────────────────


def test_selector_format_rows_plain() -> None:
    """The plain-format helper must render the spec'd columns."""
    rows = [
        RunRow(
            run_id="20260429T100000Z-aaa", status="running", phase="impl",
            outer=1, inner=2, last_verdict="needs_fixes", exit_code=None,
            approved=False, started_at="2026-04-29T10:00:00+00:00",
            updated_at="2026-04-29T10:01:00+00:00",
            workspace="/home/me/project", prompt_preview="do the thing",
            pid=12345,
        ),
    ]
    out = selector.format_rows_plain(rows)
    assert any("RUN_ID" in line for line in out)
    assert any("20260429T100000Z-aaa" in line for line in out)
    assert any("running" in line for line in out)
    assert any("do the thing" in line for line in out)
    print("  test_selector_format_rows_plain PASS")


def test_selector_is_interactive_when_not_tty(monkeypatch) -> None:
    """Smoke test: piping stdin (or stdout) flips ``is_interactive`` off."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    assert selector.is_interactive() is False
    print("  test_selector_is_interactive_when_not_tty PASS")


def main_runner() -> None:
    print("=" * 60)
    print("Test: ai selector + display layer")
    print("-" * 60)
    # When invoked directly (no pytest) we don't have ``capsys``; skip
    # those so the script-style runner stays usable. Pytest still
    # collects everything via the standard ``test_*`` discovery.
    print("Run with `pytest tests/test_cli_selector.py` for full coverage.")
    print("=" * 60)


if __name__ == "__main__":
    main_runner()
