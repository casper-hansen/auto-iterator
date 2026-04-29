"""Tests for the interactive run-selector wired into the ``ai`` CLI.

Covers the spec acceptance criteria without requiring a real terminal:

* Parser accepts omitted ``run_id`` for selector-enabled subcommands.
* In non-TTY mode, an omitted ``run_id`` exits with ``EXIT_USER_ERROR``.
* Explicit ``run_id`` still bypasses selector resolution.
* The selector resolution path picks a ``RunRow`` and sets
  ``args.run_id`` before the existing handler runs.
* ``ai tail`` defaults to 50 events while still respecting an explicit
  ``--lines``.
* ``ai show`` default output is the human-readable status view; it does
  *not* dump raw JSON.
* ``ai show --json`` remains parseable JSON.
* The no-runs case exits cleanly with an error code.

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
        "tail": ["tail"],
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


def test_tail_default_lines_is_50() -> None:
    """Spec change: ``tail`` default went from 200 → 50."""
    p = _build_parser()
    ns = p.parse_args(["tail", "rid"])
    assert ns.lines == 50
    ns_explicit = p.parse_args(["tail", "rid", "--lines", "200"])
    assert ns_explicit.lines == 200
    print("  test_tail_default_lines_is_50 PASS")


def test_tail_agent_log_and_raw_flags() -> None:
    p = _build_parser()
    ns = p.parse_args(["tail", "rid", "--agent-log"])
    assert ns.agent_log is True
    ns = p.parse_args(["tail", "rid", "--raw"])
    assert ns.raw is True
    # ``--json`` is an alias for ``--raw`` to match the documented
    # scripting contract; both must populate the same dest.
    ns = p.parse_args(["tail", "rid", "--json"])
    assert ns.raw is True
    print("  test_tail_agent_log_and_raw_flags PASS")


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


def test_show_logs_flag() -> None:
    p = _build_parser()
    ns = p.parse_args(["show", "rid", "--logs"])
    assert ns.logs is True
    assert ns.lines == 50
    ns2 = p.parse_args(["show", "rid", "--logs", "--lines", "10"])
    assert ns2.lines == 10
    print("  test_show_logs_flag PASS")


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


def test_show_default_is_human_readable(capsys) -> None:
    """``ai show RUN_ID`` (no flags) must print the rendered status view."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        rc = main(["--runs-dir", tmp, "show", paths.run_id])
    out = capsys.readouterr().out
    assert rc == EXIT_OK
    # Rendered view: labelled fields, a Run header, no top-level "{".
    assert out.startswith("Run ") or "Run " in out.splitlines()[0]
    assert "status" in out and "phase" in out and "outer/inner" in out
    assert "workspace" in out
    # No top-level JSON object.
    stripped = out.lstrip()
    assert not stripped.startswith("{"), "show default must not be raw JSON"
    print("  test_show_default_is_human_readable PASS")


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


def test_show_logs_reads_agent_log(capsys) -> None:
    """``ai show RUN_ID --logs`` must surface logs/agent.log content."""
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
    # Last 5 lines of 1..10
    assert "line 10" in out
    assert "line 6" in out
    assert "line 5" not in out  # 1..5 truncated
    print("  test_show_logs_reads_agent_log PASS")


def test_tail_default_renders_events_not_jsonl(capsys, monkeypatch) -> None:
    """Default tail must format events readably; no raw JSON dump."""
    # Disable the TTY footer hint so the assertion stays focused.
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        rc = main(["--runs-dir", tmp, "tail", paths.run_id])
    out = capsys.readouterr().out
    assert rc == EXIT_OK
    assert out, "tail emitted nothing"
    for line in out.strip().splitlines():
        # Each line must NOT be JSON; rendered lines start with HH:MM:SS.
        line = line.strip()
        if not line:
            continue
        assert not line.startswith("{"), \
            f"default tail must not emit JSON lines, got: {line!r}"
    assert "run_started" in out
    assert "review_finished" in out
    print("  test_tail_default_renders_events_not_jsonl PASS")


def test_tail_raw_emits_jsonl(capsys) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        rc = main(["--runs-dir", tmp, "tail", paths.run_id, "--raw"])
    out = capsys.readouterr().out
    assert rc == EXIT_OK
    seqs = []
    for line in out.strip().splitlines():
        obj = json.loads(line)
        seqs.append(obj.get("seq"))
    assert seqs == sorted(seqs) and len(seqs) >= 3
    print("  test_tail_raw_emits_jsonl PASS")


def test_tail_agent_log(capsys) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = _seed_run(Path(tmp))
        paths.agent_log.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        rc = main([
            "--runs-dir", tmp, "tail", paths.run_id, "--agent-log",
            "--lines", "2",
        ])
    out = capsys.readouterr().out
    assert rc == EXIT_OK
    assert "gamma" in out and "beta" in out
    assert "alpha" not in out
    print("  test_tail_agent_log PASS")


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


def test_main_selector_picks_run_and_runs_handler(capsys) -> None:
    """Interactive show with no run_id: selector picks → handler runs."""
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
    assert "status" in out  # rendered view, not raw JSON
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
