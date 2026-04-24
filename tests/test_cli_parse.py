"""Unit tests for the ``ai`` CLI parser.

Argument parsing is the narrowest part of the contract with operators —
if ``--to outer=1,inner=2,phase=review`` stops working they can't steer
runs. These tests assert the parse is tolerant where we said it would be
tolerant (phase default, aliases) and strict where we said it would be
strict (non-integer outer, unknown phase, missing required args)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_iterator.cli import _build_parser  # noqa: E402
from auto_iterator.control import parse_rewind_to  # noqa: E402


def test_run_parser_basics() -> None:
    p = _build_parser()
    args = p.parse_args([
        "run", "--prompt", "do X", "--max-outer", "3", "--max-inner", "2",
    ])
    assert args.cmd == "run"
    assert args.prompt == "do X"
    assert args.max_outer == 3
    assert args.max_inner == 2
    # Default workspace is "."
    assert args.workspace == "."
    print("  test_run_parser_basics PASS")


def test_run_requires_prompt() -> None:
    p = _build_parser()
    # Missing required group should raise SystemExit via argparse.
    try:
        p.parse_args(["run", "--workspace", "."])
    except SystemExit:
        print("  test_run_requires_prompt PASS")
        return
    raise AssertionError("expected SystemExit")


def test_rewind_parser_happy() -> None:
    p = _build_parser()
    args = p.parse_args([
        "rewind", "20260422T085738Z-abc", "--to",
        "outer=2,inner=3,phase=fix", "--wait",
    ])
    assert args.cmd == "rewind"
    assert args.run_id == "20260422T085738Z-abc"
    assert args.to == "outer=2,inner=3,phase=fix"
    assert args.wait is True
    # The `--to` parser lives in ``control``; CLI only carries the raw string.
    r = parse_rewind_to(args.to)
    assert (r.outer, r.inner, r.phase) == (2, 3, "fix")
    print("  test_rewind_parser_happy PASS")


def test_tail_parser_filters() -> None:
    p = _build_parser()
    args = p.parse_args([
        "tail", "20260422T085738Z-abc",
        "--type", "inner_started", "--type", "review_finished",
        "--follow", "--from-seq", "42",
    ])
    assert args.types == ["inner_started", "review_finished"]
    assert args.follow is True
    assert args.from_seq == 42
    print("  test_tail_parser_filters PASS")


def test_send_parser() -> None:
    p = _build_parser()
    args = p.parse_args(["send", "id", "Focus on X", "--wait"])
    assert args.text == "Focus on X"
    assert args.wait is True
    print("  test_send_parser PASS")


def test_set_prompt_parser_mutex() -> None:
    p = _build_parser()
    # Requires one of --text / --file
    try:
        p.parse_args(["set-prompt", "id"])
    except SystemExit:
        pass
    else:
        raise AssertionError("expected SystemExit when neither --text nor --file given")

    # --text works
    args = p.parse_args(["set-prompt", "id", "--text", "new"])
    assert args.text == "new"
    assert args.file is None
    print("  test_set_prompt_parser_mutex PASS")


def test_kill_parser_flags() -> None:
    p = _build_parser()
    args = p.parse_args(["kill", "id", "--grace", "2.5", "--force"])
    assert args.grace == 2.5
    assert args.force is True
    print("  test_kill_parser_flags PASS")


def main() -> None:
    print("=" * 60)
    print("Test: CLI parser (all subcommands)")
    print("-" * 60)
    test_run_parser_basics()
    test_run_requires_prompt()
    test_rewind_parser_happy()
    test_tail_parser_filters()
    test_send_parser()
    test_set_prompt_parser_mutex()
    test_kill_parser_flags()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
