#!/usr/bin/env python3
"""review-loop.py — Automated implement → review → fix loop using the Cursor CLI."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from review_loop.config import (
    DEFAULT_FIX_MODEL,
    DEFAULT_IMPL_MODEL,
    DEFAULT_REVIEWER_MODEL,
    RunConfig,
)
from review_loop.logging import banner, err, log, make_tag, ok, section, summary, warn
from review_loop.steps import run_fix, run_implementation, run_review


# ── CLI ──────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="review-loop",
        description="Automated implement → review → fix loop using the Cursor CLI",
    )
    prompt_grp = p.add_mutually_exclusive_group(required=True)
    prompt_grp.add_argument("--prompt", help="Feature / task description")
    prompt_grp.add_argument(
        "--prompt-file",
        help="Path to a UTF-8 text file containing the feature / task description",
    )
    ctx_grp = p.add_mutually_exclusive_group()
    ctx_grp.add_argument("--context", default="", help="Extra context for reviewers")
    ctx_grp.add_argument(
        "--context-file",
        help="Path to a UTF-8 text file containing extra reviewer context",
    )
    p.add_argument("--impl-model", default=DEFAULT_IMPL_MODEL)
    p.add_argument("--fix-model", default=DEFAULT_FIX_MODEL)
    p.add_argument("--reviewer", default=DEFAULT_REVIEWER_MODEL, dest="reviewer_model")
    p.add_argument("--max-outer", type=int, default=10)
    p.add_argument("--max-inner", type=int, default=10)
    p.add_argument("--workspace", default=".")
    p.add_argument("--skip-impl", action="store_true")
    p.add_argument("--extra-flags", action="append", default=[])
    return p


def _load_text(
    inline: str | None,
    file_path: str | None,
    label: str,
) -> str:
    """Resolve a --foo / --foo-file pair to a string."""
    if file_path:
        p = Path(file_path).expanduser()
        try:
            return p.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise OSError(f"Could not read --{label}-file '{p}': {exc}") from exc
    return (inline or "").strip()


def _parse_config(argv: list[str] | None) -> RunConfig:
    args = _build_parser().parse_args(argv)
    return RunConfig(
        prompt=_load_text(args.prompt, args.prompt_file, "prompt"),
        context=_load_text(args.context, args.context_file, "context"),
        impl_model=args.impl_model,
        fix_model=args.fix_model or args.impl_model,
        reviewer_model=args.reviewer_model,
        max_outer=args.max_outer,
        max_inner=args.max_inner,
        workspace=str(Path(args.workspace).resolve()),
        skip_impl=args.skip_impl,
        extra_flags=tuple(args.extra_flags),
        agent_cmd=os.environ.get("AGENT_CMD", "agent"),
    )


async def _command_exists(cmd: str) -> bool:
    try:
        proc = await asyncio.create_subprocess_exec(
            "which", cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return proc.returncode == 0
    except OSError:
        return False


# ── Main loop ────────────────────────────────────────────────────────────────


async def main(argv: list[str] | None = None) -> int:
    try:
        cfg = _parse_config(argv)
    except OSError as exc:
        err(str(exc))
        return 1

    if error := cfg.validate():
        err(error)
        return 1

    if not await _command_exists(cfg.agent_cmd):
        err(f"Cursor agent CLI not found ('{cfg.agent_cmd}').")
        print("Install it with: curl https://cursor.com/install -fsSL | bash")
        return 1

    banner("Review Loop", cfg.banner_items())

    if not cfg.skip_impl:
        await run_implementation(cfg)
    else:
        log("Skipping implementation (--skip-impl)")
        print()

    approved = False
    total_reviews = 0
    verdict = ""
    outer = 0

    for outer in range(1, cfg.max_outer + 1):
        section(f"Outer Loop {outer}/{cfg.max_outer} — fresh context")

        history: list[dict[str, str]] = []
        inner = 0

        for inner in range(1, cfg.max_inner + 1):
            total_reviews += 1
            tag = make_tag(outer, inner)

            verdict = await run_review(cfg, history, tag)
            print()

            if verdict == "APPROVED":
                ok("Reviewer approved", tag)
                break

            if inner == cfg.max_inner:
                warn(f"Inner loop exhausted ({cfg.max_inner} iterations)", tag)
                break

            await run_fix(cfg, history, tag)
            print()

        if verdict != "APPROVED":
            warn(
                f"Inner loop did not reach approval after {cfg.max_inner} "
                "iteration(s) — outer loop will retry with fresh context",
            )
            print()
            continue

        if inner == 1:
            approved = True
            ok("Approved on first pass" if outer == 1
               else f"Fresh-eyes review approved on outer loop {outer}")
            break

        ok(
            f"Inner loop converged after {inner} iteration(s) — "
            "starting fresh-eyes validation in next outer loop",
        )
        print()

    if not approved:
        if verdict == "APPROVED":
            warn(
                f"Inner loop converged but MAX_OUTER ({cfg.max_outer}) exhausted "
                "without a clean fresh-eyes pass"
            )
        else:
            warn(f"Exhausted {cfg.max_outer} outer loop(s) without reaching approval")

    summary(
        approved=approved,
        total_reviews=total_reviews,
        outer_loops=outer,
        max_outer=cfg.max_outer,
        max_inner=cfg.max_inner,
    )
    return 0 if approved else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
