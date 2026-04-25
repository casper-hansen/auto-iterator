#!/usr/bin/env python3
"""review-loop.py — foreground entry point for the review loop.

Back-compat shim: the loop's semantics now live in
``auto_iterator.runner`` and every invocation (foreground or detached)
creates a run-dir under ``~/.auto-iterator/runs/`` (or
``$AUTO_ITERATOR_RUNS_DIR``) so ``ai ls`` / ``ai tail`` can observe the
run while it's in flight.

For operators who prefer the detached workflow, the recommended entry
point is now ``ai run --prompt …`` (see ``ai --help``).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from auto_iterator.backends import BACKENDS, get_backend
from auto_iterator.cli import load_text_arg as _load_text
from auto_iterator.feature.config import RunConfig
from auto_iterator.logging import err
from auto_iterator.run_dir import create_run_dir, new_run_id, resolve_runs_dir
from auto_iterator.runner import run_review_loop_sync


# ── CLI ──────────────────────────────────────────────────────────────────────


def _build_parser(be) -> argparse.ArgumentParser:
    """Parser whose model defaults come from the active backend."""
    p = argparse.ArgumentParser(
        prog="review-loop",
        description="Automated implement → review → fix loop",
    )
    task_grp = p.add_mutually_exclusive_group(required=True)
    task_grp.add_argument("--task", help="Feature / task description")
    task_grp.add_argument(
        "--task-file",
        help="Path to a UTF-8 text file containing the feature / task description",
    )
    p.add_argument("--impl-model", default=be.default_impl_model)
    p.add_argument("--fix-model", default=be.default_fix_model)
    p.add_argument(
        "--reviewer", default=be.default_reviewer_model, dest="reviewer_model"
    )
    p.add_argument("--max-outer", type=int, default=10)
    p.add_argument("--max-inner", type=int, default=10)
    p.add_argument("--workspace", default=".")
    p.add_argument("--skip-impl", action="store_true")
    p.add_argument("--extra-flags", action="append", default=[])
    p.add_argument("--runs-dir", default=None,
                   help="Override the per-user runs dir (default: "
                        "$AUTO_ITERATOR_RUNS_DIR or ~/.auto-iterator/runs).")
    p.add_argument("--no-worktree", action="store_true",
                   help="Disable per-run git worktree isolation. Default is "
                        "to mount the agent inside <run_dir>/worktree/ on a "
                        "throwaway branch.")
    return p


def _parse_config(argv: list[str] | None):
    backend = os.environ.get("AGENT_BACKEND", "cursor")
    be = get_backend(backend)
    args = _build_parser(be).parse_args(argv)
    cfg = RunConfig(
        task=_load_text(args.task, args.task_file, "task"),
        impl_model=args.impl_model,
        fix_model=args.fix_model or args.impl_model,
        reviewer_model=args.reviewer_model,
        max_outer=args.max_outer,
        max_inner=args.max_inner,
        workspace=str(Path(args.workspace).resolve()),
        skip_impl=args.skip_impl,
        extra_flags=tuple(args.extra_flags),
        agent_cmd=os.environ.get("AGENT_CMD", be.default_cmd),
        backend=backend,
        use_worktree=not args.no_worktree,
    )
    return cfg, args.runs_dir


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


def main(argv: list[str] | None = None) -> int:
    try:
        cfg, runs_dir_override = _parse_config(argv)
    except OSError as exc:
        err(str(exc))
        return 1
    except ValueError as exc:
        err(str(exc))
        print(f"Valid AGENT_BACKEND values: {', '.join(sorted(BACKENDS))}")
        return 1

    if error := cfg.validate():
        err(error)
        return 1

    be = get_backend(cfg.backend)
    if not asyncio.run(_command_exists(cfg.agent_cmd)):
        err(f"{be.display_name} not found ('{cfg.agent_cmd}').")
        print(be.install_hint)
        return 1

    runs_dir = resolve_runs_dir(runs_dir_override)
    paths = create_run_dir(runs_dir, new_run_id())
    print(f"run_id: {paths.run_id}  (dir: {paths.run_dir})", file=sys.stderr)
    from auto_iterator.runner import bootstrap_run
    bootstrap_run(paths, cfg, pid=os.getpid(), agent_type="review-loop")
    return run_review_loop_sync(cfg, paths, agent_type="review-loop")


if __name__ == "__main__":
    sys.exit(main())
