#!/usr/bin/env python3
"""review-loop.py — Automated implement → review → fix loop using the Cursor CLI.

Each invocation produces durable, structured, queryable logs under
``auto-iterator/logs/<run_id>/``:

* ``events.jsonl`` — append-only event stream (lifecycle, orchestration,
  agent stream, tool calls, resumes).
* ``state.json`` — atomically-rewritten snapshot of "what's happening
  right now?" for a future TUI or sibling agent to poll.

The root ``auto-iterator/logs/index.jsonl`` records ``run_started`` and
``run_finished`` lines so operators can enumerate recent runs without
walking every per-run directory. These files are the machine-readable
source of truth; console output is kept for humans but is not parsed by
any other tool.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import traceback
from pathlib import Path

from iterator_loop.feature.config import (
    DEFAULT_FIX_MODEL,
    DEFAULT_IMPL_MODEL,
    DEFAULT_REVIEWER_MODEL,
    RunConfig,
)
from iterator_loop.logging import (
    _safe_print,
    banner,
    err,
    is_stdout_broken,
    log,
    make_tag,
    mark_stdout_broken,
    ok,
    section,
    summary,
    warn,
)
from iterator_loop.feature.steps import run_fix, run_implementation, run_review
from iterator_loop.run_log import InvalidRunIdError, RunLogger

#: Where all per-run log directories and the root ``index.jsonl`` live.
#: Resolved relative to this file so the path is stable regardless of the
#: caller's cwd.
DEFAULT_LOGS_DIR = Path(__file__).resolve().parent / "logs"


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
    p.add_argument(
        "--logs-dir",
        default=str(DEFAULT_LOGS_DIR),
        help="Directory to write structured run logs (default: auto-iterator/logs)",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run id (default: auto-generated timestamp + uuid)",
    )
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


def _parse_config(argv: list[str] | None) -> tuple[RunConfig, Path, str | None]:
    args = _build_parser().parse_args(argv)
    cfg = RunConfig(
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
    logs_dir = Path(args.logs_dir).expanduser().resolve()
    return cfg, logs_dir, args.run_id


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


async def _run_loop(cfg: RunConfig, logger: RunLogger) -> tuple[bool, int, int]:
    """Run the implement → review → fix loop, emitting structured events.

    Returns ``(approved, total_reviews, outer_loops)``. Exceptions are
    allowed to propagate so the caller can emit a ``run_finished`` event
    with the terminal state in a ``finally`` block.
    """
    if not cfg.skip_impl:
        await run_implementation(cfg, logger=logger)
    else:
        log("Skipping implementation (--skip-impl)")
        logger.emit("implementation_skipped", {"reason": "--skip-impl"})
        _safe_print()

    approved = False
    total_reviews = 0
    verdict = ""
    outer = 0

    for outer in range(1, cfg.max_outer + 1):
        section(f"Outer Loop {outer}/{cfg.max_outer} — fresh context")
        logger.outer_started(outer=outer, max_outer=cfg.max_outer)

        history: list[dict[str, str]] = []
        inner = 0

        for inner in range(1, cfg.max_inner + 1):
            total_reviews += 1
            tag = make_tag(outer, inner)
            logger.inner_started(
                outer=outer, inner=inner,
                max_inner=cfg.max_inner, tag=tag,
            )

            verdict = await run_review(cfg, history, tag, logger=logger)
            _safe_print()

            if verdict == "APPROVED":
                ok("Reviewer approved", tag)
                logger.inner_finished(outer=outer, inner=inner, verdict=verdict)
                break

            if inner == cfg.max_inner:
                warn(f"Inner loop exhausted ({cfg.max_inner} iterations)", tag)
                logger.inner_finished(outer=outer, inner=inner, verdict=verdict)
                break

            await run_fix(cfg, history, tag, logger=logger)
            logger.inner_finished(outer=outer, inner=inner, verdict=verdict)
            _safe_print()

        if verdict != "APPROVED":
            warn(
                f"Inner loop did not reach approval after {cfg.max_inner} "
                "iteration(s) — outer loop will retry with fresh context",
            )
            logger.outer_finished(
                outer=outer, approved=False, inner_iterations=inner,
            )
            _safe_print()
            continue

        if inner == 1:
            approved = True
            ok("Approved on first pass" if outer == 1
               else f"Fresh-eyes review approved on outer loop {outer}")
            logger.outer_finished(
                outer=outer, approved=True, inner_iterations=inner,
            )
            break

        ok(
            f"Inner loop converged after {inner} iteration(s) — "
            "starting fresh-eyes validation in next outer loop",
        )
        logger.outer_finished(
            outer=outer, approved=False, inner_iterations=inner,
        )
        _safe_print()

    if not approved:
        if verdict == "APPROVED":
            warn(
                f"Inner loop converged but MAX_OUTER ({cfg.max_outer}) exhausted "
                "without a clean fresh-eyes pass"
            )
        else:
            warn(f"Exhausted {cfg.max_outer} outer loop(s) without reaching approval")

    return approved, total_reviews, outer


async def main(argv: list[str] | None = None) -> int:
    try:
        cfg, logs_dir, run_id = _parse_config(argv)
    except OSError as exc:
        err(str(exc))
        return 1

    if error := cfg.validate():
        err(error)
        return 1

    if not await _command_exists(cfg.agent_cmd):
        err(f"Cursor agent CLI not found ('{cfg.agent_cmd}').")
        _safe_print("Install it with: curl https://cursor.com/install -fsSL | bash")
        return 1

    # Create the structured logger first so even a banner-time crash (or
    # a crash mid-loop) leaves a discoverable run directory behind with
    # whatever state we managed to observe. Surface invalid / colliding
    # ``--run-id`` values as a clean rc=1 exit — not a traceback — so
    # CI / callers get the same UX they do for a bad prompt or config.
    try:
        logger = RunLogger(root_dir=logs_dir, run_id=run_id)
    except InvalidRunIdError as exc:
        err(f"Invalid --run-id: {exc}")
        return 1
    except FileExistsError as exc:
        err(str(exc))
        return 1

    banner_items = cfg.banner_items()
    start_config = {
        "prompt": cfg.prompt,
        "context": cfg.context,
        "impl_model": cfg.impl_model,
        "fix_model": cfg.fix_model,
        "reviewer_model": cfg.reviewer_model,
        "max_outer": cfg.max_outer,
        "max_inner": cfg.max_inner,
        "workspace": cfg.workspace,
        "skip_impl": cfg.skip_impl,
        "extra_flags": list(cfg.extra_flags),
        "agent_cmd": cfg.agent_cmd,
        "started_at": banner_items.get("started_at"),
    }
    # Emit ``run_started`` *before* any human-facing console output so
    # the structured log is authoritative even when stdout is broken
    # (closed pipe, full disk, logged-out terminal, …). Reviewer's
    # reproducer was a ``BrokenPipeError`` raised from ``banner()``:
    # with the old ordering it left only the constructor's blank
    # ``state.json`` behind with no ``run_started`` anywhere, so a
    # consumer polling the root index had no way to discover the run.
    # By starting the logger first, the ``run_started`` record in
    # ``index.jsonl`` / ``events.jsonl`` is durable regardless of
    # whether the subsequent prints succeed.
    logger.start(config=start_config)

    # Human-facing startup block is a courtesy — if stdout is dead the
    # structured logs carry the same config under ``state.config``
    # anyway. The logging helpers are already ``BrokenPipeError``-safe
    # via :func:`iterator_loop.logging._safe_print` (first failure swaps
    # ``fd 1`` to ``/dev/null`` and silently drops the message), so
    # ``banner()`` can no longer propagate here. The try/except is kept
    # as belt-and-suspenders for any surprise raiser — e.g. a future
    # refactor that adds an unguarded ``print`` call — and explicitly
    # triggers :func:`mark_stdout_broken` so subsequent bare ``print``
    # calls throughout the run (and the interpreter-shutdown flush)
    # don't produce ``exit_code=120``. See ``logging.py`` module
    # docstring for the full rationale.
    try:
        banner("Review Loop", banner_items)
        _safe_print(f"Run ID: {logger.run_id}")
        _safe_print(f"Run dir: {logger.run_dir}")
        _safe_print()
    except OSError:
        mark_stdout_broken()

    try:
        approved, total_reviews, outer = await _run_loop(cfg, logger)
        exit_code = 0 if approved else 1
    except BaseException as exc:  # noqa: BLE001 - we re-raise below
        # Capture whatever terminal state we have so index.jsonl +
        # state.json reflect the crash instead of looking like the run is
        # still live. The recorded ``exit_code`` must match what the OS
        # will actually observe once we re-raise — inventing a value
        # here would make the structured logs disagree with reality.
        #
        # Intentionally do *not* pass ``total_reviews`` / ``outer_loops``
        # here. Those locals only land in scope after ``_run_loop()``
        # returns cleanly; when the loop emits progress events
        # (``outer_started``, ``review_started``, …) and *then* raises,
        # the locals are still zero even though ``state.json`` already
        # reflects the real counts via the live event stream. Passing
        # the stale zeros would overwrite the already-correct snapshot
        # in both ``state.json`` and ``index.jsonl`` — the exact
        # regression flagged in review. ``RunLogger.finish()`` falls
        # back to its live in-memory state when the counters are
        # omitted, so the structured logs stay faithful to what the
        # loop actually did before crashing.
        crash_exit_code = _expected_exit_code_for(exc)
        logger.emit("run_error", {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })
        logger.finish(
            approved=False,
            exit_code=crash_exit_code,
        )
        raise

    # Record ``run_finished`` *before* the human-facing summary block so
    # the terminal snapshot is durable even if ``summary()`` raises
    # (reviewer's shutdown repro: ``BrokenPipeError`` from ``summary()``
    # left ``index.jsonl`` with only ``run_started`` and
    # ``state.finished=False`` forever). With this ordering, any
    # console failure below cannot roll back the already-persisted
    # terminal state.
    logger.finish(
        approved=approved,
        exit_code=exit_code,
        total_reviews=total_reviews,
        outer_loops=outer,
    )

    try:
        summary(
            approved=approved,
            total_reviews=total_reviews,
            outer_loops=outer,
            max_outer=cfg.max_outer,
            max_inner=cfg.max_inner,
        )
    except OSError:
        # Matches the banner-block rationale above: the logging helpers
        # already route through ``_safe_print`` so ``summary()`` can't
        # propagate here, but belt-and-suspenders handling keeps the
        # process exit code in sync with ``state.json`` by ensuring
        # ``fd 1`` is redirected to ``/dev/null`` before the
        # interpreter-shutdown flush.
        mark_stdout_broken()
    return exit_code


def _expected_exit_code_for(exc: BaseException) -> int:
    """Return the process exit code Python will produce for *exc*.

    The structured logs must be the authoritative machine-readable source
    of truth for terminal state, which means the recorded ``exit_code``
    must match what the OS will observe once the exception propagates
    — i.e. what ``$?`` in the parent shell will show.

    Empirically (and per CPython's top-level handling + POSIX exit):

    * ``SystemExit`` with ``code=None`` → ``0``.
    * ``SystemExit`` with an ``int`` (``bool`` included) → ``code & 0xFF``.
      POSIX truncates the C-level exit status to 8 bits via
      ``WEXITSTATUS``, so raw values like ``-1``, ``300``, ``256``, or
      ``999999`` are *not* what the shell observes. Verified on Linux:

      * ``SystemExit(-1)`` → shell sees ``255``
      * ``SystemExit(300)`` → shell sees ``44``
      * ``SystemExit(256)`` → shell sees ``0``
      * ``SystemExit(999999)`` → shell sees ``63``

      Recording the raw ``code`` here would make ``state.json`` /
      ``index.jsonl`` disagree with the actual terminal status for this
      crash class. Normalize to the observable byte instead.
    * ``SystemExit`` with a non-int (``str``, ``float``, tuple, …) →
      Python prints ``repr(code)`` to stderr and exits ``1``. (Tested:
      even ``SystemExit(0.0)`` exits ``1``, not ``0``.)
    * ``KeyboardInterrupt`` → ``130`` (shell convention, 128 + SIGINT).
    * Any other uncaught exception → ``1``.

    Note: this helper is POSIX-aware because the project runs on Linux
    (and review-loop.py is only ever invoked via ``uv run``). On
    Windows the truncation semantics differ, but that platform is out
    of scope for this tool.
    """
    if isinstance(exc, SystemExit):
        code = exc.code
        if code is None:
            return 0
        # ``bool`` is a subclass of ``int``; ``True & 0xFF == 1`` and
        # ``False & 0xFF == 0`` match what Python actually exits with.
        if isinstance(code, int):
            return code & 0xFF
        return 1
    if isinstance(exc, KeyboardInterrupt):
        return 130
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
