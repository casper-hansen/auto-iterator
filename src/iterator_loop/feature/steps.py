"""High-level step functions: implementation, review, and fix.

Each function wraps a ``run_agent`` call together with the console logging
that surrounds it, so callers see only the semantic result.

When a :class:`~iterator_loop.run_log.RunLogger` is passed in, each step
also emits structured ``*_started`` / ``*_finished`` events and forwards
the agent event sink so the PTY-level stream (tool calls, resumes,
results) lands in the same ``events.jsonl`` as the orchestration events.
"""

from __future__ import annotations

from typing import Optional

from ..agent import run_agent
from ..colors import BOLD, CYAN, GREEN, YELLOW, NC
from .config import RunConfig
from ..logging import _safe_print, hr, log, ok, warn
from ..run_log import RunLogger
from .prompts import build_fix_prompt, build_review_prompt, parse_verdict


async def run_implementation(
    cfg: RunConfig,
    logger: Optional[RunLogger] = None,
) -> int:
    """Run the implementation agent and log the outcome.

    Returns the agent exit code so callers (and tests) can react to it
    without re-parsing console output.
    """
    tag = f"{BOLD}[Impl]{NC}"
    log(f"Implementing feature with {CYAN}{cfg.impl_model}{NC}", tag)
    log(f"Prompt: {cfg.prompt[:120]}...", tag)
    _safe_print()

    sink = logger.agent_event_sink() if logger is not None else None
    if logger is not None:
        logger.implementation_started(model=cfg.impl_model, tag=tag)

    rc, _ = await run_agent(
        model=cfg.impl_model, prompt=cfg.prompt,
        tag=tag, event_sink=sink, **cfg.agent_kw,
    )
    if rc == 0:
        ok("Implementation complete", tag)
    else:
        warn(f"Implementation agent exited with rc={rc}", tag)
    if logger is not None:
        logger.implementation_finished(rc=rc)
    hr()
    _safe_print()
    return rc


async def run_review(
    cfg: RunConfig,
    history: list[dict[str, str]],
    tag: str,
    logger: Optional[RunLogger] = None,
) -> str:
    """Run a review agent, append its output to *history*, return the verdict."""
    log(f"Review — {CYAN}{cfg.reviewer_model}{NC}", tag)

    sink = logger.agent_event_sink() if logger is not None else None
    if logger is not None:
        logger.review_started(model=cfg.reviewer_model, tag=tag)

    rc, review_text = await run_agent(
        model=cfg.reviewer_model,
        prompt=build_review_prompt(cfg.prompt, cfg.context, history),
        tag=tag, event_sink=sink, **cfg.agent_kw,
    )
    history.append({"role": "reviewer", "content": review_text})

    if rc != 0:
        warn(f"Reviewer agent exited with rc={rc}", tag)
        verdict = "CHANGES_NEEDED"
        if logger is not None:
            logger.review_finished(verdict=verdict, rc=rc, tag=tag)
        return verdict

    verdict = parse_verdict(review_text)
    if verdict == "APPROVED":
        ok(f"{GREEN}APPROVED{NC}", tag)
    elif verdict == "CHANGES_NEEDED":
        warn(f"{YELLOW}CHANGES_NEEDED{NC}", tag)
    else:
        warn("Could not parse verdict — treating as CHANGES_NEEDED", tag)
        verdict = "CHANGES_NEEDED"
    if logger is not None:
        logger.review_finished(verdict=verdict, rc=rc, tag=tag)
    return verdict


async def run_fix(
    cfg: RunConfig,
    history: list[dict[str, str]],
    tag: str,
    logger: Optional[RunLogger] = None,
) -> int:
    """Run a fix agent and append its output to *history*. Returns rc."""
    log(f"Fixing issues — {CYAN}{cfg.fix_model}{NC}", tag)

    sink = logger.agent_event_sink() if logger is not None else None
    if logger is not None:
        logger.fix_started(model=cfg.fix_model, tag=tag)

    rc, fix_text = await run_agent(
        model=cfg.fix_model,
        prompt=build_fix_prompt(cfg.prompt, cfg.context, history),
        tag=tag, event_sink=sink, **cfg.agent_kw,
    )
    history.append({"role": "fixer", "content": fix_text})

    if rc == 0:
        ok("Fixes applied", tag)
    else:
        warn(f"Fix agent exited with rc={rc}", tag)
    if logger is not None:
        logger.fix_finished(rc=rc, tag=tag)
    return rc
