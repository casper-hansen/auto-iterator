"""High-level step functions: implementation, review, and fix.

Each function wraps a ``run_agent`` call together with the console logging
that surrounds it, so callers see only the semantic result.
"""

from __future__ import annotations

from .agent import run_agent
from .colors import BOLD, CYAN, GREEN, YELLOW, NC
from .config import RunConfig
from .logging import hr, log, ok, warn
from .prompts import build_fix_prompt, build_review_prompt, parse_verdict


async def run_implementation(cfg: RunConfig) -> None:
    """Run the implementation agent and log the outcome."""
    tag = f"{BOLD}[Impl]{NC}"
    log(f"Implementing feature with {CYAN}{cfg.impl_model}{NC}", tag)
    log(f"Prompt: {cfg.prompt[:120]}...", tag)
    print()

    rc, _ = await run_agent(
        model=cfg.impl_model, prompt=cfg.prompt,
        tag=tag, **cfg.agent_kw,
    )
    if rc == 0:
        ok("Implementation complete", tag)
    else:
        warn("Implementation agent exited with non-zero status", tag)
    hr()
    print()


async def run_review(
    cfg: RunConfig,
    history: list[dict[str, str]],
    tag: str,
) -> str:
    """Run a review agent, append its output to *history*, return the verdict."""
    log(f"Review — {CYAN}{cfg.reviewer_model}{NC}", tag)

    rc, review_text = await run_agent(
        model=cfg.reviewer_model,
        prompt=build_review_prompt(cfg.prompt, cfg.context, history),
        tag=tag, **cfg.agent_kw,
    )
    history.append({"role": "reviewer", "content": review_text})

    if rc != 0:
        warn("Reviewer agent exited with non-zero status", tag)
        return "CHANGES_NEEDED"

    verdict = parse_verdict(review_text)
    if verdict == "APPROVED":
        ok(f"{GREEN}APPROVED{NC}", tag)
    elif verdict == "CHANGES_NEEDED":
        warn(f"{YELLOW}CHANGES_NEEDED{NC}", tag)
    else:
        warn("Could not parse verdict — treating as CHANGES_NEEDED", tag)
        verdict = "CHANGES_NEEDED"
    return verdict


async def run_fix(
    cfg: RunConfig,
    history: list[dict[str, str]],
    tag: str,
) -> None:
    """Run a fix agent and append its output to *history*."""
    log(f"Fixing issues — {CYAN}{cfg.fix_model}{NC}", tag)

    rc, fix_text = await run_agent(
        model=cfg.fix_model,
        prompt=build_fix_prompt(cfg.prompt, cfg.context, history),
        tag=tag, **cfg.agent_kw,
    )
    history.append({"role": "fixer", "content": fix_text})

    if rc == 0:
        ok("Fixes applied", tag)
    else:
        warn("Fix agent exited with non-zero status", tag)
