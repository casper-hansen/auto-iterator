"""High-level step functions: implementation, review, and fix.

Each function wraps a ``run_agent`` call together with the console logging
that surrounds it, so callers see only the semantic result.
"""

from __future__ import annotations

from ..agent import run_agent
from ..backends import get_backend
from ..colors import BOLD, CYAN, GREEN, YELLOW, NC
from .config import RunConfig
from ..logging import hr, log, ok, warn
from .prompts import build_fix_prompt, build_review_prompt, parse_verdict


async def run_implementation(cfg: RunConfig) -> None:
    """Run the implementation agent and log the outcome."""
    tag = f"{BOLD}[Impl]{NC}"
    log(f"Implementing feature with {CYAN}{cfg.impl_model}{NC} "
        f"({cfg.backend_for('impl')})", tag)
    log(f"Task: {cfg.task[:120]}...", tag)
    print()

    rc, _ = await run_agent(
        model=cfg.impl_model, prompt=cfg.task,
        tag=tag, **cfg.agent_kw_for("impl"),
    )
    if rc == 0:
        ok("Implementation complete", tag)
    else:
        warn(f"Implementation agent exited with rc={rc}", tag)
    hr()
    print()


def compose_review_prompt(
    cfg: RunConfig,
    task: str,
    history: list[dict[str, str]],
    *,
    guidance: list[str],
) -> str:
    """Build the review prompt, honouring any backend-specific template.

    Backends that ship their own ``build_review_prompt`` (e.g. Claude Code
    dispatches ``/ultrareview`` via a markdown skill) take the extras
    directly so they can place them inside the template, *before* any
    terminal output-format instructions. The default path builds the
    whole prompt itself.

    The lookup uses the *reviewer* phase's backend, not the global one,
    so a mixed setup (e.g. Claude Code impl/fix + Codex reviewer) routes
    the prompt through the reviewer CLI's template — Codex's generic
    diff-inspection prompt rather than Claude Code's ultrareview skill."""
    be = get_backend(cfg.backend_for("reviewer"))
    build_prompt = getattr(be, "build_review_prompt", None)
    if build_prompt is not None:
        return build_prompt(task, history, guidance=guidance)
    return build_review_prompt(task, history, guidance=guidance)


async def run_review(
    cfg: RunConfig,
    history: list[dict[str, str]],
    tag: str,
    *,
    task: str,
    guidance: list[str],
) -> tuple[str, str]:
    """Run a review agent, append its output to *history*, return
    ``(verdict, review_text)``.

    ``task`` and ``guidance`` are the runtime-mutable values sourced from
    ``RunState`` — the runner passes them through so operator intents
    dropped between boundaries take effect on the very next review."""
    log(f"Review — {CYAN}{cfg.reviewer_model}{NC} "
        f"({cfg.backend_for('reviewer')})", tag)

    prompt = compose_review_prompt(
        cfg,
        task=task,
        history=history,
        guidance=guidance,
    )

    rc, review_text = await run_agent(
        model=cfg.reviewer_model,
        prompt=prompt,
        tag=tag, **cfg.agent_kw_for("reviewer"),
    )
    history.append({"role": "reviewer", "content": review_text})

    if rc != 0:
        warn(f"Reviewer agent exited with rc={rc}", tag)
        return "CHANGES_NEEDED", review_text

    verdict = parse_verdict(review_text)
    if verdict == "APPROVED":
        ok(f"{GREEN}APPROVED{NC}", tag)
    elif verdict == "CHANGES_NEEDED":
        warn(f"{YELLOW}CHANGES_NEEDED{NC}", tag)
    else:
        warn("Could not parse verdict — treating as CHANGES_NEEDED", tag)
        verdict = "CHANGES_NEEDED"
    return verdict, review_text


async def run_fix(
    cfg: RunConfig,
    history: list[dict[str, str]],
    tag: str,
    *,
    task: str,
) -> tuple[int, str]:
    """Run a fix agent and append its output to *history*.

    The fix agent is launched in a fresh CLI session, so the prompt has
    to carry the full picture: the *task* the implementation is aiming
    at, the latest reviewer feedback (the last entry in *history*), and
    the prior rounds so the agent knows what's already been tried.
    ``task`` is sourced from     ``RunState.prompt`` by the runner and
    threaded through here — same pattern as :func:`run_review` — so any
    runtime ``ai set-prompt`` edit takes effect on the very next fix.

    Returns ``(rc, fix_text)`` so the runner can emit a ``fix_finished``
    event carrying the real exit code instead of guessing."""
    log(f"Fixing issues — {CYAN}{cfg.fix_model}{NC} "
        f"({cfg.backend_for('fix')})", tag)

    rc, fix_text = await run_agent(
        model=cfg.fix_model,
        prompt=build_fix_prompt(task, history),
        tag=tag, **cfg.agent_kw_for("fix"),
    )
    history.append({"role": "fixer", "content": fix_text})

    if rc == 0:
        ok("Fixes applied", tag)
    else:
        warn(f"Fix agent exited with rc={rc}", tag)
    return rc, fix_text
