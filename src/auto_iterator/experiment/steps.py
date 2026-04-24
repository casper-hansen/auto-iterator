"""High-level step functions: baseline, experiment, analysis, and adjustment.

Each function wraps a ``run_agent`` call together with the console logging
that surrounds it, so callers see only the semantic result.
"""

from __future__ import annotations

from ..agent import run_agent
from ..colors import BOLD, CYAN, GREEN, YELLOW, NC
from ..logging import hr, log, ok, warn

from .config import ExperimentConfig
from .prompts import (
    build_adjust_prompt,
    build_analysis_prompt,
    build_baseline_prompt,
    build_experiment_prompt,
    parse_verdict,
)


async def run_baseline(cfg: ExperimentConfig) -> str:
    """Run the baseline training and return the agent's captured text."""
    tag = f"{BOLD}[Baseline]{NC}"
    log(f"Running baseline with {CYAN}{cfg.experimenter_model}{NC}", tag)
    print()

    prompt = build_baseline_prompt(
        hypothesis=cfg.hypothesis,
        success_criteria=cfg.success_criteria,
        training_cmd=cfg.training_cmd,
        baseline_config=cfg.baseline_config,
    )

    rc, text = await run_agent(
        model=cfg.experimenter_model,
        prompt=prompt,
        tag=tag,
        **cfg.agent_kw,
    )
    if rc == 0:
        ok("Baseline complete", tag)
    else:
        warn(f"Baseline agent exited with rc={rc}", tag)
    hr()
    print()
    return text


async def run_experiment(
    cfg: ExperimentConfig,
    iteration: int,
) -> str:
    """Run the experiment training and return the agent's captured text."""
    tag = f"{BOLD}[Exp {iteration}]{NC}"
    log(f"Running experiment with {CYAN}{cfg.experimenter_model}{NC}", tag)
    print()

    prompt = build_experiment_prompt(
        hypothesis=cfg.hypothesis,
        success_criteria=cfg.success_criteria,
        training_cmd=cfg.training_cmd,
        experiment_config=cfg.experiment_config,
        iteration=iteration,
    )

    rc, text = await run_agent(
        model=cfg.experimenter_model,
        prompt=prompt,
        tag=tag,
        **cfg.agent_kw,
    )
    if rc == 0:
        ok("Experiment run complete", tag)
    else:
        warn(f"Experiment agent exited with rc={rc}", tag)
    hr()
    print()
    return text


async def run_analysis(
    cfg: ExperimentConfig,
    history: list[dict[str, str]],
    iteration: int,
    *,
    fresh_eyes: bool = False,
) -> str:
    """Run the analyst agent, append to *history*, return the verdict."""
    suffix = " (fresh eyes)" if fresh_eyes else ""
    tag = f"{BOLD}[Analyze {iteration}{suffix}]{NC}"
    log(f"Analyzing results with {CYAN}{cfg.analyst_model}{NC}", tag)

    analysis_history = [] if fresh_eyes else history

    prompt = build_analysis_prompt(
        hypothesis=cfg.hypothesis,
        success_criteria=cfg.success_criteria,
        history=analysis_history,
        iteration=iteration,
    )

    rc, analysis_text = await run_agent(
        model=cfg.analyst_model,
        prompt=prompt,
        tag=tag,
        **cfg.agent_kw,
    )
    history.append({"role": "analyst", "content": analysis_text})

    if rc != 0:
        warn(f"Analyst agent exited with rc={rc}", tag)
        return "ITERATE"

    verdict = parse_verdict(analysis_text)
    if verdict == "VALIDATED":
        ok(f"{GREEN}VALIDATED{NC}", tag)
    elif verdict == "ITERATE":
        warn(f"{YELLOW}ITERATE{NC}", tag)
    else:
        warn("Could not parse verdict — treating as ITERATE", tag)
        verdict = "ITERATE"
    return verdict


async def run_adjustment(
    cfg: ExperimentConfig,
    history: list[dict[str, str]],
    iteration: int,
) -> str:
    """Run the adjuster agent, append to *history*, return captured text."""
    tag = f"{BOLD}[Adjust {iteration}]{NC}"
    log(f"Adjusting with {CYAN}{cfg.adjuster_model}{NC}", tag)

    prompt = build_adjust_prompt(
        hypothesis=cfg.hypothesis,
        success_criteria=cfg.success_criteria,
        history=history,
    )

    rc, adjust_text = await run_agent(
        model=cfg.adjuster_model,
        prompt=prompt,
        tag=tag,
        **cfg.agent_kw,
    )
    history.append({"role": "adjuster", "content": adjust_text})

    if rc == 0:
        ok("Adjustment applied", tag)
    else:
        warn(f"Adjuster agent exited with rc={rc}", tag)
    hr()
    print()
    return adjust_text
