#!/usr/bin/env python3
"""experiment-loop.py — Run → analyze → iterate loop for reward validation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from auto_iterator.backends import BACKENDS, get_backend
from auto_iterator.experiment.config import ExperimentConfig
from auto_iterator.colors import BOLD, DIM, GREEN, YELLOW, NC
from auto_iterator.logging import banner, err, hr, log, ok, section, warn


# ── CLI ──────────────────────────────────────────────────────────────────────


def _build_parser(be) -> argparse.ArgumentParser:
    """Parser whose model defaults come from the active backend."""
    p = argparse.ArgumentParser(
        prog="experiment-loop",
        description="Run → analyze → iterate loop for reward validation",
    )

    hyp_grp = p.add_mutually_exclusive_group(required=True)
    hyp_grp.add_argument("--hypothesis", help="What you're testing")
    hyp_grp.add_argument("--hypothesis-file", help="Path to hypothesis text file")

    crit_grp = p.add_mutually_exclusive_group(required=True)
    crit_grp.add_argument("--success-criteria", help="Measurable pass/fail conditions")
    crit_grp.add_argument(
        "--success-criteria-file", help="Path to success criteria text file"
    )

    p.add_argument(
        "--baseline-config", required=True,
        help="YAML config for the baseline run (reward disabled)",
    )
    p.add_argument(
        "--experiment-config", required=True,
        help="YAML config for the experiment run (reward enabled)",
    )
    p.add_argument(
        "--training-cmd",
        default=(
            "mkdir -p auto-iterator/logs && cd straw && "
            ".venv/bin/python -m straw.train --config {config_path} "
            "2>&1 | tee ../auto-iterator/logs/{run_name}.log"
        ),
        help="Training command template; {config_path} and {run_name} are replaced per run",
    )
    p.add_argument("--experimenter-model", default=be.default_experimenter_model)
    p.add_argument("--adjuster-model", default=be.default_adjuster_model)
    p.add_argument("--analyst-model", default=be.default_analyst_model)
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--workspace", default=".")
    p.add_argument("--skip-baseline", action="store_true")
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


def _parse_config(argv: list[str] | None) -> ExperimentConfig:
    backend = os.environ.get("AGENT_BACKEND", "cursor")
    be = get_backend(backend)
    args = _build_parser(be).parse_args(argv)
    return ExperimentConfig(
        hypothesis=_load_text(args.hypothesis, args.hypothesis_file, "hypothesis"),
        success_criteria=_load_text(
            args.success_criteria, args.success_criteria_file, "success-criteria"
        ),
        baseline_config=args.baseline_config,
        experiment_config=args.experiment_config,
        training_cmd=args.training_cmd,
        experimenter_model=args.experimenter_model,
        adjuster_model=args.adjuster_model,
        analyst_model=args.analyst_model,
        max_iterations=args.max_iterations,
        workspace=str(Path(args.workspace).resolve()),
        skip_baseline=args.skip_baseline,
        extra_flags=tuple(args.extra_flags),
        agent_cmd=os.environ.get("AGENT_CMD", be.default_cmd),
        backend=backend,
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


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _experiment_summary(
    validated: bool,
    total_analyses: int,
    iterations: int,
    max_iterations: int,
) -> None:
    """Print the end-of-run summary block."""
    print()
    hr()
    print(f"{DIM}{_ts()}{NC} {BOLD}Experiment Summary{NC}")
    hr()
    counts = f"{total_analyses} analysis(es), {iterations} iteration(s)"
    if validated:
        print(f"{DIM}{_ts()}{NC}   {GREEN}{BOLD}VALIDATED{NC} — {counts}")
    else:
        print(f"{DIM}{_ts()}{NC}   {YELLOW}{BOLD}NOT VALIDATED{NC} — {counts}")
    detail = {
        "validated": validated,
        "total_analyses": total_analyses,
        "iterations": iterations,
        "max_iterations": max_iterations,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    print(f"{DIM}{_ts()}{NC}   {DIM}{json.dumps(detail, indent=2)}{NC}")
    print()


# ── Main loop ────────────────────────────────────────────────────────────────


async def main(argv: list[str] | None = None) -> int:
    from auto_iterator.experiment.steps import (
        run_adjustment,
        run_analysis,
        run_baseline,
        run_experiment,
    )

    try:
        cfg = _parse_config(argv)
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
    if not await _command_exists(cfg.agent_cmd):
        err(f"{be.display_name} not found ('{cfg.agent_cmd}').")
        print(be.install_hint)
        return 1

    banner("Experiment Loop", cfg.banner_items())

    # Phase 1: Baseline
    history: list[dict[str, str]] = []

    if not cfg.skip_baseline:
        baseline_text = await run_baseline(cfg)
        history.append({"role": "baseline", "content": baseline_text})
    else:
        log("Skipping baseline (--skip-baseline)")
        print()

    # Phase 2: Experiment iterations
    validated = False
    total_analyses = 0
    iteration = 0

    for iteration in range(1, cfg.max_iterations + 1):
        section(f"Iteration {iteration}/{cfg.max_iterations}")

        # Run experiment
        exp_text = await run_experiment(cfg, iteration)
        history.append({"role": "experiment", "content": exp_text})

        # Analyze results
        total_analyses += 1
        verdict = await run_analysis(cfg, history, iteration)
        print()

        if verdict == "VALIDATED":
            if iteration == 1:
                # First iteration validated — trust it
                validated = True
                ok("Hypothesis validated on first experiment")
                break

            # Validated after adjustments — run fresh-eyes confirmation
            ok(
                f"Validated after {iteration} iteration(s) — "
                "running fresh-eyes confirmation"
            )
            print()

            total_analyses += 1
            fresh_verdict = await run_analysis(
                cfg, history, iteration, fresh_eyes=True,
            )
            print()

            if fresh_verdict == "VALIDATED":
                validated = True
                ok("Fresh-eyes analysis confirmed: hypothesis validated")
                break

            warn("Fresh-eyes analysis did not confirm — continuing iteration")
            print()

        # Not validated — adjust if more iterations remain
        if iteration < cfg.max_iterations:
            await run_adjustment(cfg, history, iteration)
        else:
            warn(f"Exhausted {cfg.max_iterations} iterations without validation")

    _experiment_summary(
        validated=validated,
        total_analyses=total_analyses,
        iterations=iteration,
        max_iterations=cfg.max_iterations,
    )
    return 0 if validated else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
