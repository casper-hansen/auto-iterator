"""Prompt construction and verdict parsing for the experiment loop."""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

VERDICTS = ("VALIDATED", "ITERATE")


def parse_verdict(text: str) -> str:
    """Extract the last VERDICT line from analyst output."""
    matches = re.findall(r"VERDICT:\s*(VALIDATED|ITERATE)", text)
    return matches[-1] if matches else "UNKNOWN"


# ---------------------------------------------------------------------------
# History formatting
# ---------------------------------------------------------------------------


def _format_history(history: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for i, entry in enumerate(history, 1):
        labels = {
            "baseline": "Baseline run",
            "experiment": "Experiment run",
            "analyst": "Analysis",
            "adjuster": "Adjustment",
        }
        label = labels.get(entry["role"], entry["role"])
        parts.append(f"### Round {i} — {label}\n\n{entry['content']}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_baseline_prompt(
    hypothesis: str,
    success_criteria: str,
    training_cmd: str,
    baseline_config: str,
) -> str:
    """Prompt for the agent that runs the baseline experiment."""
    return (
        "Run the **baseline** experiment.\n\n"
        "# Hypothesis\n\n"
        f"{hypothesis}\n\n"
        "# Success criteria\n\n"
        f"{success_criteria}\n\n"
        "# Your task\n\n"
        "Run a baseline training run so we have a control to compare the "
        "experiment against.\n\n"
        f"Config file: `{baseline_config}`\n"
        f"Training command: `{training_cmd.format(config_path=baseline_config, run_name='baseline')}`\n\n"
        "Steps:\n"
        "1. Inspect the baseline config and verify it represents the control "
        "condition (i.e. the feature under test is disabled or absent).\n"
        "2. Run the training command and wait for it to complete.\n"
        "3. Collect the key metrics from the training logs — use the success "
        "criteria above to decide which metrics matter.\n"
        "4. Save a summary of baseline metrics to "
        "`/tmp/experiment-baseline-metrics.txt` so the analyst can find them.\n\n"
        "Focus on collecting clean, comparable metrics. Do not modify the "
        "implementation — this is just data collection.\n\n"
        "# Monitoring constraint\n\n"
        "When waiting for training, poll at least every 30 minutes. "
        "Never sleep for more than 1800 seconds at a time."
    )


def build_experiment_prompt(
    hypothesis: str,
    success_criteria: str,
    training_cmd: str,
    experiment_config: str,
    iteration: int,
) -> str:
    """Prompt for the agent that runs the experiment."""
    return (
        f"Run experiment iteration {iteration}.\n\n"
        "# Hypothesis\n\n"
        f"{hypothesis}\n\n"
        "# Success criteria\n\n"
        f"{success_criteria}\n\n"
        "# Your task\n\n"
        "Run the experiment — a training run with the feature under test "
        "ENABLED.\n\n"
        f"Config file: `{experiment_config}`\n"
        f"Training command: `{training_cmd.format(config_path=experiment_config, run_name=f'experiment-{iteration}')}`\n\n"
        "Steps:\n"
        "1. Inspect the experiment config and verify the feature under test "
        "is enabled.\n"
        "2. Run the training command and wait for it to complete.\n"
        "3. Collect the key metrics from the training logs — use the success "
        "criteria above to decide which metrics matter.\n"
        "4. Save a summary of experiment metrics to "
        f"`/tmp/experiment-iteration-{iteration}-metrics.txt` "
        "so the analyst can find them.\n\n"
        "Focus on collecting clean, comparable metrics. Do not modify the "
        "implementation.\n\n"
        "# Monitoring constraint\n\n"
        "When waiting for training, poll at least every 30 minutes. "
        "Never sleep for more than 1800 seconds at a time."
    )


def build_analysis_prompt(
    hypothesis: str,
    success_criteria: str,
    history: list[dict[str, str]],
    iteration: int,
) -> str:
    """Prompt for the analyst agent that compares baseline vs experiment."""
    preamble = (
        f"Analyze experiment iteration {iteration}.\n\n"
        "# Hypothesis\n\n"
        f"{hypothesis}\n\n"
        "# Success criteria\n\n"
        f"{success_criteria}\n\n"
        "# Your task\n\n"
        "Compare the baseline and experiment runs to determine whether "
        "the hypothesis is supported by the evidence.\n\n"
        "Where to find data:\n"
        "- Baseline metrics: `/tmp/experiment-baseline-metrics.txt`\n"
        f"- Experiment metrics: `/tmp/experiment-iteration-{iteration}-metrics.txt`\n"
        "- Training logs and config files in the workspace\n"
        "- The git diff on our branch shows the implementation\n\n"
        "Analysis steps:\n"
        "1. Read both metric summaries.\n"
        "2. For EACH success criterion, state PASSED or FAILED with "
        "specific numbers.\n"
        "3. Look for signs of unintended side effects or regressions.\n"
        "4. If iterating, be SPECIFIC about what change would help — name "
        "the parameter, threshold, or logic that should be adjusted.\n"
    )

    if history:
        preamble += (
            "\n--- Experiment history (oldest first) ---\n\n"
            f"{_format_history(history)}\n\n"
            "--- End of history ---\n\n"
            "Consider the full history when deciding whether the latest "
            "iteration made progress. If a previous adjustment didn't help, "
            "suggest a different approach.\n"
        )

    return (
        f"{preamble}\n\n"
        "End your response with exactly one of:\n"
        "VERDICT: VALIDATED\n"
        "VERDICT: ITERATE\n\n"
        "Use VALIDATED only if ALL success criteria pass. "
        "Use ITERATE if any criterion fails or evidence is inconclusive."
    )


def build_adjust_prompt(
    hypothesis: str,
    success_criteria: str,
    history: list[dict[str, str]],
) -> str:
    """Prompt for the adjuster agent that tweaks the implementation."""
    return (
        "Adjust the implementation based on the analyst's findings.\n\n"
        "# Hypothesis\n\n"
        f"{hypothesis}\n\n"
        "# Success criteria\n\n"
        f"{success_criteria}\n\n"
        "# Experiment history\n\n"
        f"{_format_history(history)}\n\n"
        "# Your task\n\n"
        "Based on the latest analysis above, make targeted adjustments. "
        "The analyst specified what needs to change.\n\n"
        "Guidelines:\n"
        "- Prefer config/parameter changes over code changes when possible.\n"
        "- If code changes are needed, keep them minimal and focused.\n"
        "- Run the relevant tests to verify nothing is broken.\n"
        "- Explain what you changed and why in your response.\n"
        "- Do NOT run a full training job — that happens in the next "
        "experiment iteration."
    )
