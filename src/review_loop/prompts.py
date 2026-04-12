"""Prompt construction and verdict parsing for the review loop."""

from __future__ import annotations

import re


def parse_verdict(text: str) -> str:
    """Extract the last VERDICT line from reviewer output."""
    matches = re.findall(r"VERDICT:\s*(APPROVED|CHANGES_NEEDED)", text)
    return matches[-1] if matches else "UNKNOWN"


def task_description(prompt: str, context: str) -> str:
    desc = prompt
    if context:
        desc += f"\n\nAdditional context:\n{context}"
    return desc


def _format_history(history: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for i, entry in enumerate(history, 1):
        label = "Review" if entry["role"] == "reviewer" else "Fix"
        parts.append(f"### Round {i} — {label}\n\n{entry['content']}")
    return "\n\n".join(parts)


def build_review_prompt(
    prompt: str, context: str, history: list[dict[str, str]]
) -> str:
    preamble = (
        "Inspect the git diff on our branch to main branch. "
        "Review if we have made an excellent implementation of the following:\n\n"
        f"{task_description(prompt, context)}"
    )
    if history:
        preamble += (
            "\n\n--- Review cycle history (oldest first) ---\n\n"
            f"{_format_history(history)}\n\n"
            "--- End of history ---\n\n"
            "Verify that previous concerns are fixed, but also check for "
            "regressions or any new issues in the overall diff."
        )
    return (
        f"{preamble}\n\n"
        "End your response with exactly one of:\n"
        "VERDICT: APPROVED\n"
        "VERDICT: CHANGES_NEEDED"
    )


def build_fix_prompt(
    prompt: str, context: str, history: list[dict[str, str]]
) -> str:
    return (
        "Fix these findings\n\n"
        "# Task\n\n"
        f"{task_description(prompt, context)}\n\n"
        "# Review cycle history\n\n"
        f"{_format_history(history)}\n\n"
        "Address the issues identified in the latest review above."
    )
