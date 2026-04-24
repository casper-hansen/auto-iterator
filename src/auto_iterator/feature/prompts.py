"""Prompt construction and verdict parsing for the review loop."""

from __future__ import annotations

import re


# Keep the last two rounds of review/fix (current + previous) — earlier rounds
# are dropped to keep prompts focused on what's actionable right now.
_HISTORY_ROUNDS = 2
_HISTORY_ENTRIES = _HISTORY_ROUNDS * 2


def parse_verdict(text: str) -> str:
    """Extract the last VERDICT line from reviewer output."""
    matches = re.findall(r"VERDICT:\s*(APPROVED|CHANGES_NEEDED)", text)
    return matches[-1] if matches else "UNKNOWN"


def _format_history(history: list[dict[str, str]]) -> str:
    recent = history[-_HISTORY_ENTRIES:]
    parts: list[str] = []
    for i, entry in enumerate(recent, 1):
        label = "Review" if entry["role"] == "reviewer" else "Fix"
        parts.append(f"### Round {i} — {label}\n\n{entry['content']}")
    return "\n\n".join(parts)


def build_review_prompt(task: str, history: list[dict[str, str]]) -> str:
    preamble = (
        "Inspect the git diff on our branch to main branch. "
        "Review if we have made an excellent implementation of the following:\n\n"
        f"{task}"
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


def build_fix_prompt() -> str:
    return "Here are the latest changes addressing your review."
