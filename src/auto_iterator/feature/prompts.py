"""Prompt construction and verdict parsing for the review loop.

Review prompts are built from four ingredients:

1. The *task* (``cfg.task`` at boot; mutated via ``ai set-prompt``).
2. Optional *context* — extra prose the operator wanted the reviewer to
   see on every round (``cfg.context`` / ``ai set-context``).
3. Accumulated review/fix *history* — the two most recent rounds, so the
   prompt stays focused without forgetting the immediately prior feedback.
4. Pending operator *guidance* — one-shot steering text that lands in the
   very next review and then clears.

Keeping guidance separate from context is deliberate: context is sticky
(survives rewind, applies to every round), guidance is ephemeral (applies
once and is consumed)."""

from __future__ import annotations

import re


# Keep the last two rounds of review/fix (current + previous) — earlier
# rounds are dropped to keep prompts focused on what's actionable right
# now.
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


def _format_guidance(guidance: list[str]) -> str:
    if not guidance:
        return ""
    body = "\n".join(f"- {g}" for g in guidance)
    return (
        "\n\n--- Operator guidance (apply in addition to the task above) ---\n\n"
        f"{body}\n\n"
        "--- End guidance ---"
    )


def _format_context(context: str) -> str:
    ctx = (context or "").strip()
    if not ctx:
        return ""
    return f"\n\n--- Additional context ---\n\n{ctx}\n\n--- End context ---"


def build_review_prompt(
    task: str,
    history: list[dict[str, str]],
    *,
    context: str = "",
    guidance: list[str] | None = None,
) -> str:
    preamble = (
        "Inspect the git diff on our branch to main branch. "
        "Review if we have made an excellent implementation of the following:\n\n"
        f"{task}"
    )
    preamble += _format_context(context)
    preamble += _format_guidance(guidance or [])
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
