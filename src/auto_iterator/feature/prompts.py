"""Prompt construction and verdict parsing for the review loop.

Review and fix prompts are built from the same set of ingredients:

1. The *task* (``cfg.task`` at boot; mutated via ``ai set-prompt``).
2. Accumulated review/fix *history* — the two most recent rounds, so the
   prompt stays focused without forgetting the immediately prior feedback.
3. Pending operator *guidance* — one-shot steering text that lands in the
   very next review and then clears (review prompts only; the fix agent
   sees guidance indirectly through the review it's addressing)."""

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


def build_review_prompt(
    task: str,
    history: list[dict[str, str]],
    *,
    guidance: list[str] | None = None,
) -> str:
    preamble = (
        "Inspect the git diff in our worktree relative to the main branch. "
        "Review if we have made an excellent implementation of the following:\n\n"
        f"{task}"
    )
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


def build_fix_prompt(
    task: str,
    history: list[dict[str, str]],
) -> str:
    """Build the prompt sent to the fix agent.

    The fix agent runs in a fresh session with no carryover from the
    implementer or the reviewer, so it must be handed everything it
    needs to act:

    * the *task* the implementation is aiming at,
    * the *latest review* it's addressing (surfaced explicitly so the
      agent doesn't have to guess which entry is "current"), and
    * the immediately prior review/fix rounds for continuity, so the
      agent can see what was already tried and avoid undoing it.

    *history* must end with a reviewer entry — the runner only invokes
    the fix step after a ``CHANGES_NEEDED`` review, so this is a true
    invariant; raising here surfaces caller bugs loudly instead of
    silently producing a malformed prompt."""
    if not history:
        raise ValueError("build_fix_prompt requires a non-empty history")
    if history[-1]["role"] != "reviewer":
        raise ValueError(
            "build_fix_prompt expects history to end with a reviewer entry; "
            f"got role={history[-1]['role']!r}"
        )

    latest_review = history[-1]["content"]
    prior = history[:-1]

    parts = [
        "Here is the review on your code. Address the review.",
        f"## Original task\n\n{task.strip()}",
    ]
    if prior:
        parts.append(
            "## Previous review/fix rounds (oldest first)\n\n"
            f"{_format_history(prior)}"
        )
    parts.append(f"## Latest review (act on this)\n\n{latest_review.strip()}")
    return "\n\n".join(parts)
