# auto-iterator

Automated **implement → review → fix** loop using the Cursor CLI. A prompt goes in, and the loop keeps reviewing and fixing until the code is approved — or iteration limits are hit.

## Agent Loop

The loop has three phases: an initial implementation, then a nested review/fix cycle with fresh-eyes validation.

### Steps

1. **Implement** — An agent generates the initial code from the user's prompt (skippable with `--skip-impl`).
2. **Outer loop** (up to `max_outer`) — Each iteration starts a **fresh context** with no prior review history.
   1. **Inner loop** (up to `max_inner`) — Iterates review and fix with accumulated history.
      1. **Review** — A read-only reviewer agent inspects the git diff against main and emits `VERDICT: APPROVED` or `VERDICT: CHANGES_NEEDED`.
      2. If `APPROVED` → break inner loop.
      3. If `CHANGES_NEEDED` → **Fix** — A write-mode agent addresses the findings.
      4. Repeat inner loop with the updated history.
   2. **Fresh-eyes gate** — After the inner loop converges:
      - If the reviewer approved on the **first** inner iteration (no fixes needed), the code is **fully approved** and the loop exits.
      - If fixes were applied before approval, the outer loop continues to get a **fresh-eyes validation** — a brand-new review with no prior history.
      - If the inner loop exhausted its budget without approval, the outer loop retries with fresh context.
3. **Summary** — Print final status: approved or not, total reviews, and loop counts.
