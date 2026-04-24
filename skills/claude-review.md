# Local branch code review

Provide a rigorous review of the current branch's changes relative to
`main`, including any uncommitted work. There is no GitHub PR — review
the local diff directly.

## Task the change is meant to solve

{{TASK}}

{{HISTORY_BLOCK}}

{{OPERATOR_EXTRAS_BLOCK}}

## Steps

Follow these steps precisely.

1. **Collect context.** Capture the full set of changes on this branch
   relative to `main`, including committed, staged, unstaged, and
   untracked work. At minimum run:
   - `git status --porcelain=v1 -b` — overall shape
   - `git diff main...HEAD` — committed changes on the branch
   - `git diff HEAD` — staged + unstaged changes to tracked files
   - `git log --oneline main..HEAD` — commit messages on the branch

   For every untracked path listed by `git status`, read the file
   contents directly so sub-agents can review them too. Treat these
   additions as "changes on this branch" for every downstream step.

2. **Locate CLAUDE.md files (Haiku agent).** Return paths only (not
   contents) for: the root `CLAUDE.md` (if any), plus any `CLAUDE.md`
   in directories touched by the diff.

3. **Summarise the change (Haiku agent).** Given the diff and commit
   messages, return a 3-5 sentence summary of what the change does and
   why, inferred from the diff and commit messages.

4. **Parallel review (5 Sonnet agents, launched in one message).**
   Each agent returns a list of `{issue, reason}` entries. Share the
   diff and the step-3 summary with every agent.

   a. **CLAUDE.md compliance.** Audit the diff against the CLAUDE.md
      files from step 2. Remember CLAUDE.md is guidance for writing
      code; not every instruction applies during review.

   b. **Shallow bug scan.** Read the diff only. Flag large, obvious
      bugs. Skip nitpicks. Ignore likely false positives.

   c. **Historical context.** Run `git blame` and `git log -p` on the
      modified hunks. Flag bugs that only become visible with history
      (e.g. the change reverts or duplicates a prior fix).

   d. **Prior-change review.** For each modified file, run
      `git log --oneline -- <path>` then `git show` on a few recent
      commits. Surface concerns raised in earlier commits that may
      apply here.

   e. **Inline-comment compliance.** Read code comments in the
      modified files. Verify the change complies with any guidance
      those comments contain.

5. **Confidence filter (parallel Haiku agents).** For every issue
   surfaced in step 4, launch a Haiku agent. Give it the diff, the
   issue, and the CLAUDE.md list from step 2. It returns a score
   0-100 using this rubric verbatim:

   - **0** — Not confident. False positive under light scrutiny, or a
     pre-existing issue.
   - **25** — Somewhat confident. Could be real or a false positive;
     agent could not verify. Stylistic issues that CLAUDE.md doesn't
     explicitly call out land here.
   - **50** — Moderately confident. Verified real, but may be a
     nitpick or rarely hit. Low importance relative to the rest of
     the diff.
   - **75** — Highly confident. Double-checked; very likely to hit in
     practice. Existing approach insufficient. Directly impacts
     functionality, or is directly called out in CLAUDE.md.
   - **100** — Certain. Double-checked; confirmed real and will
     happen frequently. Evidence directly confirms this.

   For CLAUDE.md-flagged issues, the agent must verify that CLAUDE.md
   specifically calls the issue out.

6. **Filter.** Drop any issue scoring less than 80.

7. **Report.** See "Output format" below.

## False positives (exclude in steps 4 and 5)

- Pre-existing issues not introduced by this diff.
- Things that look like bugs but aren't.
- Pedantic nitpicks a senior engineer wouldn't call out.
- Issues a linter / typechecker / compiler catches (imports, types,
  broken tests, formatting). Assume CI runs these.
- General code quality (coverage, docs, security) unless CLAUDE.md
  explicitly requires it.
- Issues CLAUDE.md calls out but the code has explicitly silenced
  (e.g. an ignore comment).
- Intentional changes that are clearly part of the task.
- Real issues on lines the change did not modify.

## Notes

- Do **not** run builds, tests, or typecheckers. Assume CI handles
  them.
- Make a todo list first.
- Every issue in the final report must cite `file:line` from the diff.

## Output format

Write a brief review describing what you checked and any remaining
findings. For each issue, cite `path/to/file.ext:LINE — short
description`.

End your response with **exactly one** of the following lines, on its
own line, and nothing after it:

    VERDICT: APPROVED

(when zero issues remain after filtering in step 6)

    VERDICT: CHANGES_NEEDED

(when one or more issues remain after filtering in step 6)
