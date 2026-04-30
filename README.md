# auto-iterator

auto-iterator runs an AI coding task as a controlled implementation and review loop. Give it a prompt, and it starts a run that can implement the task, review the diff, fix the findings, and repeat until a fresh review approves the result or the configured limits are hit.

The default CLI entrypoint is `ai`. Runs are filesystem-backed: each run has a run directory with its spec, state, event stream, heartbeat, logs, and optional worktree. There is no long-lived supervisor process; the runner writes files, and the operator commands read or mutate those files.

## Basic Usage

Start a detached run:

```sh
uv run ai run --prompt "Add the feature" --workspace .
```

Watch it:

```sh
uv run ai show <run_id>
```

Run in the foreground when you want the loop output attached to your terminal:

```sh
uv run ai run --foreground --prompt "Add the feature" --workspace .
```

Useful run options:

- `--skip-impl` starts directly with review/fix against the current workspace changes.
- `--max-outer` controls how many fresh-context review passes are allowed.
- `--max-inner` controls how many review/fix rounds are allowed inside one outer pass.
- `--backend`, `--agent-cmd`, `--impl-model`, `--fix-model`, and `--reviewer` choose the agent backend and models.
- `--impl-backend`, `--fix-backend`, `--reviewer-backend` pin individual phases to a different backend — e.g. `--backend claude-code --reviewer-backend codex` runs Claude Code as the implementer/fixer with Codex as a fresh-eyes reviewer. Matching `--impl-cmd`, `--fix-cmd`, `--reviewer-cmd` override the per-phase CLI binary. The matching env vars `AGENT_IMPL_BACKEND` / `AGENT_FIX_BACKEND` / `AGENT_REVIEWER_BACKEND` (and `AGENT_*_CMD`) are honoured by both `ai run` and the bare `ai` TUI's "new run" verb so the same shell config produces the same mixed run from either entry point.
- `--no-worktree` disables the default per-run git worktree isolation.

## The Loop

The loop is structured like this:

1. Implementation, unless `--skip-impl` is set.
   The implementation agent receives the task prompt and writes the first version of the change.

2. Outer loop, up to `--max-outer`.
   Each outer pass is a fresh-eyes review context: review history is cleared, the inner counter is reset, and the reviewer sees the current code without the previous outer pass's conversation.

3. Inner loop, up to `--max-inner` inside each outer pass.
   The reviewer runs against the current task and inner-loop history. The review must end with `VERDICT: APPROVED` or `VERDICT: CHANGES_NEEDED`.

4. Fix, when the reviewer asks for changes.
   If there is inner-loop budget left, the fixer runs with the task, latest review, and prior inner-loop history. The next review sees that accumulated history.

Approval is deliberately strict. If a reviewer approves on the first inner review of an outer pass, the whole run is approved. If the reviewer approves only after one or more fixes, the runner starts another outer pass so a new reviewer context can validate the final diff from scratch. If an inner loop exhausts its budget, the runner also moves to a new outer pass until `--max-outer` is exhausted.

At the end, the runner prints and records whether the run was approved, how many reviews ran, and which loop counts were reached. Approved runs exit with code `0`; unapproved runs exit with code `1`; unexpected runner crashes exit with code `2`.

## Worktrees

For git workspaces, runs use an isolated worktree by default. The runner creates a throwaway branch under the run directory and launches agents there, so the source workspace stays clean while the loop works.

Useful commands:

```sh
uv run ai worktree <run_id>
uv run ai diff <run_id>
uv run ai apply <run_id>
uv run ai revert <run_id>
uv run ai worktree-remove <run_id>
```

If worktree creation fails, or the workspace is not a git repository, the runner falls back to running directly in the requested workspace. Pass `--no-worktree` when direct workspace edits are intentional.

## Operating A Run

`ai show` is the main way to watch a run. In a terminal it shows a live status view with recent structured events and the agent log tail. With `--once`, or when stdout is not a TTY, it prints a single snapshot. `--json` returns the raw `state.json`.

Other operator commands:

- `ai ls` lists known runs.
- `ai send <run_id> "guidance"` queues one-shot guidance for the next review.
- `ai set-prompt <run_id> --text "new prompt"` replaces the task prompt for later review/fix steps.
- `ai pause <run_id>` pauses at the next safe boundary.
- `ai resume <run_id>` lets a paused run continue.
- `ai rewind <run_id> --to outer=N,inner=M,phase=review` jumps back to a loop boundary.
- `ai kill <run_id>` stops a runner and records the run as killed.
- `ai restart <run_id>` starts a new runner from the saved `spec.json`.

Pause, send, set-prompt, and rewind are applied at inner-loop boundaries before a review starts. The runner does not interrupt an agent while it is streaming.

## Run Files

Each run records enough state to be inspected or restarted:

- `spec.json` is the startup configuration.
- `state.json` is the latest loop state.
- `events.jsonl` is the structured event stream used by `ai show`.
- `agent.log` is the runner and agent output.
- `heartbeat` is updated while the runner is alive.
- `control/` contains operator intents such as pause, guidance, and rewind.

The files are side-band state for observation and control; the loop semantics remain the implementation, review/fix, fresh-eyes validation, and final summary described above.
