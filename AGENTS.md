# Agent instructions for auto-iterator

These instructions apply to **every** agent backend that runs against this repo (Cursor, Claude Code, Codex CLI, …). auto-iterator drives agents through any of those backends, and the testing requirements below are backend-agnostic.

## Testing changes to auto-iterator

When a task involves changing this repo, always exercise the `ai` CLI end-to-end before declaring the work done. Don't rely on unit tests alone — they don't catch argparse wiring, TUI rendering, or detached-runner regressions.

### Always exercise both surfaces

1. **Single CLI commands** — at minimum smoke-test the read paths and any command touched by the change. Examples:
   - `ai ls` and `ai ls --json`
   - `ai show <run_id> --once` (non-TTY path)
   - `ai show <run_id> --json`
   - whichever subcommand the change touched (`run`, `send`, `rewind`, `pause`/`resume`, `apply`/`revert`, `worktree`, `worktree-remove`, `set-prompt`, `kill`, `restart`, …)
   - `ai --help` and `ai <subcommand> --help` if you touched the parser.

2. **Full TUI** — actually open it and drive it with keystrokes:
   - `ai` (bare, no subcommand) → run-list TUI.
   - `ai show <run_id>` in a TTY → per-run detail screen.
   - Verify the keybindings relevant to your change still work (e.g. `n` to spawn a new run, navigation, quit).
   - If you can't open a real TTY in the sandbox, say so explicitly in your summary instead of silently skipping it.

## Self-improvement test task

When auto-iterator is being used to test changes to itself (e.g. spawning a run via `ai run` to validate the loop end-to-end), use this task as the prompt:

> Make a 3D visualization of HBM with read and writes in a single HTML file

It's self-contained, has no external dependencies, and produces a single artifact that's easy to eyeball for "did the loop actually do something useful?".

Example:

```bash
ai run --prompt "Make a 3D visualization of HBM with read and writes in a single HTML file"
```

## Reporting

In the final summary, list the exact CLI invocations you ran and what you observed (output snippet, TUI behavior). If a path was skipped, state why.
