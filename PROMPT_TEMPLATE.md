# Task

Describe the change clearly, including the goal and acceptance criteria.

## How to test

- Start with the narrowest deterministic tests closest to the changed code.
- Prefer unit or mock tests before live or networked integration tests.
- Expand to integration coverage only when it materially reduces risk.
- Keep focused integration checks fast when practical.

## How to run

- Prefer `uv run`.
- First identify the owning project or subproject for the files being changed.
- Run commands from the relevant project root rather than assuming the current directory is correct.
- Do not assume `.env` is located at repo root or the current working directory.
- Before concluding that prerequisites are missing, inspect the relevant area for `pyproject.toml`, `README`, existing test files, `Makefile` or `justfile`, and nearby `.env` or `.env.example` files.
- If task-specific execution notes are provided in the private context file, follow them exactly.

## Reviewer guidance

- Do not report “tests cannot run” or “env is missing” until you have checked the relevant subproject root and likely env-file locations.
- Prefer evidence from existing repo commands, configs, and tests over assumptions.
- When something is truly missing, state which commands and paths you checked.

## Privacy

- Keep repo-local operational details out of this public template.
- Put private or repo-specific execution notes in the task’s `*.context.txt`.