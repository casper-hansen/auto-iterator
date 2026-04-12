#!/usr/bin/env python3
"""review-loop.py — Automated implement → review → fix loop using the Cursor CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Colors ───────────────────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ

BOLD = "\033[1m" if _USE_COLOR else ""
DIM = "\033[2m" if _USE_COLOR else ""
RED = "\033[0;31m" if _USE_COLOR else ""
GREEN = "\033[0;32m" if _USE_COLOR else ""
YELLOW = "\033[0;33m" if _USE_COLOR else ""
BLUE = "\033[0;34m" if _USE_COLOR else ""
CYAN = "\033[0;36m" if _USE_COLOR else ""
MAGENTA = "\033[0;35m" if _USE_COLOR else ""
NC = "\033[0m" if _USE_COLOR else ""

# ── Timestamped logging ──────────────────────────────────────────────────────


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str, tag: str = "") -> None:
    prefix = f"{DIM}{_ts()}{NC} {tag} " if tag else f"{DIM}{_ts()}{NC} "
    print(f"{prefix}{BLUE}▸{NC} {msg}")


def ok(msg: str, tag: str = "") -> None:
    prefix = f"{DIM}{_ts()}{NC} {tag} " if tag else f"{DIM}{_ts()}{NC} "
    print(f"{prefix}{GREEN}✓{NC} {msg}")


def warn(msg: str, tag: str = "") -> None:
    prefix = f"{DIM}{_ts()}{NC} {tag} " if tag else f"{DIM}{_ts()}{NC} "
    print(f"{prefix}{YELLOW}⚠{NC} {msg}")


def err(msg: str) -> None:
    print(f"{DIM}{_ts()}{NC} {RED}✗{NC} {msg}", file=sys.stderr)


def hr() -> None:
    print(f"{DIM}{'─' * 72}{NC}")


def make_tag(outer: int, inner: int) -> str:
    return f"{BOLD}[Outer {outer}, Inner {inner}]{NC}"


def parse_verdict(text: str) -> str:
    matches = re.findall(r"VERDICT:\s*(APPROVED|CHANGES_NEEDED)", text)
    return matches[-1] if matches else "UNKNOWN"


# ── Output formatter ─────────────────────────────────────────────────────────

_THINKING_OPEN = re.compile(r"<(?:antml:)?thinking>")
_THINKING_CLOSE = re.compile(r"</(?:antml:)?thinking>")


class OutputFormatter:
    """Streams agent output, dimming reasoning and printing messages plainly."""

    THINKING = "thinking"
    ASSISTANT = "assistant"

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self._state = self.ASSISTANT

    def feed(self, line: str) -> None:
        ts = f"{DIM}{_ts()}{NC}"

        if _THINKING_OPEN.search(line):
            self._state = self.THINKING
            after = _THINKING_OPEN.sub("", line).strip()
            if after:
                print(f"{ts} {self.tag}   {DIM}{after}{NC}")
            return

        if _THINKING_CLOSE.search(line):
            before = _THINKING_CLOSE.sub("", line).strip()
            if before:
                print(f"{ts} {self.tag}   {DIM}{before}{NC}")
            self._state = self.ASSISTANT
            return

        if self._state == self.THINKING:
            print(f"{ts} {self.tag}   {DIM}{line}{NC}")
        else:
            print(f"{ts} {self.tag}   {line}")

    def flush(self) -> None:
        self._state = self.ASSISTANT


# ── Agent runner ─────────────────────────────────────────────────────────────


async def run_agent(
    *,
    mode: str,
    model: str,
    prompt: str,
    tag: str,
    workspace: str,
    agent_cmd: str,
    extra_flags: list[str],
) -> tuple[int, str]:
    """Launch the Cursor agent CLI, stream formatted output to stdout.

    Returns (exit_code, captured_output).
    """
    cmd: list[str] = [
        agent_cmd, "-p",
        "--model", model,
        "--workspace", workspace,
        "--trust",
    ]
    if mode == "write":
        cmd.append("--force")
    cmd.extend(extra_flags)
    cmd.append(prompt)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None

    lines: list[str] = []
    fmt = OutputFormatter(tag)

    async for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip("\n")
        lines.append(line)
        fmt.feed(line)

    fmt.flush()
    await proc.wait()
    return proc.returncode or 0, "\n".join(lines)


# ── Prompt builders ──────────────────────────────────────────────────────────


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


# ── Main ─────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="review-loop.py",
        description="Automated implement → review → fix loop using the Cursor CLI",
    )
    p.add_argument("--prompt", required=True, help="Feature / task description")
    p.add_argument("--context", default="", help="Extra context for reviewers")
    p.add_argument("--impl-model", default="claude-opus-4.6")
    p.add_argument("--fix-model", default="claude-opus-4.6-thinking")
    p.add_argument("--reviewer", default="gpt-5.4-xhigh", dest="reviewer_model")
    p.add_argument("--max-outer", type=int, default=10)
    p.add_argument("--max-inner", type=int, default=10)
    p.add_argument("--workspace", default=".")
    p.add_argument("--skip-impl", action="store_true")
    p.add_argument("--extra-flags", action="append", default=[])
    return p


async def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    agent_cmd = os.environ.get("AGENT_CMD", "agent")
    prompt: str = args.prompt
    context: str = args.context
    impl_model: str = args.impl_model
    fix_model: str = args.fix_model or impl_model
    reviewer_model: str = args.reviewer_model
    max_outer: int = args.max_outer
    max_inner: int = args.max_inner
    workspace: str = str(Path(args.workspace).resolve())
    skip_impl: bool = args.skip_impl
    extra_flags: list[str] = args.extra_flags

    if max_outer < 1:
        err(f"--max-outer must be a positive integer (got '{max_outer}')")
        return 1
    if max_inner < 1:
        err(f"--max-inner must be a positive integer (got '{max_inner}')")
        return 1

    if not await _command_exists(agent_cmd):
        err(f"Cursor agent CLI not found ('{agent_cmd}').")
        print("Install it with: curl https://cursor.com/install -fsSL | bash")
        return 1

    agent_kw = dict(
        workspace=workspace,
        agent_cmd=agent_cmd,
        extra_flags=extra_flags,
    )

    # ── Banner ───────────────────────────────────────────────────────────
    print()
    print(f"{DIM}{_ts()}{NC} {BOLD}Review Loop{NC}")
    hr()
    config = {
        "prompt": prompt,
        "context": context,
        "impl_model": impl_model,
        "fix_model": fix_model,
        "reviewer_model": reviewer_model,
        "max_outer": max_outer,
        "max_inner": max_inner,
        "workspace": workspace,
        "skip_impl": skip_impl,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    for k, v in config.items():
        label = k.replace("_", " ").ljust(16)
        print(f"{DIM}{_ts()}{NC}   {label}: {CYAN}{v}{NC}")
    hr()
    print()

    # ── Step 1: Implementation ───────────────────────────────────────────
    if not skip_impl:
        impl_tag = f"{BOLD}[Impl]{NC}"
        log(f"Implementing feature with {CYAN}{impl_model}{NC}", impl_tag)
        log(f"Prompt: {prompt[:120]}...", impl_tag)
        print()

        rc, _ = await run_agent(
            mode="write", model=impl_model, prompt=prompt,
            tag=impl_tag,
            **agent_kw,
        )
        if rc == 0:
            ok("Implementation complete", impl_tag)
        else:
            warn("Implementation agent exited with non-zero status", impl_tag)
        hr()
        print()
    else:
        log("Skipping implementation (--skip-impl)")
        print()

    # ── Step 2: Outer loop ───────────────────────────────────────────────
    approved = False
    total_reviews = 0
    verdict = ""
    outer = 0

    for outer in range(1, max_outer + 1):
        print(f"{DIM}{_ts()}{NC} {BOLD}Outer Loop {outer}/{max_outer} — fresh context{NC}")
        hr()

        history: list[dict[str, str]] = []
        inner = 0
        for inner in range(1, max_inner + 1):
            total_reviews += 1
            tag = make_tag(outer, inner)

            # ── Review ───────────────────────────────────────────────
            log(f"Review — {CYAN}{reviewer_model}{NC}", tag)

            review_prompt = build_review_prompt(prompt, context, history)
            rc, review_text = await run_agent(
                mode="readonly", model=reviewer_model, prompt=review_prompt,
                tag=tag,
                **agent_kw,
            )
            history.append({"role": "reviewer", "content": review_text})

            if rc == 0:
                verdict = parse_verdict(review_text)
                if verdict == "APPROVED":
                    ok(f"{GREEN}APPROVED{NC}", tag)
                elif verdict == "CHANGES_NEEDED":
                    warn(f"{YELLOW}CHANGES_NEEDED{NC}", tag)
                else:
                    warn("Could not parse verdict — treating as CHANGES_NEEDED", tag)
                    verdict = "CHANGES_NEEDED"
            else:
                warn("Reviewer agent exited with non-zero status", tag)
                verdict = "CHANGES_NEEDED"

            print()

            if verdict == "APPROVED":
                ok("Reviewer approved", tag)
                break

            if inner == max_inner:
                warn(f"Inner loop exhausted ({max_inner} iterations)", tag)
                break

            # ── Fix ──────────────────────────────────────────────────
            log(f"Fixing issues — {CYAN}{fix_model}{NC}", tag)

            fix_prompt = build_fix_prompt(prompt, context, history)
            rc, fix_text = await run_agent(
                mode="write", model=fix_model, prompt=fix_prompt,
                tag=tag,
                **agent_kw,
            )
            history.append({"role": "fixer", "content": fix_text})

            if rc == 0:
                ok("Fixes applied", tag)
            else:
                warn("Fix agent exited with non-zero status", tag)
            print()

        if verdict != "APPROVED":
            warn("Inner loop did not reach approval — stopping")
            break

        if inner == 1:
            approved = True
            if outer == 1:
                ok("Approved on first pass")
            else:
                ok(f"Fresh-eyes review found no issues on outer loop {outer}")
            break

        hr()
        print()

    if not approved and verdict == "APPROVED":
        warn(
            f"Inner loop converged but MAX_OUTER ({max_outer}) exhausted "
            "without a clean fresh-eyes pass"
        )

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    hr()
    print(f"{DIM}{_ts()}{NC} {BOLD}Summary{NC}")
    hr()
    summary = {
        "approved": approved,
        "total_reviews": total_reviews,
        "outer_loops": outer,
        "max_outer": max_outer,
        "max_inner": max_inner,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if approved:
        print(f"{DIM}{_ts()}{NC}   {GREEN}{BOLD}APPROVED{NC} — "
              f"{total_reviews} review(s), {outer} outer loop(s)")
    else:
        print(f"{DIM}{_ts()}{NC}   {YELLOW}{BOLD}NOT FULLY APPROVED{NC} — "
              f"{total_reviews} review(s), {outer} outer loop(s)")
    print(f"{DIM}{_ts()}{NC}   {DIM}{json.dumps(summary, indent=2)}{NC}")
    print()

    return 0 if approved else 1


async def _command_exists(cmd: str) -> bool:
    try:
        proc = await asyncio.create_subprocess_exec(
            "which", cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return proc.returncode == 0
    except OSError:
        return False


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
