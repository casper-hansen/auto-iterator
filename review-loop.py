#!/usr/bin/env python3
"""review-loop.py — Automated implement → review → fix loop using the Cursor CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pty
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

# Defaults aligned with currently available Cursor models.
DEFAULT_IMPL_MODEL = "claude-4.6-opus-high"
DEFAULT_FIX_MODEL = "claude-4.6-opus-high-thinking"
DEFAULT_REVIEWER_MODEL = "gpt-5.4-xhigh"

# ── Timestamped logging ──────────────────────────────────────────────────────


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str, tag: str = "") -> None:
    prefix = f"{DIM}{_ts()}{NC} {tag} " if tag else f"{DIM}{_ts()}{NC} "
    print(f"{prefix}{BLUE}▸{NC} {msg}", flush=True)


def ok(msg: str, tag: str = "") -> None:
    prefix = f"{DIM}{_ts()}{NC} {tag} " if tag else f"{DIM}{_ts()}{NC} "
    print(f"{prefix}{GREEN}✓{NC} {msg}", flush=True)


def warn(msg: str, tag: str = "") -> None:
    prefix = f"{DIM}{_ts()}{NC} {tag} " if tag else f"{DIM}{_ts()}{NC} "
    print(f"{prefix}{YELLOW}⚠{NC} {msg}", flush=True)


def err(msg: str) -> None:
    print(f"{DIM}{_ts()}{NC} {RED}✗{NC} {msg}", file=sys.stderr, flush=True)


def hr() -> None:
    print(f"{DIM}{'─' * 72}{NC}", flush=True)


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
                print(f"{ts} {self.tag}   {DIM}{after}{NC}", flush=True)
            return

        if _THINKING_CLOSE.search(line):
            before = _THINKING_CLOSE.sub("", line).strip()
            if before:
                print(f"{ts} {self.tag}   {DIM}{before}{NC}", flush=True)
            self._state = self.ASSISTANT
            return

        if self._state == self.THINKING:
            print(f"{ts} {self.tag}   {DIM}{line}{NC}", flush=True)
        else:
            print(f"{ts} {self.tag}   {line}", flush=True)

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
    """Launch the Cursor agent CLI with stream-json output.

    Streams tool-call and assistant-text events to the console as they
    arrive, so the log file is populated in real time.

    Returns (exit_code, captured_full_text).
    """
    cmd: list[str] = [
        agent_cmd, "-p",
        "--output-format", "stream-json",
        "--stream-partial-output",
        "--model", model,
        "--workspace", workspace,
        "--trust",
    ]
    if mode == "write":
        cmd.append("--force")
    cmd.extend(extra_flags)
    cmd.append(prompt)

    master_fd, slave_fd = pty.openpty()

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
    )
    os.close(slave_fd)

    full_text: list[str] = []
    fmt = OutputFormatter(tag)
    text_buf: list[str] = []

    def _flush_text() -> None:
        """Flush any accumulated partial text to the formatter."""
        if not text_buf:
            return
        assembled = "".join(text_buf)
        text_buf.clear()
        for line in assembled.split("\n"):
            fmt.feed(line)
        sys.stdout.flush()

    def _flush_complete_lines() -> None:
        """Flush only complete lines (up to last newline) from text_buf."""
        combined = "".join(text_buf)
        if "\n" not in combined:
            return
        parts = combined.split("\n")
        for l in parts[:-1]:
            fmt.feed(l)
        text_buf.clear()
        if parts[-1]:
            text_buf.append(parts[-1])
        sys.stdout.flush()

    def _read_pty() -> None:
        """Blocking reader in a thread — parses stream-json events."""
        buf = b""
        while True:
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                raw_line, buf = buf.split(b"\n", 1)
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r")
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    fmt.feed(line)
                    sys.stdout.flush()
                    continue
                _handle_event(evt)
        if buf:
            leftover = buf.decode("utf-8", errors="replace").rstrip("\r")
            if leftover:
                try:
                    evt = json.loads(leftover)
                    _handle_event(evt)
                except json.JSONDecodeError:
                    fmt.feed(leftover)
                    sys.stdout.flush()
        try:
            os.close(master_fd)
        except OSError:
            pass

    def _handle_event(evt: dict) -> None:
        etype = evt.get("type", "")

        if etype == "assistant":
            content_parts = evt.get("message", {}).get("content", [])
            ts_ms = evt.get("timestamp_ms")
            has_model_call_id = "model_call_id" in evt
            is_partial = ts_ms is not None and not has_model_call_id

            if is_partial:
                for part in content_parts:
                    if part.get("type") == "text" and part["text"]:
                        text_buf.append(part["text"])
                _flush_complete_lines()
            else:
                _flush_text()
                assembled = "".join(
                    p.get("text", "")
                    for p in content_parts
                    if p.get("type") == "text"
                )
                if assembled:
                    full_text.append(assembled)

        elif etype == "tool_call":
            _flush_text()
            sub = evt.get("subtype", "")
            tc = evt.get("tool_call", {})
            if sub == "started":
                fmt.feed(f"[tool] {_tool_summary(tc)}")
                sys.stdout.flush()
            elif sub == "completed":
                fmt.feed(f"[tool] {_tool_summary(tc, completed=True)}")
                sys.stdout.flush()

        elif etype == "result":
            _flush_text()
            result_text = evt.get("result", "")
            if result_text and not full_text:
                full_text.append(result_text)

    loop = asyncio.get_running_loop()
    read_task = loop.run_in_executor(None, _read_pty)
    await proc.wait()
    await read_task

    fmt.flush()
    return proc.returncode or 0, "\n".join(full_text)


_TOOL_CALL_NOISY_KEYS = {
    "toolCallId", "simpleCommands", "hasInputRedirect", "hasOutputRedirect",
    "parsingResult", "fileOutputThresholdBytes", "isBackground",
    "skipApproval", "timeoutBehavior", "closeStdin",
}


def _tool_summary(tc: dict, *, completed: bool = False) -> str:
    """Return a compact one-line summary of a stream-json tool_call payload."""
    for key in tc:
        if not key.endswith("ToolCall"):
            continue
        name = key.replace("ToolCall", "")
        inner = tc[key]

        if completed and "result" in inner:
            result = inner["result"]
            short = json.dumps(result, separators=(",", ":"))
            if len(short) > 200:
                short = short[:197] + "..."
            return f"{name} => {short}"

        args = {k: v for k, v in inner.get("args", {}).items()
                if k not in _TOOL_CALL_NOISY_KEYS}
        if args:
            short = json.dumps(args, separators=(",", ":"))
            if len(short) > 200:
                short = short[:197] + "..."
            return f"{name}({short})"
        return name
    return json.dumps(tc, separators=(",", ":"))[:200]


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
    prompt_group = p.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Feature / task description")
    prompt_group.add_argument(
        "--prompt-file",
        help="Path to a UTF-8 text file containing the feature / task description",
    )
    context_group = p.add_mutually_exclusive_group()
    context_group.add_argument("--context", default="", help="Extra context for reviewers")
    context_group.add_argument(
        "--context-file",
        help="Path to a UTF-8 text file containing extra reviewer context",
    )
    p.add_argument("--impl-model", default=DEFAULT_IMPL_MODEL)
    p.add_argument("--fix-model", default=DEFAULT_FIX_MODEL)
    p.add_argument("--reviewer", default=DEFAULT_REVIEWER_MODEL, dest="reviewer_model")
    p.add_argument("--max-outer", type=int, default=10)
    p.add_argument("--max-inner", type=int, default=10)
    p.add_argument("--workspace", default=".")
    p.add_argument("--skip-impl", action="store_true")
    p.add_argument("--extra-flags", action="append", default=[])
    return p


async def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    agent_cmd = os.environ.get("AGENT_CMD", "agent")
    try:
        prompt = _load_text_arg(args.prompt, args.prompt_file, label="prompt")
        context = _load_text_arg(args.context, args.context_file, label="context")
    except OSError as exc:
        err(str(exc))
        return 1
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


def _load_text_arg(
    inline_value: str | None,
    file_value: str | None,
    *,
    label: str,
) -> str:
    if file_value:
        path = Path(file_value).expanduser()
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise OSError(f"Could not read --{label}-file '{path}': {exc}") from exc
    return (inline_value or "").strip()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
