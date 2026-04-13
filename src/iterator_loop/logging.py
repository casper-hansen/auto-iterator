"""Timestamped console logging helpers."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from .colors import BOLD, DIM, BLUE, GREEN, YELLOW, RED, CYAN, NC


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _emit(
    icon: str,
    color: str,
    msg: str,
    tag: str = "",
    *,
    file: object = None,
) -> None:
    file = file or sys.stdout
    prefix = f"{DIM}{_ts()}{NC} {tag} " if tag else f"{DIM}{_ts()}{NC} "
    print(f"{prefix}{color}{icon}{NC} {msg}", file=file, flush=True)


def log(msg: str, tag: str = "") -> None:
    _emit("▸", BLUE, msg, tag)


def ok(msg: str, tag: str = "") -> None:
    _emit("✓", GREEN, msg, tag)


def warn(msg: str, tag: str = "") -> None:
    _emit("⚠", YELLOW, msg, tag)


def err(msg: str) -> None:
    _emit("✗", RED, msg, file=sys.stderr)


def hr() -> None:
    print(f"{DIM}{'─' * 72}{NC}", flush=True)


def make_tag(outer: int, inner: int) -> str:
    return f"{BOLD}[Outer {outer}, Inner {inner}]{NC}"


def section(msg: str) -> None:
    """Print a bold section header with separator line."""
    print(f"{DIM}{_ts()}{NC} {BOLD}{msg}{NC}")
    hr()


# ── Composite log blocks ─────────────────────────────────────────────────────


def banner(title: str, items: dict[str, object]) -> None:
    """Print a timestamped banner with key-value config lines."""
    print()
    print(f"{DIM}{_ts()}{NC} {BOLD}{title}{NC}")
    hr()
    for k, v in items.items():
        label = k.replace("_", " ").ljust(16)
        print(f"{DIM}{_ts()}{NC}   {label}: {CYAN}{v}{NC}")
    hr()
    print()


def summary(
    approved: bool,
    total_reviews: int,
    outer_loops: int,
    max_outer: int,
    max_inner: int,
) -> None:
    """Print the end-of-run summary block."""
    print()
    hr()
    print(f"{DIM}{_ts()}{NC} {BOLD}Summary{NC}")
    hr()
    counts = f"{total_reviews} review(s), {outer_loops} outer loop(s)"
    if approved:
        print(f"{DIM}{_ts()}{NC}   {GREEN}{BOLD}APPROVED{NC} — {counts}")
    else:
        print(f"{DIM}{_ts()}{NC}   {YELLOW}{BOLD}NOT FULLY APPROVED{NC} — {counts}")
    detail = {
        "approved": approved,
        "total_reviews": total_reviews,
        "outer_loops": outer_loops,
        "max_outer": max_outer,
        "max_inner": max_inner,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    print(f"{DIM}{_ts()}{NC}   {DIM}{json.dumps(detail, indent=2)}{NC}")
    print()
