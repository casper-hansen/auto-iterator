"""Timestamped console logging helpers.

These helpers are intentionally defensive about stdout failures. A
long-running ``review-loop`` run can outlive its terminal (SSH dropped,
pager quit, piped to ``head``/``grep`` that exits early, …) and the next
write then raises :class:`BrokenPipeError`. Two concrete problems follow
from that in a loop whose authoritative source of truth is the
structured log at ``logs/<run_id>/``:

1. **Loop death mid-work.** Before the defence below was added, a plain
   ``log()`` call halfway through ``_run_loop`` would propagate the
   exception all the way up, aborting the run before any review/fix step
   could emit its structured events. The operator's terminal being
   broken is not a reason to lose a long-running task: the structured
   log is perfectly capable of driving the run without stdout.
2. **Exit-status disagreement with the structured log.** Even if we
   catch the first ``BrokenPipeError`` in application code, Python's
   interpreter-shutdown flush of ``sys.stdout`` hits the same broken
   pipe and exits with status **120** (``_PyIO_cleanup`` in CPython).
   That disagrees with whatever ``exit_code`` we recorded in
   ``state.json`` / ``index.jsonl`` — breaking the task's requirement
   that the structured files be the authoritative source of truth for
   terminal status. Reviewer repro: ``review-loop.py ... | head -0``
   exited ``120`` in the parent shell while ``state.exit_code`` was
   ``1``.

Mitigation, triggered at the *first* ``OSError`` we observe against
stdout: swap ``fd 1`` to ``/dev/null`` via :func:`os.dup2` and rebind
:attr:`sys.stdout` to a fresh file object wrapping that fd. Every
subsequent ``print()`` — including those outside this module and the
interpreter's own shutdown flush — then writes to ``/dev/null`` and
succeeds silently. The recorded exit code stays in sync with whatever
``sys.exit()`` produces.

The swap is idempotent, lock-guarded, and thread-safe (``_StreamReader``
runs the PTY reader in a worker thread and also calls these helpers via
``warn()``).
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone

from .colors import BOLD, DIM, BLUE, GREEN, YELLOW, RED, CYAN, NC


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ── Broken-stdout defence ────────────────────────────────────────────────────

_stdout_lock = threading.Lock()
_stdout_broken = False


def is_stdout_broken() -> bool:
    """Return True once the first ``OSError`` on stdout has been observed.

    Public query for callers that want to skip optional human-facing
    output once the terminal is known to be dead. The structured log
    layer does not need this — it writes to filesystem files that are
    independent of stdout.
    """
    return _stdout_broken


def mark_stdout_broken() -> bool:
    """Swap ``fd 1`` to ``/dev/null`` so further writes silently succeed.

    Idempotent. Safe to call from any thread. Returns ``True`` on the
    *first* call that actually performed the swap, ``False`` on
    subsequent calls (so callers that want to log the event once can
    gate on the return value).

    Why swap fd 1 instead of just setting a flag and skipping writes:

    * There are dozens of ``print()`` / ``sys.stdout.write()`` call
      sites across ``review-loop.py``, ``steps.py``,
      ``output_formatter.py``, and the PTY reader thread. Swapping the
      fd is a single, uniform fix that doesn't require threading a
      "skip stdout" flag through every caller.
    * Python's interpreter-shutdown flush of ``sys.stdout`` is not
      under our control. If fd 1 still points at a closed pipe at
      shutdown, the interpreter exits ``120`` regardless of what we
      wrote to ``state.json``. Redirecting fd 1 to ``/dev/null``
      preemptively makes the final flush succeed, so ``sys.exit(rc)``
      produces the exit code we actually returned.

    If the swap itself fails (very unusual — e.g. ``/dev/null`` not
    openable), we still flip the flag so callers can stop trying to
    write, but we don't raise — the structured log is already the
    source of truth, and propagating here would defeat the whole
    defence.
    """
    global _stdout_broken
    with _stdout_lock:
        if _stdout_broken:
            return False
        _stdout_broken = True
        try:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(devnull_fd, 1)
            finally:
                os.close(devnull_fd)
            # Rebind ``sys.stdout`` to a fresh file object wrapping the
            # now-/dev/null fd. Python's interpreter-shutdown flush
            # targets ``sys.stdout`` (the current attribute), so this
            # ensures the final flush lands in /dev/null and succeeds.
            # The old ``sys.stdout`` object still exists and its
            # finalizer may raise on flush, but that becomes an
            # "Exception ignored" warning, not an exit-code change.
            #
            # ``closefd=False`` is critical: fd 1 is owned by the
            # process, not by this file object. If we let the
            # replacement stream close fd 1 when it's garbage
            # collected (e.g. a test restores the original stream
            # and the replacement becomes unreachable), fd 1 would
            # be closed out from under whatever the test then
            # restored it to, producing ``EBADF`` on the next
            # write. ``closefd=False`` makes the replacement a pure
            # view over fd 1 — matching how Python's default
            # ``sys.stdout`` treats fds 0/1/2.
            try:
                sys.stdout = os.fdopen(
                    1, "w", buffering=1, encoding="utf-8",
                    closefd=False,
                )
            except OSError:
                # If we can't rewrap, the dup2 still protects fd-level
                # writes; future ``print()`` calls routed through the
                # old ``sys.stdout`` object write to /dev/null too
                # because its underlying fd is the same.
                pass
        except OSError:
            # Even opening /dev/null failed. Nothing more to do; the
            # flag is set so callers can stop trying.
            pass
        return True


def _safe_print(*args, **kwargs) -> None:
    """``print()`` that never raises ``OSError`` on stdout failures.

    On the first failure we swap ``fd 1`` to ``/dev/null`` via
    :func:`mark_stdout_broken`, then silently drop the message. All
    subsequent calls succeed because fd 1 now points at ``/dev/null``.

    Writes to ``sys.stderr`` (passed via ``file=sys.stderr``) are also
    guarded, but we never mark stdout broken from a stderr failure —
    stderr dying is a separate concern and should not redirect stdout.
    """
    file = kwargs.get("file")
    try:
        print(*args, **kwargs)
    except OSError:
        if file is None or file is sys.stdout:
            mark_stdout_broken()
        # For any other file (stderr, custom streams), silently drop.


# ── Core helpers ─────────────────────────────────────────────────────────────


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
    _safe_print(f"{prefix}{color}{icon}{NC} {msg}", file=file, flush=True)


def log(msg: str, tag: str = "") -> None:
    _emit("▸", BLUE, msg, tag)


def ok(msg: str, tag: str = "") -> None:
    _emit("✓", GREEN, msg, tag)


def warn(msg: str, tag: str = "") -> None:
    _emit("⚠", YELLOW, msg, tag)


def err(msg: str) -> None:
    _emit("✗", RED, msg, file=sys.stderr)


def hr() -> None:
    _safe_print(f"{DIM}{'─' * 72}{NC}", flush=True)


def make_tag(outer: int, inner: int) -> str:
    return f"{BOLD}[Outer {outer}, Inner {inner}]{NC}"


def section(msg: str) -> None:
    """Print a bold section header with separator line."""
    _safe_print(f"{DIM}{_ts()}{NC} {BOLD}{msg}{NC}")
    hr()


# ── Composite log blocks ─────────────────────────────────────────────────────


def banner(title: str, items: dict[str, object]) -> None:
    """Print a timestamped banner with key-value config lines."""
    _safe_print()
    _safe_print(f"{DIM}{_ts()}{NC} {BOLD}{title}{NC}")
    hr()
    for k, v in items.items():
        label = k.replace("_", " ").ljust(16)
        _safe_print(f"{DIM}{_ts()}{NC}   {label}: {CYAN}{v}{NC}")
    hr()
    _safe_print()


def summary(
    approved: bool,
    total_reviews: int,
    outer_loops: int,
    max_outer: int,
    max_inner: int,
) -> None:
    """Print the end-of-run summary block."""
    _safe_print()
    hr()
    _safe_print(f"{DIM}{_ts()}{NC} {BOLD}Summary{NC}")
    hr()
    counts = f"{total_reviews} review(s), {outer_loops} outer loop(s)"
    if approved:
        _safe_print(f"{DIM}{_ts()}{NC}   {GREEN}{BOLD}APPROVED{NC} — {counts}")
    else:
        _safe_print(f"{DIM}{_ts()}{NC}   {YELLOW}{BOLD}NOT FULLY APPROVED{NC} — {counts}")
    detail = {
        "approved": approved,
        "total_reviews": total_reviews,
        "outer_loops": outer_loops,
        "max_outer": max_outer,
        "max_inner": max_inner,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    _safe_print(f"{DIM}{_ts()}{NC}   {DIM}{json.dumps(detail, indent=2)}{NC}")
    _safe_print()
