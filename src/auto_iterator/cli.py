"""``ai`` — stateless operator CLI for auto-iterator runs.

Every subcommand is a one-shot process: it opens files in
``<runs-dir>/<run_id>/``, reads or writes, and exits. There is no
long-lived supervisor and no IPC channel to anything; the filesystem is
the protocol.

Subcommand families
-------------------
* Spawn / lifecycle — ``run``, ``restart``, ``kill``.
* Read — ``ls``, ``show`` (the primary observation command).
* Mutate — ``send``, ``rewind``, ``set-prompt``, ``pause``, ``resume``.

``ai show`` is the single user-facing observation experience: in a TTY
it draws a continuously-refreshing combined view (status + recent
structured events + tail of the agent transcript) and exits cleanly
on Ctrl-C; outside a TTY it produces a one-shot text version of the
same combined view. Scripting against the raw event stream is done by
reading ``events.jsonl`` directly — the CLI does not expose a
rendered events-only view.

Exit codes follow the spec:
``0`` success / ``1`` user error / ``2`` IO or permission error /
``3`` targeted run is no longer alive when a mutation is attempted."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from .backends import BACKENDS, get_backend
from .control import parse_rewind_to
from .feature.config import RunConfig
from .ls import list_runs, summarize_run
from .meta import read_meta, update_meta
from .run_dir import (
    CTL_GUIDANCE,
    CTL_PAUSE,
    CTL_PROMPT,
    CTL_REWIND,
    RunPaths,
    atomic_write_json,
    atomic_write_text,
    append_jsonl,
    create_run_dir,
    new_run_id,
    now_iso,
    pid_alive,
    read_json,
    resolve_runs_dir,
    touch,
)
from .runner import bootstrap_run, cfg_to_spec


# ── Exit codes ────────────────────────────────────────────────────────────────


EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_IO_ERROR = 2
EXIT_RUN_GONE = 3


# ── Helpers ───────────────────────────────────────────────────────────────────


@dataclass
class _ResolvedRun:
    paths: RunPaths
    meta: dict


def _resolve_run(runs_dir: Path, run_id: str) -> _ResolvedRun:
    """Look up an existing run by id. Exits with code 1 on not-found.

    Printed errors go to stderr so shell scripts can grep stdout for the
    structured output without noise."""
    paths = RunPaths(runs_dir=runs_dir, run_id=run_id)
    meta = read_meta(paths)
    if meta is None:
        print(
            f"error: run '{run_id}' not found under {runs_dir}",
            file=sys.stderr,
        )
        sys.exit(EXIT_USER_ERROR)
    return _ResolvedRun(paths=paths, meta=meta)


def load_text_arg(
    inline: Optional[str],
    file_path: Optional[str],
    label: str,
) -> str:
    """Resolve a ``--foo`` / ``--foo-file`` arg pair to text.

    Shared between ``ai`` subcommands and the legacy ``review-loop.py``
    wrapper. ``label`` is the short flag name (e.g. ``"prompt"`` or
    ``"task"``); the error message assembles ``--{label}-file``. Raises
    ``OSError`` on read failure so callers can convert to a clean exit
    code."""
    if file_path:
        p = Path(file_path).expanduser()
        try:
            return p.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise OSError(f"could not read --{label}-file '{p}': {exc}") from exc
    return (inline or "").strip()


def _make_cfg_from_args(args: argparse.Namespace) -> RunConfig:
    """Translate the ``ai run`` namespace into a typed :class:`RunConfig`."""
    backend = args.backend or os.environ.get("AGENT_BACKEND", "cursor")
    if backend not in BACKENDS:
        valid = ", ".join(sorted(BACKENDS))
        raise ValueError(f"unknown backend '{backend}'. valid: {valid}")
    be = BACKENDS[backend]
    cfg = RunConfig(
        task=load_text_arg(args.prompt, args.prompt_file, "prompt"),
        impl_model=args.impl_model or be.default_impl_model,
        fix_model=args.fix_model or be.default_fix_model,
        reviewer_model=args.reviewer_model or be.default_reviewer_model,
        max_outer=args.max_outer,
        max_inner=args.max_inner,
        workspace=str(Path(args.workspace).expanduser().resolve()),
        skip_impl=args.skip_impl,
        extra_flags=tuple(args.extra_flags or []),
        agent_cmd=args.agent_cmd or os.environ.get("AGENT_CMD", be.default_cmd),
        backend=backend,
        use_worktree=not getattr(args, "no_worktree", False),
    )
    err = cfg.validate()
    if err:
        raise ValueError(err)
    if not cfg.task:
        raise ValueError("--prompt (or --prompt-file) must be non-empty")
    return cfg


# ── Parser ────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ai",
        description="Auto-iterator operator CLI (filesystem-backed runs).",
    )
    p.add_argument(
        "--runs-dir",
        default=None,
        help="Per-user runs directory (default: $AUTO_ITERATOR_RUNS_DIR or "
             "~/.auto-iterator/runs).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── run ──
    run_p = sub.add_parser("run", help="Start a new review-loop run (detached).")
    prompt_g = run_p.add_mutually_exclusive_group(required=True)
    prompt_g.add_argument("--prompt", help="Feature / task description.")
    prompt_g.add_argument("--prompt-file", help="Path to a UTF-8 prompt file.")
    run_p.add_argument("--impl-model", default=None)
    run_p.add_argument("--fix-model", default=None)
    run_p.add_argument("--reviewer", default=None, dest="reviewer_model")
    run_p.add_argument("--max-outer", type=int, default=10)
    run_p.add_argument("--max-inner", type=int, default=10)
    run_p.add_argument("--workspace", default=".")
    run_p.add_argument("--skip-impl", action="store_true")
    run_p.add_argument("--extra-flags", action="append", default=[])
    run_p.add_argument("--agent-type", default="review-loop",
                       choices=["review-loop"])
    run_p.add_argument("--backend", default=None,
                       help="Override $AGENT_BACKEND (cursor/claude-code/codex).")
    run_p.add_argument("--agent-cmd", default=None,
                       help="Override the backend's default CLI binary name.")
    run_p.add_argument("--foreground", action="store_true",
                       help="Run in the foreground (don't detach). Useful for "
                            "debugging and for legacy review-loop.py parity.")
    run_p.add_argument("--no-worktree", action="store_true",
                       help="Disable per-run git worktree isolation. Default is "
                            "to mount the agent inside <run_dir>/worktree/ on a "
                            "throwaway branch.")

    # ── restart ──
    restart_p = sub.add_parser("restart",
                               help="Kill a run and respawn from its spec.json.")
    restart_p.add_argument("run_id", nargs="?")
    restart_p.add_argument("--grace", type=float, default=5.0,
                           help="Seconds to wait after SIGTERM before SIGKILL.")
    restart_p.add_argument("--yes", "-y", action="store_true",
                           help="Skip the confirmation prompt that follows "
                                "the interactive selector.")

    # ── kill ──
    kill_p = sub.add_parser("kill", help="Signal a running runner and wait.")
    kill_p.add_argument("run_id", nargs="?")
    kill_p.add_argument("--grace", type=float, default=5.0)
    kill_p.add_argument("--force", action="store_true",
                        help="Skip SIGTERM and send SIGKILL immediately.")
    kill_p.add_argument("--yes", "-y", action="store_true",
                        help="Skip the confirmation prompt that follows "
                             "the interactive selector.")

    # ── ls ──
    ls_p = sub.add_parser("ls", help="List runs across all workspaces.")
    ls_p.add_argument("--json", action="store_true",
                      help="Emit one JSON object per run on stdout.")

    # ── show ──
    # ``ai show`` is the primary observation command. In a TTY it
    # draws a live combined view (status + events + agent output);
    # outside a TTY (or with ``--once``) it prints the same view once
    # and exits. ``--json`` keeps emitting raw ``state.json`` so scripts
    # don't need to learn the new renderer.
    show_p = sub.add_parser(
        "show",
        help="Live combined view of a run (status + events + agent output).",
    )
    show_p.add_argument("run_id", nargs="?")
    show_p.add_argument("--json", action="store_true",
                        help="Emit raw state.json (one-shot, scriptable). "
                             "Bypasses the combined renderer.")
    show_p.add_argument("--once", action="store_true",
                        help="Print the combined view once and exit "
                             "(no live refresh). Implied when stdout is "
                             "not a TTY.")
    show_p.add_argument("--event-lines", type=int, default=12,
                        help="Recent events to show in the combined view "
                             "(default 12).")
    show_p.add_argument("--log-lines", type=int, default=30,
                        help="Agent-output tail lines to show in the "
                             "combined view (default 30).")
    show_p.add_argument("--lines", type=int, default=None,
                        help="Backwards-compat alias: sets --log-lines.")
    show_p.add_argument("--refresh", type=float, default=0.4,
                        help="Live refresh interval in seconds "
                             "(default 0.4, min 0.05).")
    # ``--logs`` predates the combined view; it used to drop into a raw
    # ``agent.log`` tail. It is now an internal alias for ``--once`` so
    # any scripts that still invoke it keep producing readable output.
    show_p.add_argument("--logs", action="store_true",
                        help=argparse.SUPPRESS)

    # ── send ──
    # Both positionals are ``nargs="?"`` so argparse can parse the four
    # supported forms uniformly:
    #   ai send TEXT
    #   ai send RUN_ID TEXT
    #   ai send --wait TEXT
    #   ai send RUN_ID --wait TEXT
    # The single-positional case (``run_id`` set, ``text`` is None) is
    # disambiguated post-parse by :func:`_normalize_send_args` — argparse
    # has no way to express "the lone positional is *text*, not run_id"
    # while still accepting "RUN_ID TEXT" with optional flags interleaved.
    send_p = sub.add_parser("send", help="Queue operator guidance for the next review.")
    send_p.add_argument("run_id", nargs="?")
    send_p.add_argument("text", nargs="?")
    send_p.add_argument("--wait", action="store_true",
                        help="Block until the runner records guidance_received.")

    # ── rewind ──
    rewind_p = sub.add_parser("rewind",
                              help="Jump back to (outer, inner, phase).")
    rewind_p.add_argument("run_id", nargs="?")
    rewind_p.add_argument("--to", required=True,
                          help="outer=N,inner=M[,phase=review|fix|after_impl]")
    rewind_p.add_argument("--wait", action="store_true")

    # ── set-prompt ──
    sp_p = sub.add_parser("set-prompt",
                          help="Replace the review-loop's task prompt.")
    sp_p.add_argument("run_id", nargs="?")
    sp_g = sp_p.add_mutually_exclusive_group(required=True)
    sp_g.add_argument("--text")
    sp_g.add_argument("--prompt-file", dest="file")
    sp_p.add_argument("--wait", action="store_true")

    # ── pause / resume ──
    pause_p = sub.add_parser("pause", help="Stall the runner at the next boundary.")
    pause_p.add_argument("run_id", nargs="?")
    resume_p = sub.add_parser("resume", help="Let a paused runner continue.")
    resume_p.add_argument("run_id", nargs="?")

    # ── worktree subcommands ──
    wt_p = sub.add_parser("worktree", help="Print the run's worktree path.")
    wt_p.add_argument("run_id", nargs="?")

    diff_p = sub.add_parser(
        "diff",
        help="Show the worktree's changes vs the source workspace's base commit.",
    )
    diff_p.add_argument("run_id", nargs="?")
    diff_p.add_argument("--full", action="store_true",
                        help="Emit the full unified diff instead of the per-file summary.")
    diff_p.add_argument("--stat", action="store_true",
                        help="Emit ``git diff --stat`` instead of the per-file summary.")

    apply_p = sub.add_parser(
        "apply",
        help="Apply the worktree's changes to the source workspace.",
    )
    apply_p.add_argument("run_id", nargs="?")
    apply_p.add_argument("--yes", "-y", action="store_true",
                         help="Skip the confirmation prompt that follows "
                              "the interactive selector.")

    revert_p = sub.add_parser(
        "revert",
        help="Reverse a previous ``ai apply`` in the source workspace.",
    )
    revert_p.add_argument("run_id", nargs="?")
    revert_p.add_argument("--yes", "-y", action="store_true",
                          help="Skip the confirmation prompt that follows "
                               "the interactive selector.")

    wtrm_p = sub.add_parser(
        "worktree-remove",
        help="Delete the run's git worktree and its branch.",
    )
    wtrm_p.add_argument("run_id", nargs="?")
    wtrm_p.add_argument(
        "--force", action="store_true",
        help="Remove the worktree even if its changes are still applied "
             "to the source workspace. The recorded patch is preserved "
             "so `ai revert` can still undo the apply later.",
    )
    wtrm_p.add_argument("--yes", "-y", action="store_true",
                        help="Skip the confirmation prompt that follows "
                             "the interactive selector.")

    return p


# ── Subcommand impls ──────────────────────────────────────────────────────────


def cmd_run(args: argparse.Namespace, runs_dir: Path) -> int:
    try:
        cfg = _make_cfg_from_args(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USER_ERROR

    run_id = new_run_id()
    paths = create_run_dir(runs_dir, run_id)

    if args.foreground:
        # Foreground path: same process runs the loop; stdout stays attached.
        bootstrap_run(paths, cfg, pid=os.getpid(), agent_type=args.agent_type)
        print(f"run_id: {run_id}", file=sys.stderr)
        from .runner import run_review_loop_sync
        return run_review_loop_sync(cfg, paths, agent_type=args.agent_type)

    # Detached path: spawn a new-session child that re-enters this module
    # via ``python -m auto_iterator.runner <run_dir>``. stdout/stderr go
    # to ``logs/agent.log`` so SSH disconnect doesn't kill the runner and
    # humans still have a raw transcript to grep.
    atomic_write_json(paths.spec, cfg_to_spec(cfg, agent_type=args.agent_type))
    update_meta(
        paths,
        run_id=run_id,
        pid=0,  # placeholder; the child stamps its real pid on startup
        status="running",
        started_at=now_iso(),
        finished_at=None,
        workspace=cfg.workspace,
        agent_type=args.agent_type,
        heartbeat_at=now_iso(),
    )
    append_jsonl(paths.index, {
        "event": "run_started",
        "timestamp": now_iso(),
        "run_id": run_id,
        "workspace": cfg.workspace,
        "agent_type": args.agent_type,
    })

    log_fh = open(paths.agent_log, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "auto_iterator.runner", str(paths.run_dir)],
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
            close_fds=True,
            cwd=cfg.workspace,
        )
    except OSError as exc:
        print(f"error: failed to spawn runner: {exc}", file=sys.stderr)
        return EXIT_IO_ERROR
    finally:
        log_fh.close()

    update_meta(paths, pid=proc.pid)
    print(run_id)
    return EXIT_OK


def cmd_restart(args: argparse.Namespace, runs_dir: Path) -> int:
    run = _resolve_run(runs_dir, args.run_id)
    try:
        spec = read_json(run.paths.spec)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read spec.json: {exc}", file=sys.stderr)
        return EXIT_IO_ERROR

    # Kill old runner first (best-effort) so it can't race the new one.
    _signal_runner(run, grace=args.grace, force=False)

    # Spawn a fresh run-dir from the recorded spec.
    from .runner import spec_to_cfg
    try:
        cfg = spec_to_cfg(spec)
    except KeyError as exc:
        print(f"error: spec.json missing field {exc}", file=sys.stderr)
        return EXIT_IO_ERROR

    agent_type = spec.get("agent_type", "review-loop")
    new_id = new_run_id()
    new_paths = create_run_dir(runs_dir, new_id)
    atomic_write_json(new_paths.spec, cfg_to_spec(cfg, agent_type=agent_type))
    update_meta(
        new_paths,
        run_id=new_id,
        pid=0,
        status="running",
        started_at=now_iso(),
        finished_at=None,
        workspace=cfg.workspace,
        agent_type=agent_type,
        heartbeat_at=now_iso(),
        restarted_from=args.run_id,
    )
    append_jsonl(new_paths.index, {
        "event": "run_started",
        "timestamp": now_iso(),
        "run_id": new_id,
        "restarted_from": args.run_id,
        "workspace": cfg.workspace,
        "agent_type": agent_type,
    })
    log_fh = open(new_paths.agent_log, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "auto_iterator.runner", str(new_paths.run_dir)],
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
            close_fds=True,
            cwd=cfg.workspace,
        )
    except OSError as exc:
        print(f"error: failed to spawn runner: {exc}", file=sys.stderr)
        return EXIT_IO_ERROR
    finally:
        log_fh.close()
    update_meta(new_paths, pid=proc.pid)
    print(new_id)
    return EXIT_OK


def cmd_kill(args: argparse.Namespace, runs_dir: Path) -> int:
    run = _resolve_run(runs_dir, args.run_id)
    if _signal_runner(run, grace=args.grace, force=args.force):
        return EXIT_OK
    return EXIT_RUN_GONE


def _signal_runner(run: _ResolvedRun, *, grace: float, force: bool) -> bool:
    """SIGTERM → wait → SIGKILL. Returns True if we signalled a live pid."""
    pid = run.meta.get("pid")
    if not isinstance(pid, int) or not pid_alive(pid):
        update_meta(run.paths, status=run.meta.get("status", "crashed"))
        return False

    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError) as exc:
        print(f"warning: could not signal pid {pid}: {exc}", file=sys.stderr)
        return False

    deadline = time.time() + max(0.0, grace)
    while time.time() < deadline and pid_alive(pid):
        time.sleep(0.1)

    if pid_alive(pid):
        # Escalate to SIGKILL regardless of the initial signal.
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        # Short second wait so meta lands as "killed" reliably.
        t0 = time.time()
        while time.time() - t0 < 2.0 and pid_alive(pid):
            time.sleep(0.05)

    update_meta(run.paths, status="killed", finished_at=now_iso())
    return True


def cmd_ls(args: argparse.Namespace, runs_dir: Path) -> int:
    rows = list_runs(runs_dir)
    if args.json:
        for row in rows:
            sys.stdout.write(json.dumps(row.as_dict()) + "\n")
        sys.stdout.flush()
        return EXIT_OK
    _print_ls_table(rows)
    return EXIT_OK


def _print_ls_table(rows: list) -> None:
    """Tabular ``ls`` output — one line per run, columns aligned.

    Kept simple (no external tabulate dep) so ``ai`` stays importable
    with only stdlib. Columns mirror the JSON fields for easy mental
    mapping."""
    if not rows:
        print("(no runs)")
        return
    cols = [
        ("RUN_ID", 26),
        ("STATUS", 10),
        ("PHASE", 12),
        ("O/I", 7),
        ("VERDICT", 15),
        ("UPDATED", 25),
        ("PROMPT", 40),
    ]
    header = "  ".join(name.ljust(w) for name, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = "  ".join(s.ljust(w) for s, (_name, w) in zip([
            r.run_id,
            r.status,
            r.phase or "",
            f"{r.outer}/{r.inner}",
            r.last_verdict or "",
            (r.updated_at or "")[:25],
            (r.prompt_preview or "").replace("\n", " ")[:40],
        ], cols))
        print(line)


def _stdout_is_tty() -> bool:
    """Single check used by the show dispatcher; tolerates patched sys.stdout."""
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, OSError):
        return False


def cmd_show(args: argparse.Namespace, runs_dir: Path) -> int:
    """Render a run.

    Default flow:

    * ``--json`` → one-shot raw ``state.json`` (scriptable).
    * Non-TTY stdout, or ``--once`` / ``--logs`` → one-shot combined view.
    * Interactive TTY → live combined view, refreshed on a timer until
      Ctrl-C.
    """
    run = _resolve_run(runs_dir, args.run_id)
    from .display import (
        render_combined_view,
        run_live_show,
        state_json_text,
    )

    if args.json:
        sys.stdout.write(state_json_text(run.paths))
        return EXIT_OK

    event_lines = max(1, int(getattr(args, "event_lines", 12) or 12))
    log_lines_default = int(getattr(args, "log_lines", 30) or 30)
    log_lines = max(1, log_lines_default)
    if getattr(args, "lines", None) is not None:
        # Backwards-compat alias: ``--lines`` continues to control the
        # agent-output tail size.
        log_lines = max(1, int(args.lines))

    once = bool(getattr(args, "once", False) or getattr(args, "logs", False))
    if once or not _stdout_is_tty():
        sys.stdout.write(render_combined_view(
            run.paths,
            event_lines=event_lines,
            log_lines=log_lines,
        ))
        return EXIT_OK

    refresh = max(0.05, float(getattr(args, "refresh", 0.4) or 0.4))
    return run_live_show(
        run.paths,
        event_lines=event_lines,
        log_lines=log_lines,
        refresh_seconds=refresh,
    )


def _drop_mutation(
    run: _ResolvedRun,
    writer,
    *,
    wait_for_type: Optional[str] = None,
    match: Optional[Callable[[dict], bool]] = None,
) -> int:
    """Shared mutation plumbing: check run is alive, write, optionally wait.

    ``writer`` is a callable that does the actual file drop (atomic
    write, append, rename — whichever the intent demands). ``match`` is
    an optional predicate applied to the audit payload to confirm *our*
    intent landed (vs. someone else's concurrent ``ai send``).
    ``wait_for_type`` is the ``control-applied.jsonl`` event name to poll
    for; ``None`` means return as soon as the writer succeeds."""
    pid = run.meta.get("pid")
    status = run.meta.get("status")
    if status in ("killed", "crashed", "exited") or (
        isinstance(pid, int) and not pid_alive(pid)
    ):
        # Runner is gone: writing control files is pointless.
        print(
            f"error: run '{run.paths.run_id}' is no longer alive "
            f"(status={status!r}, pid={pid!r}).",
            file=sys.stderr,
        )
        return EXIT_RUN_GONE

    try:
        writer()
    except OSError as exc:
        print(f"error: writing control file failed: {exc}", file=sys.stderr)
        return EXIT_IO_ERROR

    if wait_for_type is None:
        return EXIT_OK

    deadline = time.time() + 30.0
    audit = run.paths.control_applied
    cursor = 0
    while time.time() < deadline:
        if not isinstance(pid, int) or not pid_alive(pid):
            # Runner died mid-wait — intent may still be picked up on
            # restart, but surface the state so the caller can decide.
            print(
                f"warning: runner exited before ack (run {run.paths.run_id}).",
                file=sys.stderr,
            )
            return EXIT_RUN_GONE
        if audit.exists():
            try:
                with audit.open("r", encoding="utf-8") as fh:
                    fh.seek(cursor)
                    for line in fh:
                        cursor += len(line.encode("utf-8"))
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if payload.get("event") != wait_for_type:
                            continue
                        if match is not None and not match(payload):
                            continue
                        return EXIT_OK
            except OSError:
                pass
        time.sleep(0.2)

    print(
        f"warning: timed out waiting for '{wait_for_type}' on run "
        f"{run.paths.run_id}.",
        file=sys.stderr,
    )
    return EXIT_RUN_GONE


def cmd_send(args: argparse.Namespace, runs_dir: Path) -> int:
    run = _resolve_run(runs_dir, args.run_id)
    text = args.text
    line = f"{now_iso()}\t{text}\n"

    def writer() -> None:
        # O_APPEND: each write is atomic for lines under PIPE_BUF, so
        # concurrent `ai send` calls survive together.
        fd = os.open(
            run.paths.control_file(CTL_GUIDANCE),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)

    match = (lambda p: p.get("text", "") == text) if args.wait else None
    wait_for = "guidance_received" if args.wait else None
    return _drop_mutation(run, writer, wait_for_type=wait_for, match=match)


def cmd_rewind(args: argparse.Namespace, runs_dir: Path) -> int:
    run = _resolve_run(runs_dir, args.run_id)
    try:
        intent = parse_rewind_to(args.to)
    except ValueError as exc:
        print(f"error: bad --to: {exc}", file=sys.stderr)
        return EXIT_USER_ERROR

    def writer() -> None:
        atomic_write_json(
            run.paths.control_file(CTL_REWIND),
            {
                "outer": intent.outer,
                "inner": intent.inner,
                "phase": intent.phase,
            },
        )

    def match(payload: dict) -> bool:
        to = payload.get("to") or {}
        return (
            to.get("outer") == intent.outer
            and to.get("inner") == intent.inner
            and to.get("phase") == intent.phase
        )

    wait_for = "rewind_applied" if args.wait else None
    return _drop_mutation(run, writer,
                          wait_for_type=wait_for,
                          match=match if args.wait else None)


def cmd_set_prompt(args: argparse.Namespace, runs_dir: Path) -> int:
    run = _resolve_run(runs_dir, args.run_id)
    try:
        text = load_text_arg(args.text, args.file, "prompt")
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_IO_ERROR
    if not text:
        print("error: empty prompt", file=sys.stderr)
        return EXIT_USER_ERROR
    writer = lambda: atomic_write_text(run.paths.control_file(CTL_PROMPT), text)
    wait_for = "prompt_updated" if args.wait else None
    return _drop_mutation(run, writer, wait_for_type=wait_for)


def cmd_pause(args: argparse.Namespace, runs_dir: Path) -> int:
    run = _resolve_run(runs_dir, args.run_id)
    try:
        run.paths.control_dir.mkdir(exist_ok=True)
        touch(run.paths.control_file(CTL_PAUSE))
    except OSError as exc:
        print(f"error: pause failed: {exc}", file=sys.stderr)
        return EXIT_IO_ERROR
    return EXIT_OK


def cmd_resume(args: argparse.Namespace, runs_dir: Path) -> int:
    run = _resolve_run(runs_dir, args.run_id)
    try:
        run.paths.control_file(CTL_PAUSE).unlink()
    except FileNotFoundError:
        # Not paused — treat as success, matches ``rm -f`` semantics.
        pass
    except OSError as exc:
        print(f"error: resume failed: {exc}", file=sys.stderr)
        return EXIT_IO_ERROR
    return EXIT_OK


# ── Worktree subcommands ────────────────────────────────────────────────────

def cmd_worktree(args: argparse.Namespace, runs_dir: Path) -> int:
    run = _resolve_run(runs_dir, args.run_id)
    from .worktree import load_worktree_info

    info = load_worktree_info(run.paths)
    if info is None:
        print(
            f"error: run '{run.paths.run_id}' has no worktree "
            "(was it started with --no-worktree, or in a non-git workspace?)",
            file=sys.stderr,
        )
        return EXIT_USER_ERROR
    print(info.path)
    return EXIT_OK


def cmd_diff(args: argparse.Namespace, runs_dir: Path) -> int:
    run = _resolve_run(runs_dir, args.run_id)
    from .worktree import (
        is_applied,
        load_worktree_info,
        make_diff_stat,
        make_full_patch,
        make_status_short,
    )

    info = load_worktree_info(run.paths)
    if info is None:
        print(
            f"error: run '{run.paths.run_id}' has no worktree.",
            file=sys.stderr,
        )
        return EXIT_USER_ERROR

    try:
        if args.full:
            sys.stdout.write(make_full_patch(info))
        elif args.stat:
            sys.stdout.write(make_diff_stat(info))
        else:
            # VS Code source-control parity: short status + stat header.
            short = make_status_short(info)
            stat = make_diff_stat(info)
            applied = is_applied(run.paths)
            print(f"# worktree:        {info.path}")
            print(f"# source workspace: {info.source_workspace}")
            print(f"# base commit:     {info.base_commit[:12]} "
                  f"({info.base_branch or 'detached'})")
            print(f"# applied to source: {'yes' if applied else 'no'}")
            print()
            if not short.strip() and "(no changes)" in stat:
                print("(no changes)")
                return EXIT_OK
            if short.strip():
                print("Changed files (git status --short):")
                sys.stdout.write(short)
                if not short.endswith("\n"):
                    sys.stdout.write("\n")
                print()
            print("Diff stat:")
            sys.stdout.write(stat)
        sys.stdout.flush()
        return EXIT_OK
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_IO_ERROR


def cmd_apply(args: argparse.Namespace, runs_dir: Path) -> int:
    run = _resolve_run(runs_dir, args.run_id)
    from .worktree import apply_to_source

    ok_, msg = apply_to_source(run.paths)
    if ok_:
        print(msg)
        return EXIT_OK
    print(f"error: {msg}", file=sys.stderr)
    return EXIT_IO_ERROR


def cmd_revert(args: argparse.Namespace, runs_dir: Path) -> int:
    run = _resolve_run(runs_dir, args.run_id)
    from .worktree import revert_from_source

    ok_, msg = revert_from_source(run.paths)
    if ok_:
        print(msg)
        return EXIT_OK
    print(f"error: {msg}", file=sys.stderr)
    return EXIT_IO_ERROR


def cmd_worktree_remove(args: argparse.Namespace, runs_dir: Path) -> int:
    run = _resolve_run(runs_dir, args.run_id)
    from .worktree import (
        applied_state_path,
        is_applied,
        load_worktree_info,
        remove_worktree,
        worktree_meta_path,
    )

    info = load_worktree_info(run.paths)
    if info is None:
        print(
            f"error: run '{run.paths.run_id}' has no worktree.",
            file=sys.stderr,
        )
        return EXIT_USER_ERROR

    # If the user has an outstanding apply, refuse by default — removing
    # the worktree at that point doesn't undo the source-workspace edits
    # but *does* drop ``applied.json`` cleanup paths. Force lets them
    # opt in anyway (e.g. after manually reverting). Either way, we
    # always preserve ``applied.json`` so a later ``ai revert`` can use
    # the recorded patch even though the worktree is gone.
    if is_applied(run.paths) and not getattr(args, "force", False):
        print(
            f"error: run '{run.paths.run_id}' has changes applied to the "
            "source workspace. Run `ai revert` first, or pass --force to "
            "remove the worktree anyway (the recorded patch will be "
            "preserved so `ai revert` keeps working).",
            file=sys.stderr,
        )
        return EXIT_USER_ERROR

    ok_, msg = remove_worktree(info, force=True)
    if not ok_:
        detail = f": {msg}" if msg else ""
        print(
            f"error: failed to remove worktree for run '{run.paths.run_id}'{detail}",
            file=sys.stderr,
        )
        print(
            "worktree metadata preserved so cleanup can be retried.",
            file=sys.stderr,
        )
        return EXIT_IO_ERROR

    # Drop only worktree.json after cleanup succeeds — keep applied.json
    # so a previously-applied patch can still be reverted.
    try:
        worktree_meta_path(run.paths).unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(
            f"error: worktree removed but could not delete metadata: {exc}",
            file=sys.stderr,
        )
        return EXIT_IO_ERROR

    msg_extra = ""
    if applied_state_path(run.paths).exists():
        msg_extra = (
            " (applied.json preserved — `ai revert` still available)"
        )
    print(f"worktree removed{msg_extra}")
    return EXIT_OK


# ── Entry ────────────────────────────────────────────────────────────────────


_COMMAND_MAP = {
    "run": cmd_run,
    "restart": cmd_restart,
    "kill": cmd_kill,
    "ls": cmd_ls,
    "show": cmd_show,
    "send": cmd_send,
    "rewind": cmd_rewind,
    "set-prompt": cmd_set_prompt,
    "pause": cmd_pause,
    "resume": cmd_resume,
    "worktree": cmd_worktree,
    "diff": cmd_diff,
    "apply": cmd_apply,
    "revert": cmd_revert,
    "worktree-remove": cmd_worktree_remove,
}


# Subcommands that target a single run and therefore support the
# interactive selector when ``run_id`` is omitted. ``run`` and ``ls``
# are intentionally excluded — they don't take a ``run_id`` argument at
# all.
SELECTOR_COMMANDS = frozenset({
    "restart", "kill", "show", "send", "rewind", "set-prompt",
    "pause", "resume", "worktree", "diff", "apply", "revert",
    "worktree-remove",
})

# Subset of selector commands whose effect is destructive enough that
# we prompt for confirmation when the run id was picked from the
# selector. Explicit ``run_id`` arguments — and a ``--yes`` flag —
# bypass this. The intent is "stop a tired operator from killing the
# wrong run because they pressed Enter on the wrong row".
DESTRUCTIVE_COMMANDS = frozenset({
    "kill", "restart", "apply", "revert", "worktree-remove",
})


def _normalize_send_args(args: argparse.Namespace) -> Optional[int]:
    """Disambiguate ``ai send``'s two optional positionals.

    ``run_id`` and ``text`` are both registered ``nargs="?"`` so that
    optional flags can appear between them (e.g. ``ai send RID --wait
    TEXT``). When the user passes only one positional, argparse fills
    ``run_id`` with it and leaves ``text=None``; we want the opposite —
    the single positional is the *guidance text* and ``run_id`` should
    be left empty so the selector can fill it in.

    Returns an exit code if the args are unrecoverable, ``None`` if
    normalization succeeded and dispatch should continue."""
    if getattr(args, "cmd", None) != "send":
        return None
    run_id = getattr(args, "run_id", None)
    text = getattr(args, "text", None)
    if text is None and run_id is not None:
        # Single positional: treat it as the guidance text.
        args.text = run_id
        args.run_id = None
        return None
    if text is None and run_id is None:
        print(
            "error: `ai send` requires guidance text "
            "(usage: ai send [RUN_ID] TEXT [--wait])",
            file=sys.stderr,
        )
        return EXIT_USER_ERROR
    return None


def _resolve_selector_run_id(
    args: argparse.Namespace, runs_dir: Path
) -> int:
    """If a selector-enabled command omits ``run_id``, fill it in.

    Return value is an exit code: ``EXIT_OK`` means "args.run_id is now
    populated, dispatch normally"; anything else means we already
    printed an error message and the CLI should return that code.

    Sets ``args._selected = True`` iff the id was chosen via the
    interactive selector. That flag drives the destructive-action
    confirmation in :func:`_maybe_confirm`."""
    args._selected = False
    if args.cmd not in SELECTOR_COMMANDS:
        return EXIT_OK
    if getattr(args, "run_id", None):
        return EXIT_OK

    from .selector import is_interactive, select_run

    if not is_interactive():
        print(
            f"error: '{args.cmd}' requires a run_id when stdin/stdout "
            "is not a TTY. Pass it explicitly (see `ai ls`) or run "
            "interactively to use the selector.",
            file=sys.stderr,
        )
        return EXIT_USER_ERROR

    rows = list_runs(runs_dir)
    if not rows:
        print("(no runs)", file=sys.stderr)
        return EXIT_USER_ERROR

    chosen = select_run(rows, prompt=f"Select run for `ai {args.cmd}`")
    if chosen is None:
        print("cancelled", file=sys.stderr)
        return EXIT_USER_ERROR
    args.run_id = chosen.run_id
    args._selected = True
    return EXIT_OK


def _maybe_confirm(args: argparse.Namespace) -> bool:
    """Prompt before running a destructive command picked from the selector.

    Skipped when:

    * the command isn't destructive,
    * the run id was passed explicitly (operator already named the
      target),
    * the operator passed ``--yes`` / ``-y``,
    * stdin isn't a TTY (scripted callers must not block on input).

    Returns ``True`` iff the command should proceed."""
    if args.cmd not in DESTRUCTIVE_COMMANDS:
        return True
    if not getattr(args, "_selected", False):
        return True
    if getattr(args, "yes", False):
        return True
    try:
        if not sys.stdin.isatty():
            return True
    except (AttributeError, OSError):
        return True
    try:
        ans = input(
            f"Confirm `ai {args.cmd} {args.run_id}`? [y/N] "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("cancelled", file=sys.stderr)
        return False
    if ans in ("y", "yes"):
        return True
    print("cancelled", file=sys.stderr)
    return False


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    runs_dir = resolve_runs_dir(args.runs_dir)
    try:
        rc = _normalize_send_args(args)
        if rc is not None:
            return rc
        rc = _resolve_selector_run_id(args, runs_dir)
        if rc != EXIT_OK:
            return rc
        if not _maybe_confirm(args):
            return EXIT_USER_ERROR
        return _COMMAND_MAP[args.cmd](args, runs_dir)
    except KeyboardInterrupt:
        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
