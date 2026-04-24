"""``ai`` — stateless operator CLI for auto-iterator runs.

Every subcommand is a one-shot process: it opens files in
``<runs-dir>/<run_id>/``, reads or writes, and exits. There is no
long-lived supervisor and no IPC channel to anything; the filesystem is
the protocol.

Subcommand families
-------------------
* Spawn / lifecycle — ``run``, ``restart``, ``kill``.
* Read — ``ls``, ``show``, ``tail``.
* Mutate — ``send``, ``rewind``, ``set-prompt``, ``set-context``,
  ``pause``, ``resume``.

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
from .events import TERMINAL_EVENT_TYPES, iter_events_from_seq, tail_events
from .feature.config import RunConfig
from .ls import list_runs, summarize_run
from .meta import read_meta, update_meta
from .run_dir import (
    CTL_CONTEXT,
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
        context=load_text_arg(args.context, args.context_file, "context"),
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
    ctx_g = run_p.add_mutually_exclusive_group()
    ctx_g.add_argument("--context", default="", help="Additional static context.")
    ctx_g.add_argument("--context-file", help="Path to a UTF-8 context file.")
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

    # ── restart ──
    restart_p = sub.add_parser("restart",
                               help="Kill a run and respawn from its spec.json.")
    restart_p.add_argument("run_id")
    restart_p.add_argument("--grace", type=float, default=5.0,
                           help="Seconds to wait after SIGTERM before SIGKILL.")

    # ── kill ──
    kill_p = sub.add_parser("kill", help="Signal a running runner and wait.")
    kill_p.add_argument("run_id")
    kill_p.add_argument("--grace", type=float, default=5.0)
    kill_p.add_argument("--force", action="store_true",
                        help="Skip SIGTERM and send SIGKILL immediately.")

    # ── ls ──
    ls_p = sub.add_parser("ls", help="List runs; defaults to the current workspace.")
    ls_p.add_argument("--all", action="store_true",
                      help="Include runs from other workspaces.")
    ls_p.add_argument("--json", action="store_true",
                      help="Emit one JSON object per run on stdout.")

    # ── show ──
    show_p = sub.add_parser("show", help="Print a run's state.json snapshot.")
    show_p.add_argument("run_id")
    show_p.add_argument("--json", action="store_true")

    # ── tail ──
    tail_p = sub.add_parser("tail", help="Stream events.jsonl (optionally follow).")
    tail_p.add_argument("run_id")
    tail_p.add_argument("--lines", type=int, default=200,
                        help="Emit the last N events before following (default 200).")
    tail_p.add_argument("--follow", action="store_true")
    tail_p.add_argument("--from-seq", type=int, default=None,
                        help="Start from events with seq > K (ignores --lines).")
    tail_p.add_argument("--type", action="append", default=[], dest="types",
                        help="Filter to these event types (repeatable).")

    # ── send ──
    send_p = sub.add_parser("send", help="Queue operator guidance for the next review.")
    send_p.add_argument("run_id")
    send_p.add_argument("text")
    send_p.add_argument("--wait", action="store_true",
                        help="Block until the runner records guidance_received.")

    # ── rewind ──
    rewind_p = sub.add_parser("rewind",
                              help="Jump back to (outer, inner, phase).")
    rewind_p.add_argument("run_id")
    rewind_p.add_argument("--to", required=True,
                          help="outer=N,inner=M[,phase=review|fix|after_impl]")
    rewind_p.add_argument("--wait", action="store_true")

    # ── set-prompt ──
    sp_p = sub.add_parser("set-prompt",
                          help="Replace the review-loop's task prompt.")
    sp_p.add_argument("run_id")
    sp_g = sp_p.add_mutually_exclusive_group(required=True)
    sp_g.add_argument("--text")
    sp_g.add_argument("--prompt-file", dest="file")
    sp_p.add_argument("--wait", action="store_true")

    # ── set-context ──
    sc_p = sub.add_parser("set-context",
                          help="Replace the review-loop's static context.")
    sc_p.add_argument("run_id")
    sc_g = sc_p.add_mutually_exclusive_group(required=True)
    sc_g.add_argument("--text")
    sc_g.add_argument("--context-file", dest="file")
    sc_p.add_argument("--wait", action="store_true")

    # ── pause / resume ──
    pause_p = sub.add_parser("pause", help="Stall the runner at the next boundary.")
    pause_p.add_argument("run_id")
    resume_p = sub.add_parser("resume", help="Let a paused runner continue.")
    resume_p.add_argument("run_id")

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
    workspace = None if args.all else str(Path.cwd().resolve())
    rows = list_runs(runs_dir, workspace=workspace)
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


def cmd_show(args: argparse.Namespace, runs_dir: Path) -> int:
    run = _resolve_run(runs_dir, args.run_id)
    try:
        state_text = run.paths.state.read_text(encoding="utf-8")
    except FileNotFoundError:
        # State not yet written — fall back to meta for something useful.
        state_text = json.dumps(run.meta, indent=2)
    # ``ai show`` always prints a JSON object; ``--json`` is accepted for
    # parity with ``ls`` but doesn't change the payload.
    if args.json:
        try:
            obj = json.loads(state_text)
        except json.JSONDecodeError:
            obj = {"raw": state_text}
        print(json.dumps(obj, indent=2))
    else:
        sys.stdout.write(state_text)
        if not state_text.endswith("\n"):
            sys.stdout.write("\n")
    return EXIT_OK


def cmd_tail(args: argparse.Namespace, runs_dir: Path) -> int:
    run = _resolve_run(runs_dir, args.run_id)
    types = set(args.types) if args.types else None

    def _process(evt: dict) -> bool:
        """Emit the event (respecting ``--type``); return True iff terminal.

        Terminal check runs *before* the type filter so a ``--type`` that
        excludes ``run_finished`` still short-circuits ``--follow`` rather
        than spinning forever on a finished run."""
        terminal = evt.get("type") in TERMINAL_EVENT_TYPES
        if not types or evt.get("type") in types:
            _emit_event(evt)
        return terminal

    last_seq = 0
    saw_terminal = False
    if args.from_seq is not None:
        last_seq = args.from_seq
        for evt in iter_events_from_seq(run.paths.events, last_seq):
            saw_terminal |= _process(evt)
            last_seq = max(last_seq, int(evt.get("seq", last_seq)))
    else:
        initial = tail_events(run.paths.events, n=max(1, args.lines))
        for evt in initial:
            saw_terminal |= _process(evt)
        last_seq = int(initial[-1]["seq"]) if initial and "seq" in initial[-1] else 0

    if not args.follow or saw_terminal:
        return EXIT_OK

    # Polling follow: simple, lockless, never contends with the writer.
    # inotify would be nicer but adds a dependency for a small optimisation.
    try:
        while True:
            any_new = False
            for evt in iter_events_from_seq(run.paths.events, last_seq):
                any_new = True
                last_seq = max(last_seq, int(evt.get("seq", last_seq)))
                if _process(evt):
                    return EXIT_OK
            if not any_new:
                time.sleep(0.25)
    except KeyboardInterrupt:
        return EXIT_OK


def _emit_event(evt: dict) -> None:
    """One JSON object per line — friendly to ``jq`` and shell loops."""
    sys.stdout.write(json.dumps(evt) + "\n")
    sys.stdout.flush()


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


def cmd_set_context(args: argparse.Namespace, runs_dir: Path) -> int:
    run = _resolve_run(runs_dir, args.run_id)
    try:
        text = load_text_arg(args.text, args.file, "context")
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_IO_ERROR
    writer = lambda: atomic_write_text(run.paths.control_file(CTL_CONTEXT), text)
    wait_for = "context_updated" if args.wait else None
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


# ── Entry ────────────────────────────────────────────────────────────────────


_COMMAND_MAP = {
    "run": cmd_run,
    "restart": cmd_restart,
    "kill": cmd_kill,
    "ls": cmd_ls,
    "show": cmd_show,
    "tail": cmd_tail,
    "send": cmd_send,
    "rewind": cmd_rewind,
    "set-prompt": cmd_set_prompt,
    "set-context": cmd_set_context,
    "pause": cmd_pause,
    "resume": cmd_resume,
}


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    runs_dir = resolve_runs_dir(args.runs_dir)
    try:
        return _COMMAND_MAP[args.cmd](args, runs_dir)
    except KeyboardInterrupt:
        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
