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
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from . import actions
from .backends import BACKENDS
from .control import parse_rewind_to
from .feature.config import RunConfig
from .ls import list_runs
from .meta import read_meta
from .run_dir import (
    RunPaths,
    create_run_dir,
    new_run_id,
    pid_alive,
    read_json,
    resolve_runs_dir,
)
from .runner import bootstrap_run


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


def _validate_backend(name: str, *, label: str = "backend") -> str:
    """Reject an unknown backend name with a clean error.

    Used both for the global ``--backend`` and the per-phase
    ``--{impl,fix,reviewer}-backend`` overrides so error messages
    consistently name the offending flag (``label``)."""
    if name not in BACKENDS:
        valid = ", ".join(sorted(BACKENDS))
        raise ValueError(f"unknown {label} '{name}'. valid: {valid}")
    return name


def _resolve_phase(
    *,
    phase: str,
    args: argparse.Namespace,
    global_backend: str,
) -> tuple[str | None, str | None]:
    """Resolve per-phase ``(backend_override, agent_cmd_override)``.

    A phase pinned to a different backend defaults its CLI binary to
    that backend's ``default_cmd`` so spec.json is a complete snapshot
    — operators don't have to remember to also pass
    ``--reviewer-cmd codex`` when they pass ``--reviewer-backend codex``.

    Resolution order matches the global ``--backend`` flag: explicit
    CLI flag → matching env var (``AGENT_{IMPL,FIX,REVIEWER}_BACKEND``
    / ``..._CMD``) → ``None``. Reading env vars here keeps
    ``_make_cfg_from_args`` and :func:`auto_iterator.actions.default_run_config`
    (used by the TUI's ``n`` verb) in lockstep, so an operator who
    exports ``AGENT_REVIEWER_BACKEND=codex`` gets the mixed Claude/Codex
    setup from either entry point.

    Returns ``(None, None)`` for phases left at the global backend so
    the persisted config keeps the legacy single-backend shape and
    older spec readers stay happy."""
    phase_backend = getattr(args, f"{phase}_backend", None) or os.environ.get(
        f"AGENT_{phase.upper()}_BACKEND"
    ) or None
    phase_cmd = getattr(args, f"{phase}_agent_cmd", None) or os.environ.get(
        f"AGENT_{phase.upper()}_CMD"
    ) or None

    if phase_backend is None and phase_cmd is None:
        return None, None

    if phase_backend is not None:
        _validate_backend(phase_backend, label=f"--{phase}-backend")

    # An explicit cmd without a backend is allowed (operator wants the
    # same backend with a different binary path); the global backend
    # then governs the stream-json adapter.
    resolved_backend = phase_backend or global_backend

    # Normalize: a phase pinned to the same backend as the global one
    # is indistinguishable from "no override" — keep the cfg in single-
    # backend shape so legacy spec readers stay happy.
    out_backend = (
        resolved_backend if resolved_backend != global_backend else None
    )

    # Only fall back to the per-phase backend's ``default_cmd`` when the
    # phase actually diverges from the global backend. Otherwise an
    # explicit-but-redundant ``--reviewer-backend claude-code`` would
    # silently bypass a custom global ``--agent-cmd /tmp/custom-claude``
    # by overwriting the inherited cmd with the backend default.
    if phase_cmd is not None:
        resolved_cmd = phase_cmd
    elif out_backend is not None:
        resolved_cmd = BACKENDS[out_backend].default_cmd
    else:
        resolved_cmd = None

    return out_backend, resolved_cmd


def _phase_default_model(
    *,
    phase: str,
    explicit: str | None,
    phase_backend: str | None,
    global_backend_obj,
) -> str:
    """Pick the right model default for a phase.

    When ``--impl-backend claude-code`` is set without ``--impl-model``,
    the implementation model should default to Claude Code's
    ``default_impl_model`` — not the global backend's. This mirrors the
    pre-mixed-backend behaviour where the resolved backend's model
    fingerprints landed in ``spec.json`` automatically."""
    if explicit:
        return explicit
    be = BACKENDS[phase_backend] if phase_backend else global_backend_obj
    attr = f"default_{phase}_model"
    return getattr(be, attr)


def _make_cfg_from_args(args: argparse.Namespace) -> RunConfig:
    """Translate the ``ai run`` namespace into a typed :class:`RunConfig`."""
    backend = _validate_backend(
        args.backend or os.environ.get("AGENT_BACKEND", "cursor")
    )
    be = BACKENDS[backend]

    impl_backend, impl_cmd = _resolve_phase(
        phase="impl", args=args, global_backend=backend,
    )
    fix_backend, fix_cmd = _resolve_phase(
        phase="fix", args=args, global_backend=backend,
    )
    reviewer_backend, reviewer_cmd = _resolve_phase(
        phase="reviewer", args=args, global_backend=backend,
    )

    cfg = RunConfig(
        task=load_text_arg(args.prompt, args.prompt_file, "prompt"),
        impl_model=_phase_default_model(
            phase="impl",
            explicit=args.impl_model,
            phase_backend=impl_backend,
            global_backend_obj=be,
        ),
        fix_model=_phase_default_model(
            phase="fix",
            explicit=args.fix_model,
            phase_backend=fix_backend,
            global_backend_obj=be,
        ),
        reviewer_model=_phase_default_model(
            phase="reviewer",
            explicit=args.reviewer_model,
            phase_backend=reviewer_backend,
            global_backend_obj=be,
        ),
        max_outer=args.max_outer,
        max_inner=args.max_inner,
        workspace=str(Path(args.workspace).expanduser().resolve()),
        skip_impl=args.skip_impl,
        extra_flags=tuple(args.extra_flags or []),
        agent_cmd=args.agent_cmd or os.environ.get("AGENT_CMD", be.default_cmd),
        backend=backend,
        use_worktree=not getattr(args, "no_worktree", False),
        impl_backend=impl_backend,
        fix_backend=fix_backend,
        reviewer_backend=reviewer_backend,
        impl_agent_cmd=impl_cmd,
        fix_agent_cmd=fix_cmd,
        reviewer_agent_cmd=reviewer_cmd,
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
    # ``required=False`` so plain ``ai`` (no subcommand) is a legal
    # invocation that opens the interactive run-list TUI. The
    # dispatcher treats ``args.cmd is None`` as "open the TUI". Every
    # existing subcommand keeps working unchanged.
    sub = p.add_subparsers(dest="cmd", required=False)

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
    # Per-phase backend overrides — let an operator mix CLIs across the
    # loop. The canonical use case is "Claude Code as implementer/fixer
    # with Codex as a fresh-eyes reviewer": pass ``--reviewer-backend
    # codex`` and the reviewer runs through Codex while impl/fix stay
    # on the global ``--backend`` (e.g. ``--backend claude-code``). When
    # a per-phase backend is set without a matching ``--…-cmd``, the
    # binary defaults to that backend's ``default_cmd``.
    run_p.add_argument("--impl-backend", default=None,
                       help="Pin the implementation phase to a specific backend "
                            "(cursor/claude-code/codex). Falls back to --backend.")
    run_p.add_argument("--fix-backend", default=None,
                       help="Pin the fix phase to a specific backend. "
                            "Falls back to --backend.")
    run_p.add_argument("--reviewer-backend", default=None,
                       help="Pin the reviewer phase to a specific backend. "
                            "Falls back to --backend.")
    run_p.add_argument("--impl-cmd", default=None, dest="impl_agent_cmd",
                       help="Override the CLI binary for the implementation "
                            "phase. Defaults to the impl backend's default_cmd.")
    run_p.add_argument("--fix-cmd", default=None, dest="fix_agent_cmd",
                       help="Override the CLI binary for the fix phase. "
                            "Defaults to the fix backend's default_cmd.")
    run_p.add_argument("--reviewer-cmd", default=None, dest="reviewer_agent_cmd",
                       help="Override the CLI binary for the reviewer phase. "
                            "Defaults to the reviewer backend's default_cmd.")
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
    show_p.add_argument("--stream", action="store_true",
                        help="Force the SSH-friendly streaming tail. "
                             "This is the default for an interactive TTY "
                             "now; the flag is kept so muscle memory and "
                             "scripts still work and so it can be used "
                             "alongside other dispatch flags. The local "
                             "terminal's native scrollback handles "
                             "navigation, so mouse-wheel / PageUp / "
                             "tmux copy-mode all work at zero round-trip. "
                             "Esc / q / Ctrl-C to exit.")
    show_p.add_argument("--tui", action="store_true",
                        help="Opt back into the in-process pyratatui "
                             "detail screen. Useful for local terminals "
                             "where round-trip latency is negligible; "
                             "over high-latency SSH the default streaming "
                             "mode is much smoother because scrolling is "
                             "handled client-side.")
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

    if args.foreground:
        # Foreground path: same process runs the loop; stdout stays
        # attached. Bootstrapping is in-line because the foreground
        # caller owns the run-dir creation too — no detached fork to
        # delegate to.
        run_id = new_run_id()
        paths = create_run_dir(runs_dir, run_id)
        bootstrap_run(paths, cfg, pid=os.getpid(), agent_type=args.agent_type)
        print(f"run_id: {run_id}", file=sys.stderr)
        from .runner import run_review_loop_sync
        return run_review_loop_sync(cfg, paths, agent_type=args.agent_type)

    # Detached path: hand off to the shared spawn primitive in
    # ``actions``. Both ``ai run`` and the TUI's ``n`` keybinding land
    # at the same Popen site — same env, same ``start_new_session=True``,
    # same ``stdout=agent.log``. Keep the printed run_id on stdout so
    # scripts that ``RUN_ID=$(ai run ...)`` keep working.
    result = actions.spawn_runner_detached(
        runs_dir, cfg, agent_type=args.agent_type,
    )
    if not result.ok:
        print(f"error: {result.message}", file=sys.stderr)
        return EXIT_IO_ERROR
    print(result.run_id)
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

    from .runner import spec_to_cfg
    try:
        cfg = spec_to_cfg(spec)
    except KeyError as exc:
        print(f"error: spec.json missing field {exc}", file=sys.stderr)
        return EXIT_IO_ERROR

    agent_type = spec.get("agent_type", "review-loop")
    result = actions.spawn_runner_detached(
        runs_dir, cfg,
        agent_type=agent_type,
        restarted_from=args.run_id,
    )
    if not result.ok:
        print(f"error: {result.message}", file=sys.stderr)
        return EXIT_IO_ERROR
    print(result.run_id)
    return EXIT_OK


def cmd_kill(args: argparse.Namespace, runs_dir: Path) -> int:
    run = _resolve_run(runs_dir, args.run_id)
    if _signal_runner(run, grace=args.grace, force=args.force):
        return EXIT_OK
    return EXIT_RUN_GONE


def _signal_runner(run: _ResolvedRun, *, grace: float, force: bool) -> bool:
    """Thin CLI-side wrapper around :func:`actions.signal_runner`.

    Kept as a separate function so existing tests that patch
    ``auto_iterator.cli.pid_alive`` keep working — the actions module
    consults ``pid_alive`` through its own import, which the same
    monkeypatch reaches via the underlying ``run_dir.pid_alive``."""
    return actions.signal_runner(
        run.paths, run.meta, grace=grace, force=force,
    )


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

    Default flow (in priority order):

    * ``--json`` → one-shot raw ``state.json`` (scriptable). Never
      imports the TUI library.
    * ``--once`` / ``--logs`` → one-shot combined text view. Never
      imports the TUI library.
    * ``--tui`` → in-process pyratatui detail screen (escape hatch
      for local terminals where round-trip latency is negligible).
    * Non-TTY stdout (without ``--stream``) → same as ``--once``.
      Never imports the TUI library.
    * Interactive TTY default (or explicit ``--stream``) → header +
      tail-and-follow on the regular screen buffer so the local
      terminal's native scrollback owns navigation. This is the
      default because over even modest network latency the
      pyratatui frame loop turns scrolling into a server-side
      round-trip per keystroke; native scrollback dodges that
      entirely. Never imports the TUI library.

    The TTY path lazy-imports :mod:`auto_iterator.tui` only when
    the operator explicitly asks for it via ``--tui``, so plain
    ``ai ls`` / ``ai show <id>`` / ``ai show --json`` / ``--once`` /
    ``--stream`` invocations don't pay the pyratatui native-binding
    startup cost.
    """
    run = _resolve_run(runs_dir, args.run_id)

    if args.json:
        # Scriptable contract: byte-identical to ``state_json_text``.
        from .display import state_json_text
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
    if once:
        # Byte-identical to today's one-shot output. No pyratatui.
        # ``--once`` beats every other rendering mode: if the operator
        # asked for a single snapshot we honour that even when
        # ``--stream`` / ``--tui`` are also set.
        from .display import render_combined_view
        sys.stdout.write(render_combined_view(
            run.paths,
            event_lines=event_lines,
            log_lines=log_lines,
        ))
        return EXIT_OK

    if getattr(args, "tui", False):
        # Explicit pyratatui escape hatch. Useful on a *local*
        # terminal where round-trips don't hurt; on a high-latency
        # link the operator should prefer the streaming default.
        if not _stdout_is_tty():
            print(
                "error: --tui requires an interactive terminal; "
                "stdout is not a TTY.",
                file=sys.stderr,
            )
            return EXIT_USER_ERROR
        refresh = max(0.05, float(getattr(args, "refresh", 0.4) or 0.4))
        from .tui import run_detail_app
        return run_detail_app(
            run.paths,
            refresh_seconds=refresh,
            initial_log_lines=log_lines,
        )

    if not _stdout_is_tty() and not getattr(args, "stream", False):
        # Same one-shot bytes as ``--once``; the non-TTY heuristic
        # exists so ``ai show <id> | grep ...`` Just Works. An
        # explicit ``--stream`` on piped stdout still follows
        # (``tail -f`` semantics).
        from .display import render_combined_view
        sys.stdout.write(render_combined_view(
            run.paths,
            event_lines=event_lines,
            log_lines=log_lines,
        ))
        return EXIT_OK

    # TTY default (and explicit ``--stream``) → native-scrollback-
    # friendly tail. Deliberately bypasses the pyratatui TUI even in
    # a TTY: the whole point is that scroll input is served by the
    # local terminal emulator, not by pyratatui's frame loop on the
    # remote host. Eliminates the per-keystroke round-trip that makes
    # the in-process TUI feel laggy over SSH.
    from .display import stream_log
    refresh = max(0.05, float(getattr(args, "refresh", 0.4) or 0.4))
    return stream_log(
        run.paths,
        log_lines=log_lines,
        poll_seconds=refresh,
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
    writer = lambda: actions.write_guidance(run.paths, text)
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
        actions.write_rewind(
            run.paths,
            outer=intent.outer,
            inner=intent.inner,
            phase=intent.phase,
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
    writer = lambda: actions.write_prompt(run.paths, text)
    wait_for = "prompt_updated" if args.wait else None
    return _drop_mutation(run, writer, wait_for_type=wait_for)


def cmd_pause(args: argparse.Namespace, runs_dir: Path) -> int:
    run = _resolve_run(runs_dir, args.run_id)
    try:
        actions.write_pause(run.paths)
    except OSError as exc:
        print(f"error: pause failed: {exc}", file=sys.stderr)
        return EXIT_IO_ERROR
    return EXIT_OK


def cmd_resume(args: argparse.Namespace, runs_dir: Path) -> int:
    run = _resolve_run(runs_dir, args.run_id)
    try:
        actions.clear_pause(run.paths)
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


def cmd_tui(_args: argparse.Namespace, runs_dir: Path) -> int:
    """Open the interactive pyratatui run-list TUI.

    Routed to from the bare ``ai`` invocation (no subcommand). The
    TUI is lazy-imported so ``ai ls`` / ``ai show --json`` etc. don't
    pay the pyratatui native-binding startup cost. Returning the
    app's exit code makes Ctrl-C from inside the TUI propagate
    cleanly through the shell.

    On Enter-on-row, the run-list TUI exits and we hand off to
    :func:`auto_iterator.display.stream_log` for the selected run.
    The reason for the handoff (instead of pushing an in-process
    detail screen) is the high-latency-SSH lag story: streaming on
    the regular screen buffer means the local terminal's native
    scrollback owns navigation, so PageUp / mouse-wheel / tmux
    copy-mode all work at zero round-trip. Pushing a pyratatui
    ``RunDetailScreen`` would route every scroll keystroke back to
    the remote host and re-introduce exactly the lag this change
    was made to fix."""
    if not _stdout_is_tty():
        print(
            "error: `ai` (no subcommand) opens an interactive TUI; "
            "stdout is not a TTY. Try `ai ls` instead.",
            file=sys.stderr,
        )
        return EXIT_USER_ERROR
    from .tui import run_list_app_with_selection
    rc, selection = run_list_app_with_selection(runs_dir)
    if selection is None:
        return rc
    # Operator picked a run from the list. Drop into the streaming
    # tail on the regular screen buffer with ``log_lines=None`` so
    # the *entire* transcript is dumped into the local terminal's
    # scrollback. Anything less truncates older history that's no
    # longer reachable once the alt-screen TUI tears down — the
    # whole point of native-scrollback streaming is that the local
    # terminal owns navigation, and that's only useful if the bytes
    # you want to scroll back to were actually written there. A
    # bounded seed (we used to pass ``log_lines=200``) silently
    # hides anything older than the cap, which matches the
    # operator's "I cannot see the full log" complaint exactly.
    from .display import stream_log
    return stream_log(
        selection,
        log_lines=None,
        poll_seconds=0.4,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    runs_dir = resolve_runs_dir(args.runs_dir)
    try:
        if args.cmd is None:
            # Bare ``ai`` (no subcommand) → run-list TUI.
            return cmd_tui(args, runs_dir)
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
