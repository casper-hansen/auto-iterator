"""Operator-action primitives — the file-drop side of the protocol.

The ``ai`` CLI (``cli.py``) and the Textual UI (``tui.py``) both express
operator intents the same way: drop the appropriate file under
``<run_dir>/control/`` (or spawn a detached runner via ``Popen``). To
keep the two front-ends honest about the protocol — and to keep the
TUI from re-implementing it — the writer functions live here, in one
place, and are imported by both.

Every function in this module is intentionally low-level: it knows the
exact shape of the file it writes (``guidance.txt`` is timestamp-tab-
text-newline; ``rewind.json`` is ``{outer, inner, phase}``; ``pause``
is an empty marker file), but it does *not* know about argparse
namespaces, exit codes, or terminal formatting. The callers handle
those.

Two design rules to preserve:

* **Filesystem is still the protocol.** No socket, no shared memory,
  no daemon — every primitive here is a tiny POSIX call against the
  run directory. Tests assert this by mocking the surrounding Popen /
  os.kill sites and verifying no other channel was used.
* **Runners are detached.** :func:`spawn_runner_detached` is the only
  place that calls ``subprocess.Popen([..., "-m",
  "auto_iterator.runner", ...], start_new_session=True)``. Any caller
  that wants to spawn a runner must end up here so behaviour stays
  uniform across CLI and TUI.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .backends import BACKENDS
from .control import parse_rewind_to
from .feature.config import RunConfig
from .meta import read_meta, update_meta
from .run_dir import (
    CTL_GUIDANCE,
    CTL_PAUSE,
    CTL_PROMPT,
    CTL_REWIND,
    RunPaths,
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    create_run_dir,
    new_run_id,
    now_iso,
    pid_alive,
    touch,
)
from .runner import cfg_to_spec


# ── Result type ─────────────────────────────────────────────────────────────


@dataclass
class ActionResult:
    """Outcome of an action that may fail with a user-visible message.

    ``ok`` is the only value callers should branch on. ``message`` is a
    short human-readable string that the CLI prints to stderr on
    failure and the TUI surfaces in a notification toast. ``run_id``
    is set by the spawn primitives so the caller can echo it back."""

    ok: bool
    message: str = ""
    run_id: Optional[str] = None


# ── Liveness check shared by mutators ───────────────────────────────────────


def runner_is_alive(meta: dict) -> bool:
    """Return True iff *meta* describes a runner this process can signal.

    Mirrors the gate used by :func:`auto_iterator.cli._drop_mutation`:
    a meta status of ``killed`` / ``crashed`` / ``exited`` short-circuits
    even if the pid happens to be reused by something else, and an
    integer pid that no longer satisfies :func:`pid_alive` is also
    treated as gone."""
    status = meta.get("status")
    if status in ("killed", "crashed", "exited"):
        return False
    pid = meta.get("pid")
    if isinstance(pid, int) and not pid_alive(pid):
        return False
    return True


# ── Control-file writers ────────────────────────────────────────────────────
#
# Each writer takes a :class:`RunPaths` plus the typed payload and lays
# down exactly one file in the canonical location. They never read from
# the filesystem (no liveness checks, no audit polling) — the caller is
# responsible for sequencing those steps. Returning ``None`` keeps the
# functions composable inside a single ``try`` block.


def write_guidance(paths: RunPaths, text: str) -> None:
    """Append one operator-guidance line to ``control/guidance.txt``.

    Format is ``<ISO8601>\\t<text>\\n`` so the runner can split on tab
    and recover the send-time. ``O_APPEND`` is atomic for lines below
    ``PIPE_BUF``, so two concurrent ``ai send`` calls (CLI + TUI, or two
    operators) never interleave bytes — at worst they reorder lines."""
    line = f"{now_iso()}\t{text}\n"
    paths.control_dir.mkdir(exist_ok=True)
    fd = os.open(
        paths.control_file(CTL_GUIDANCE),
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def write_rewind(paths: RunPaths, *, outer: int, inner: int, phase: str) -> None:
    """Drop a ``control/rewind.json`` payload for the next inner-boundary drain.

    Field shape matches ``control._validate_rewind`` exactly: an object
    with integer ``outer`` / ``inner`` and a ``phase`` of ``review`` /
    ``fix`` / ``after_impl``. The runner re-validates on read so a
    malformed payload is rejected cleanly rather than blowing up the
    drain step."""
    paths.control_dir.mkdir(exist_ok=True)
    atomic_write_json(
        paths.control_file(CTL_REWIND),
        {"outer": outer, "inner": inner, "phase": phase},
    )


def write_rewind_from_to_string(paths: RunPaths, to: str) -> None:
    """Convenience for callers that already have the ``--to=`` shorthand."""
    intent = parse_rewind_to(to)
    write_rewind(paths, outer=intent.outer, inner=intent.inner, phase=intent.phase)


def write_prompt(paths: RunPaths, text: str) -> None:
    """Replace ``control/prompt.txt`` with *text* (atomic rename)."""
    paths.control_dir.mkdir(exist_ok=True)
    atomic_write_text(paths.control_file(CTL_PROMPT), text)


def write_pause(paths: RunPaths) -> None:
    """Create ``control/pause`` so the runner stalls at its next boundary."""
    paths.control_dir.mkdir(exist_ok=True)
    touch(paths.control_file(CTL_PAUSE))


def clear_pause(paths: RunPaths) -> None:
    """Remove ``control/pause``. Missing file is not an error (rm -f)."""
    try:
        paths.control_file(CTL_PAUSE).unlink()
    except FileNotFoundError:
        pass


# ── Signal-the-runner primitive ─────────────────────────────────────────────


def signal_runner(
    paths: RunPaths,
    meta: dict,
    *,
    grace: float = 5.0,
    force: bool = False,
) -> bool:
    """SIGTERM → wait → SIGKILL the run's pid. Returns True iff a live pid was hit.

    Lifted verbatim from ``cli._signal_runner`` so the CLI ``kill``
    command and the TUI's ``k`` keybinding take exactly the same code
    path. Updates ``meta.json`` to ``killed`` on success so subsequent
    list views show the right status without waiting for the runner's
    own teardown."""
    pid = meta.get("pid")
    if not isinstance(pid, int) or not pid_alive(pid):
        update_meta(paths, status=meta.get("status", "crashed"))
        return False

    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        return False

    deadline = time.time() + max(0.0, grace)
    while time.time() < deadline and pid_alive(pid):
        time.sleep(0.1)

    if pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        t0 = time.time()
        while time.time() - t0 < 2.0 and pid_alive(pid):
            time.sleep(0.05)

    update_meta(paths, status="killed", finished_at=now_iso())
    return True


# ── Detached runner spawn ───────────────────────────────────────────────────


def spawn_runner_detached(
    runs_dir: Path,
    cfg: RunConfig,
    *,
    agent_type: str = "review-loop",
    restarted_from: Optional[str] = None,
) -> ActionResult:
    """Spawn ``python -m auto_iterator.runner <run_dir>`` as a detached child.

    Single source of truth for the detached spawn site. Both
    ``cli.cmd_run`` and ``cli.cmd_restart`` (and the TUI's "new run" /
    "restart" verbs) end up here so the spawn behaviour is uniform —
    same env, same ``start_new_session=True``, same ``stdin=DEVNULL``,
    same ``stdout=stderr=agent.log``. If a future change wants to
    e.g. add a wrapper script or change the cwd, this is the one place
    to do it."""
    run_id = new_run_id()
    paths = create_run_dir(runs_dir, run_id)
    atomic_write_json(paths.spec, cfg_to_spec(cfg, agent_type=agent_type))

    meta_fields = dict(
        run_id=run_id,
        pid=0,
        status="running",
        started_at=now_iso(),
        finished_at=None,
        workspace=cfg.workspace,
        agent_type=agent_type,
        heartbeat_at=now_iso(),
    )
    if restarted_from is not None:
        meta_fields["restarted_from"] = restarted_from
    update_meta(paths, **meta_fields)

    index_payload = {
        "event": "run_started",
        "timestamp": now_iso(),
        "run_id": run_id,
        "workspace": cfg.workspace,
        "agent_type": agent_type,
    }
    if restarted_from is not None:
        index_payload["restarted_from"] = restarted_from
    append_jsonl(paths.index, index_payload)

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
        log_fh.close()
        return ActionResult(
            ok=False,
            message=f"failed to spawn runner: {exc}",
            run_id=run_id,
        )
    finally:
        if not log_fh.closed:
            log_fh.close()

    update_meta(paths, pid=proc.pid)
    return ActionResult(ok=True, run_id=run_id)


# ── Default-config builder for env-driven callers ──────────────────────────


def default_run_config(
    *,
    task: str,
    workspace: str,
    backend: Optional[str] = None,
    agent_cmd: Optional[str] = None,
    max_outer: int = 10,
    max_inner: int = 10,
    skip_impl: bool = False,
    extra_flags: tuple[str, ...] = (),
    use_worktree: bool = True,
) -> RunConfig:
    """Build a :class:`RunConfig` the way ``ai run`` would build it.

    The CLI's ``_make_cfg_from_args`` reads ``args.backend`` /
    ``args.agent_cmd`` (either of which may be ``None``) and falls
    back to the matching environment variables before consulting the
    selected backend's defaults. Front-ends that don't have an
    argparse namespace — notably the TUI's "new run" verb — used to
    hardcode ``"cursor"`` / ``"agent"``, which silently ignored
    ``AGENT_BACKEND`` / ``AGENT_CMD`` and produced a different runner
    from what ``ai run`` would have spawned in the same shell.

    This helper centralises the resolution so both call sites end up
    with the same ``RunConfig`` shape:

    * ``backend`` defaults to ``$AGENT_BACKEND`` (then ``"cursor"``).
    * ``agent_cmd`` defaults to ``$AGENT_CMD`` (then the backend's
      own ``default_cmd``).
    * Model defaults come from the resolved backend so an operator
      whose shell points at Codex doesn't accidentally get Cursor's
      model fingerprints written into ``spec.json``.

    Raises ``ValueError`` if the resolved backend isn't registered or
    the resulting config fails ``RunConfig.validate``."""
    resolved_backend = backend or os.environ.get("AGENT_BACKEND") or "cursor"
    if resolved_backend not in BACKENDS:
        valid = ", ".join(sorted(BACKENDS))
        raise ValueError(
            f"unknown backend '{resolved_backend}'. valid: {valid}"
        )
    be = BACKENDS[resolved_backend]
    resolved_agent_cmd = (
        agent_cmd
        or os.environ.get("AGENT_CMD")
        or be.default_cmd
    )
    cfg = RunConfig(
        task=task,
        impl_model=be.default_impl_model,
        fix_model=be.default_fix_model,
        reviewer_model=be.default_reviewer_model,
        max_outer=max_outer,
        max_inner=max_inner,
        workspace=workspace,
        skip_impl=skip_impl,
        extra_flags=tuple(extra_flags),
        agent_cmd=resolved_agent_cmd,
        backend=resolved_backend,
        use_worktree=use_worktree,
    )
    err = cfg.validate()
    if err:
        raise ValueError(err)
    return cfg


# ── Worktree-relevant convenience ───────────────────────────────────────────


def reload_meta(paths: RunPaths) -> dict:
    """Re-read ``meta.json`` so callers can re-evaluate liveness fresh.

    Tiny helper so TUI screens don't have to import ``meta.read_meta``
    directly — they get the whole control surface from this module."""
    return read_meta(paths) or {}


__all__ = [
    "ActionResult",
    "clear_pause",
    "default_run_config",
    "reload_meta",
    "runner_is_alive",
    "signal_runner",
    "spawn_runner_detached",
    "write_guidance",
    "write_pause",
    "write_prompt",
    "write_rewind",
    "write_rewind_from_to_string",
]
