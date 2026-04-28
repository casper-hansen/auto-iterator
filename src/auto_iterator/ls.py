"""Status reconciliation for ``ai ls`` / ``ai show``.

Key invariant: ``meta.status`` is never trusted as authoritative. The
runner may have been SIGKILLed before it could transition to ``exited``,
or may be genuinely wedged with a stale heartbeat even though meta still
claims ``running``. Every reader routes through :func:`reconcile_status`
below to get the *true* state right now."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .events import tail_events
from .heartbeat import STUCK_THRESHOLD_SECONDS
from .meta import read_meta
from .run_dir import RunPaths, pid_alive, read_json, read_last_jsonl


# Reconciled status values. Keep this set small so CLI output is easy to
# scan. The runner-written statuses in meta.json are folded into these
# at read time — callers don't need to learn both vocabularies.
RECONCILED_STATUSES = frozenset({
    "running", "exited", "killed", "crashed", "stuck", "approved", "unapproved",
})


@dataclass
class RunRow:
    """One row in ``ai ls`` output. Built by :func:`summarize_run`."""

    run_id: str
    status: str
    phase: str
    outer: int
    inner: int
    last_verdict: str
    exit_code: Optional[int]
    approved: bool
    started_at: str
    updated_at: str
    workspace: str
    prompt_preview: str
    pid: Optional[int]

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "phase": self.phase,
            "outer": self.outer,
            "inner": self.inner,
            "last_verdict": self.last_verdict,
            "exit_code": self.exit_code,
            "approved": self.approved,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "workspace": self.workspace,
            "prompt_preview": self.prompt_preview,
            "pid": self.pid,
        }


def reconcile_status(
    paths: RunPaths,
    meta: dict,
    *,
    state: Optional[dict] = None,
) -> str:
    """Return the *actual* status of a run right now.

    Order of precedence (matches spec):
      1. Event-log tail = ``run_finished`` OR ``state.finished`` → terminal.
      2. pid alive + heartbeat fresh (< 30s) → ``running``.
      3. pid alive + heartbeat stale → ``stuck``.
      4. else → ``crashed``.

    ``state`` may be passed in by callers that already loaded ``state.json``
    (``summarize_run`` does) to avoid a duplicate read."""
    # 1) Terminal state wins unconditionally.
    if state is None:
        state = _read_state(paths)
    last_evt = read_last_jsonl(paths.events)
    if (state and state.get("finished")) or (
        last_evt and last_evt.get("type") == "run_finished"
    ):
        approved = bool(state.get("approved")) if state else False
        exit_code = state.get("exit_code") if state else None
        if meta.get("status") == "killed":
            return "killed"
        if exit_code == 0 or approved:
            return "approved" if approved else "exited"
        return "unapproved" if state else "exited"

    # 2/3/4) pid + heartbeat reconciliation.
    pid = meta.get("pid")
    if not isinstance(pid, int) or not pid_alive(pid):
        # meta's "exited"/"killed" only makes it here if the terminal
        # event never wrote — rare but possible (the signal handler
        # scheduled meta but not the event). Surface that explicitly.
        if meta.get("status") in ("exited", "killed"):
            return meta["status"]
        return "crashed"

    hb_age = _heartbeat_age(paths)
    if hb_age is None or hb_age > STUCK_THRESHOLD_SECONDS:
        return "stuck"
    return "running"


def _heartbeat_age(paths: RunPaths) -> Optional[float]:
    try:
        mtime = paths.heartbeat.stat().st_mtime
    except OSError:
        return None
    return max(0.0, time.time() - mtime)


def _read_state(paths: RunPaths) -> Optional[dict]:
    try:
        return read_json(paths.state)
    except (FileNotFoundError, ValueError):
        return None


def summarize_run(paths: RunPaths) -> Optional[RunRow]:
    """Build a one-row summary by reconciling meta + state + events + pid."""
    meta = read_meta(paths)
    if not meta:
        return None
    state = _read_state(paths) or {}
    status = reconcile_status(paths, meta, state=state)
    return RunRow(
        run_id=meta.get("run_id", paths.run_id),
        status=status,
        phase=state.get("phase", meta.get("status", "unknown")),
        outer=int(state.get("outer", 0) or 0),
        inner=int(state.get("inner", 0) or 0),
        last_verdict=state.get("last_verdict", "") or "",
        exit_code=state.get("exit_code"),
        approved=bool(state.get("approved", False)),
        started_at=meta.get("started_at", "") or state.get("started_at", "") or "",
        updated_at=state.get("updated_at", "") or meta.get("heartbeat_at", "") or "",
        workspace=meta.get("workspace", "") or state.get("workspace", "") or "",
        prompt_preview=state.get("prompt_preview", "") or "",
        pid=meta.get("pid") if isinstance(meta.get("pid"), int) else None,
    )


def list_runs(runs_dir: Path) -> list[RunRow]:
    """Scan *runs_dir* and return summaries for every run, newest first."""
    from .run_dir import iter_run_dirs

    out: list[RunRow] = []
    for paths in iter_run_dirs(runs_dir):
        row = summarize_run(paths)
        if row is None:
            continue
        out.append(row)
    out.sort(key=lambda r: r.started_at, reverse=True)
    return out
