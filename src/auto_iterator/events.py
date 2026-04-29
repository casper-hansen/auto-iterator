"""Event + state writers for a single run.

Every meaningful state transition in the runner produces a JSON object that
is both appended to ``events.jsonl`` and folded into the latest snapshot in
``state.json``. Keeping both forms on disk means:

* ``events.jsonl`` is a plain append-only log that no reader ever has to
  seek backwards in, so jq-style scripting against the raw stream stays
  trivial.
* ``ai show`` / ``ai ls`` get the current state in O(1) by reading the
  tiny ``state.json`` snapshot, without replaying the full event log.

Event shape: ``{"seq": N, "type": "...", "timestamp": "...", ...}``. The
``seq`` field monotonically increases so a resuming reader can pick up
where it left off after a disconnect.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .run_dir import RunPaths, append_jsonl, atomic_write_json, now_iso


# The set of event types the runner emits. Kept as a frozenset (rather
# than inline magic strings in every ``log.emit`` call site) so a grep
# turns up the full vocabulary, tests can assert against it, and future
# additions have an obvious home. Downstream tooling can add new event
# types without pinning the CLI's version because consumers read raw
# JSONL and don't need a closed enum.
EVENT_TYPES = frozenset({
    "run_started",
    "run_finished",
    "impl_started",
    "impl_finished",
    "outer_started",
    "inner_started",
    "review_started",
    "review_finished",
    "fix_started",
    "fix_finished",
    "guidance_received",
    "rewind_applied",
    "rewind_narrowed",
    "prompt_updated",
    "paused",
    "resumed",
    "control_rejected",
    "outer_finished",
})


@dataclass
class RunState:
    """Mutable snapshot of what a run is doing right now.

    This object is the single source of truth for both ``state.json`` on
    disk and the in-process control flow of the runner.

    * ``prompt`` starts from ``RunConfig.task`` but can be mutated by
      ``ai set-prompt`` between boundaries.
    * ``history`` is the accumulated review/fix conversation within the
      current outer loop; it resets at the top of each outer, or is
      truncated by ``rewind``.
    * ``guidance_queue`` holds operator-sent steering text waiting to be
      folded into the next review prompt (drained at ``inner_started``).
    """

    run_id: str
    prompt: str
    outer: int = 0
    inner: int = 0
    phase: str = "init"
    history: list[dict[str, str]] = field(default_factory=list)
    approved: bool = False
    last_verdict: str = ""
    total_reviews: int = 0
    exit_code: int | None = None
    finished: bool = False
    started_at: str = field(default_factory=now_iso)
    finished_at: str | None = None
    updated_at: str = field(default_factory=now_iso)
    workspace: str = ""
    guidance_queue: list[str] = field(default_factory=list)
    paused: bool = False

    def snapshot(self) -> dict[str, Any]:
        """Render as the ``state.json`` payload.

        Excludes transient in-memory structures (``history`` contents,
        ``guidance_queue``) — those belong to the runner's internal flow
        and are either large or redundant with events.jsonl. The preview
        fields are derived so readers don't need the full strings.
        """
        return {
            "run_id": self.run_id,
            "status": "finished" if self.finished else "running",
            "phase": self.phase,
            "outer": self.outer,
            "inner": self.inner,
            "approved": self.approved,
            "last_verdict": self.last_verdict,
            "total_reviews": self.total_reviews,
            "exit_code": self.exit_code,
            "finished": self.finished,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "updated_at": self.updated_at,
            "workspace": self.workspace,
            "history_entries": len(self.history),
            "guidance_pending": len(self.guidance_queue),
            "paused": self.paused,
            "prompt_preview": (self.prompt or "")[:200],
        }


class EventLog:
    """Thread-safe event appender + state snapshotter.

    The runner and the heartbeat thread can both touch the snapshot, so we
    serialize writes with a lock. Appends to ``events.jsonl`` go through
    ``O_APPEND`` which is itself atomic for small lines, but we still take
    the lock so the ``seq`` counter stays monotonic under concurrent
    emitters (we don't currently have one, but the invariant is cheap to
    preserve and surprise-free for callers)."""

    def __init__(self, paths: RunPaths, state: RunState) -> None:
        self.paths = paths
        self.state = state
        self._lock = threading.Lock()
        self._seq = self._recover_seq()

    def _recover_seq(self) -> int:
        """Resume seq numbering after a restart by scanning the tail."""
        from .run_dir import read_last_jsonl  # local import avoids cycles
        last = read_last_jsonl(self.paths.events)
        if isinstance(last, dict) and isinstance(last.get("seq"), int):
            return int(last["seq"])
        return 0

    def emit(self, evt_type: str, **fields: Any) -> dict[str, Any]:
        """Append one event and refresh ``state.json``.

        Returns the emitted event dict (with ``seq`` and ``timestamp``
        populated) so the caller can thread it back into
        ``control-applied.jsonl`` audits."""
        if evt_type not in EVENT_TYPES:
            # Soft-fail: unknown types are still written (so a forward-compat
            # event doesn't vanish) but we add a debug hint for the reader.
            fields = {**fields, "_unknown_type": True}
        with self._lock:
            self._seq += 1
            evt = {
                "seq": self._seq,
                "type": evt_type,
                "timestamp": now_iso(),
                **fields,
            }
            append_jsonl(self.paths.events, evt)
            self.state.updated_at = evt["timestamp"]
            self._refresh_snapshot_locked()
        return evt

    def _refresh_snapshot_locked(self) -> None:
        atomic_write_json(self.paths.state, self.state.snapshot())

    def refresh_snapshot(self) -> None:
        """Rewrite ``state.json`` without emitting a new event.

        Used by the heartbeat thread and by callers that mutated the state
        directly (e.g. ``paused`` flag toggles) and want the snapshot
        back in sync."""
        with self._lock:
            self.state.updated_at = now_iso()
            self._refresh_snapshot_locked()

    def audit(self, payload: dict[str, Any]) -> None:
        """Append to ``control-applied.jsonl`` — the operator audit log.

        Separate from ``events.jsonl`` because operator intents and runner
        events are two different audit lenses; keeping them in separate
        files lets readers of ``events.jsonl`` stay focused on runner
        progress while ``ai send --wait`` can watch the audit log
        without noise."""
        append_jsonl(self.paths.control_applied, {
            "timestamp": now_iso(),
            **payload,
        })


def tail_events(path: Path, n: int = 200) -> list[dict[str, Any]]:
    """Return up to *n* most-recent complete events from *path*.

    Handles the case where the file doesn't exist yet (returns ``[]``) and
    skips unparseable leftovers rather than erroring out — a partially
    written trailing line is common if a reader races with the writer."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
