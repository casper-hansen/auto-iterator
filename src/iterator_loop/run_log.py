"""Structured, queryable run logging for the review loop.

This module is the machine-readable source of truth for a single review-loop
run. Each run owns a dedicated log directory with two files:

* ``events.jsonl`` — append-only JSON Lines stream. One object per meaningful
  event (run lifecycle, phase transitions, outer/inner loop boundaries,
  implementation/review/fix steps, reviewer verdicts, agent session
  resumes, tool-call start/completion, operator guidance, …). Grep-able,
  tail-able, stable ordering by monotonic ``seq``.
* ``state.json`` — a single-object snapshot representing "what would a human
  (or TUI) want to see right now?" for the run. Atomically rewritten on
  every emitted event so a concurrent reader always observes a consistent
  view.

The shared root (``logs/index.jsonl``) records at least ``run_started`` and
``run_finished`` so operators can enumerate runs without walking every
per-run directory.

The writer is intentionally:

* Dependency-free (stdlib only) and daemon-free.
* Thread-safe — ``_StreamReader`` reads the agent PTY in a worker thread,
  so state mutations acquire a lock.
* Crash-safe — events are flushed on append; ``state.json`` is written to
  a ``.tmp`` sibling, ``fsync``-ed, and ``os.replace``-d into place.
* TUI-independent — no external consumer is required. A future TUI or
  another agent is free to read these files directly.
"""

from __future__ import annotations

import json
import numbers
import os
import re
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional


SCHEMA_VERSION = 1
DEFAULT_TAIL_SIZE = 40
INDEX_FILENAME = "index.jsonl"
EVENTS_FILENAME = "events.jsonl"
STATE_FILENAME = "state.json"

# ── Run-id validation ─────────────────────────────────────────────────────
#
# Explicit ``--run-id`` values are untrusted input from the CLI. Without
# validation, a value like ``"../escape"`` resolves *outside* the
# configured logs root (confirmed: ``Path("/tmp/x") / "../escape"`` →
# ``/tmp/escape``), and ``"/abs"`` escapes entirely because Python's
# ``Path("/x") / "/abs"`` is just ``/abs``. Two runs sharing the same
# explicit id also merge into one directory, producing duplicate ``seq``
# values (``[1, 2, 1, 2]``) and breaking every invariant downstream
# consumers rely on.
#
# The rules below guarantee the "one dedicated ``logs/<run_id>/`` per
# run" contract the task spec requires:
#
# * Must be a single filesystem component (no ``/``, ``\\``, null byte).
# * Must not be ``.``/``..`` and must not start with ``.`` or ``-``
#   (avoids hidden-file tricks and CLI-flag ambiguity when the id is
#   pasted into another command).
# * Must match ``[A-Za-z0-9][A-Za-z0-9._-]*`` so directory names stay
#   portable (Windows reserved chars, shell globs, ANSI, …) are out.
# * Length-bounded so pathological inputs can't produce filesystem-
#   length errors far away from the source.
#
# Uniqueness is enforced by rejecting construction when the target
# directory already contains ``events.jsonl`` / ``state.json`` — a pre-
# existing empty sibling directory is tolerated to keep the logger
# friendly for operators who pre-allocate paths, but real run data is
# never clobbered or appended-to.
_RUN_ID_MAX_LEN = 120
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class InvalidRunIdError(ValueError):
    """An explicit ``run_id`` violates the per-run-directory contract.

    Raised by :func:`_sanitize_run_id` (and :class:`RunLogger`) for path
    traversal (``".."``), path separators, absolute paths, empty
    strings, null bytes, or characters outside the portable filename
    allowlist. Subclass of :class:`ValueError` so existing callers that
    just catch ``ValueError`` still pick it up, while more careful
    callers can target the specific type.
    """


def _sanitize_run_id(run_id: Any) -> str:
    """Return *run_id* unchanged if it is a safe directory-name component.

    Raises :class:`InvalidRunIdError` otherwise. Callers should only
    invoke this for user-supplied values; :func:`new_run_id` outputs
    values that match ``_RUN_ID_RE`` by construction.
    """
    if not isinstance(run_id, str):
        raise InvalidRunIdError(
            f"run_id must be a string, got {type(run_id).__name__}"
        )
    if not run_id:
        raise InvalidRunIdError("run_id must not be empty")
    if run_id in (".", ".."):
        raise InvalidRunIdError(
            f"run_id {run_id!r} is a reserved path name"
        )
    if "\x00" in run_id:
        raise InvalidRunIdError("run_id must not contain null bytes")
    if "/" in run_id or "\\" in run_id:
        raise InvalidRunIdError(
            f"run_id {run_id!r} must not contain path separators "
            "(got '/' or '\\\\'); it must be a single directory "
            "component so logs/<run_id>/ remains well-defined"
        )
    if len(run_id) > _RUN_ID_MAX_LEN:
        raise InvalidRunIdError(
            f"run_id too long: {len(run_id)} > {_RUN_ID_MAX_LEN} chars"
        )
    if not _RUN_ID_RE.match(run_id):
        raise InvalidRunIdError(
            f"run_id {run_id!r} must match [A-Za-z0-9][A-Za-z0-9._-]* "
            "(alphanumeric plus '.', '-', '_'; may not start with "
            "'.' or '-')"
        )
    return run_id

#: Phases surfaced in ``state.json``. Kept as plain strings (not an Enum) so
#: a future TUI or another agent can treat them as opaque labels without
#: importing this module.
PHASE_INIT = "init"
PHASE_IMPLEMENTATION = "implementation"
PHASE_REVIEW = "review"
PHASE_FIX = "fix"
PHASE_FINISHED = "finished"


EventSink = Callable[[str, dict], None]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    """Return a sortable, collision-resistant run identifier.

    Format: ``YYYYMMDDTHHMMSSZ-XXXXXX`` where the suffix is the first 6
    hex chars of a random UUID4. Sortable lexicographically by start time
    yet short enough to be comfortable on a filesystem path.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:6]
    return f"{stamp}-{suffix}"


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Atomically replace *path* with a JSON-serialized *obj*.

    Writes to ``<path>.tmp``, ``fsync``s the fd, then ``os.replace``s into
    place. This guarantees any reader observes either the previous or the
    new file, never a torn write.
    """
    tmp = path.with_name(path.name + ".tmp")
    payload = json.dumps(obj, indent=2, sort_keys=False, default=str)
    with open(tmp, "w", encoding="utf-8") as fp:
        fp.write(payload)
        fp.flush()
        try:
            os.fsync(fp.fileno())
        except OSError:
            # fsync may fail on exotic filesystems; os.replace still gives
            # crash-atomicity at the rename step, which is the important
            # property for a "latest snapshot" file.
            pass
    os.replace(tmp, path)


def _append_jsonl(path: Path, obj: Any) -> None:
    """Append a single JSON object as one line to *path*, flushed.

    Kept open/write/close on every call on purpose: a single line is small
    and cheap, and we do not need to keep a long-lived file handle for a
    logger that may persist across many tool-call events.
    """
    line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False, default=str)
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(line + "\n")
        fp.flush()


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "\u2026"


def _merge_tokens(accum: Any, new: Any) -> Any:
    """Merge token-usage *new* into *accum* by summing numeric fields.

    Real Cursor stream ``result`` events carry a dict of numeric counters
    (``input_tokens``, ``output_tokens``, ``cache_read_tokens``, …) per
    session. Every session contributes more work, so the *running* total
    exposed in ``state.json`` must sum counters across sessions instead of
    replacing them. Non-numeric fields (labels, model names, …) are kept
    as the latest observed value rather than concatenated.

    ``None`` inputs are tolerated on both sides to keep the caller's code
    straightforward; the result is ``None`` iff *accum* and *new* are
    both ``None``.
    """
    if new is None:
        return accum
    if not isinstance(new, dict):
        # Scalar usage field: just take the latest observation. We never
        # fabricate arithmetic on opaque payloads.
        return new
    if accum is None:
        accum = {}
    if not isinstance(accum, dict):
        accum = {}
    out = dict(accum)
    for k, v in new.items():
        if isinstance(v, bool):
            # bool is a subclass of int; don't sum flags.
            out[k] = v
        elif isinstance(v, numbers.Real):
            prev = out.get(k)
            if isinstance(prev, numbers.Real) and not isinstance(prev, bool):
                out[k] = prev + v
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def _add_cost(accum: Any, new: Any) -> Any:
    """Accumulate *new* onto *accum*, preferring numeric addition."""
    if new is None:
        return accum
    if accum is None:
        return new
    if (
        isinstance(accum, numbers.Real)
        and isinstance(new, numbers.Real)
        and not isinstance(accum, bool)
        and not isinstance(new, bool)
    ):
        return accum + new
    return new


class _TailBuffer:
    """Rolling "last N lines of agent-visible text" that matches the terminal.

    The CLI streams assistant text as byte-level deltas via
    ``assistant_partial`` events, then emits a single ``assistant_message``
    event once the message completes. Naively pushing every delta as its
    own line fragments a visible line into pieces like
    ``["hel", "lo"]`` and also double-counts the final message. That makes
    ``state.json.tail`` useless as a "what the human sees right now" view.

    This buffer mirrors what ``_StreamReader`` feeds to ``OutputFormatter``
    on stdout:

    * Partial deltas append to an internal ``_pending`` string; every
      ``\\n`` in ``_pending`` closes out a complete line into the rolling
      ring buffer.
    * When the final ``assistant_message`` arrives we know the partials
      already covered its content — flush whatever remains in
      ``_pending`` as the trailing line and move on. We do **not**
      re-process the assembled text, which is how the old code produced
      duplicates.
    * If no partials arrived (agents without streaming), the whole
      ``assistant_message`` text is treated as the content for that
      message: absorbed, then finalized.
    * ``snapshot()`` returns the committed complete lines plus the
      currently-in-flight partial so a TUI sees live streaming, not a
      stale N-1 lines.
    * ``session_reset()`` closes any dangling partial line, so the tail
      is consistent across ``agent_session_started`` boundaries even on
      abrupt resume paths.

    Instances are *not* thread-safe on their own; :class:`RunLogger`
    guards calls with its existing lock.
    """

    def __init__(self, maxlen: int) -> None:
        self._lines: deque[str] = deque(maxlen=maxlen)
        self._pending: str = ""
        self._has_partial_in_message: bool = False
        self._saw_assistant_text_in_session: bool = False

    # ── Event hooks ───────────────────────────────────────────────────────

    def on_partial(self, text: str) -> None:
        """Append a streamed delta, extracting complete lines as they form."""
        if not text:
            return
        self._absorb(text)
        self._has_partial_in_message = True
        self._saw_assistant_text_in_session = True

    def on_assistant_message(self, text: str) -> None:
        """Close out the current message.

        If partials were seen, the pending buffer holds the trailing
        partial line and we simply flush it. If no partials were seen,
        treat *text* as the atomic message content.
        """
        if self._has_partial_in_message:
            self._finalize_pending()
        elif text:
            self._absorb(text)
            self._finalize_pending()
        self._has_partial_in_message = False
        if text:
            self._saw_assistant_text_in_session = True

    def on_aux_text(self, text: str) -> None:
        """Fold text from ``agent_result`` / ``stream_closed`` into the tail.

        Only used when the session has no prior assistant text, so the
        final ``result`` echo is not duplicated on top of the partials
        that already produced the visible lines.
        """
        if not text or self._saw_assistant_text_in_session:
            return
        self._absorb(text)
        self._finalize_pending()
        self._saw_assistant_text_in_session = True

    def session_reset(self) -> None:
        """Close out any in-flight partial line at a session boundary."""
        self._finalize_pending()
        self._has_partial_in_message = False
        self._saw_assistant_text_in_session = False

    # ── Snapshot ──────────────────────────────────────────────────────────

    def snapshot(self) -> list[str]:
        """Return committed lines plus (if any) the in-flight partial.

        Including the partial means a polling TUI sees a streaming line
        as it grows rather than waiting for a newline to land. Consumers
        that need strict "complete lines only" can simply ignore the
        last element when it differs from the previous snapshot.
        """
        out = list(self._lines)
        if self._pending:
            out.append(self._pending)
        return out

    # ── Internals ─────────────────────────────────────────────────────────

    def _absorb(self, text: str) -> None:
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                self._lines.append(line)

    def _finalize_pending(self) -> None:
        if self._pending:
            line = self._pending.rstrip("\r")
            if line:
                self._lines.append(line)
        self._pending = ""


class RunLogger:
    """Writes structured, queryable run logs for a single review-loop run.

    One instance per run. Methods are safe to call from both the asyncio
    event loop and the ``_StreamReader`` PTY thread; a single
    ``threading.Lock`` guards every event append plus state rewrite so
    ``events.jsonl`` stays ordered and ``state.json`` never reflects a
    partially-applied event.
    """

    def __init__(
        self,
        *,
        root_dir: Path | str,
        run_id: Optional[str] = None,
        tail_size: int = DEFAULT_TAIL_SIZE,
    ) -> None:
        self.root = Path(root_dir)
        # Validate explicit ids before any filesystem operations so an
        # attacker-ish ``run_id="../escape"`` can never cause a side
        # effect outside the configured logs root. Auto-generated ids
        # go through :func:`new_run_id` and match the allowlist by
        # construction, but we run them through the same sanitizer to
        # keep the invariant uniform.
        if run_id is None:
            run_id = new_run_id()
        self.run_id = _sanitize_run_id(run_id)

        # Pre-create the root so ``run_dir.parent.resolve()`` has a
        # stable target even when the caller hasn't materialized it yet.
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.root / self.run_id

        # Belt-and-suspenders check. The sanitizer already rejects
        # traversal inputs, but exotic filesystem setups can still let
        # an explicit ``--run-id`` escape the configured logs root
        # *after* resolution. Two concrete cases this guard blocks:
        #
        # 1. ``root_dir`` itself resolves elsewhere (symlinked root,
        #    bind-mounted parent, …) — caught by the parent-resolution
        #    check. A symlinked *root* is fine as long as ``run_dir``
        #    lands directly under it; this check only fails when the
        #    canonical parent doesn't match the canonical root.
        # 2. ``root_dir/<run_id>`` is a pre-existing symlink (or has a
        #    symlink anywhere along its path) that points outside the
        #    root. The old code only checked ``run_dir.parent``, which
        #    doesn't follow a symlink located *at* ``run_dir`` itself.
        #    Reproducer: ``ln -s /tmp/outside logs/linkrun`` then
        #    ``RunLogger(root_dir='logs', run_id='linkrun')`` wrote
        #    ``events.jsonl`` / ``state.json`` into ``/tmp/outside/``
        #    because ``run_dir.resolve()`` was never compared against
        #    ``root.resolve() / run_id``.
        #
        # Both checks together mean: run_dir must canonicalize to
        # ``<canonical root>/<run_id>``, with no symlink redirecting
        # the tail component. We refuse construction otherwise rather
        # than silently leaking artifacts outside the
        # "logs/<run_id>/ per run" tree.
        resolved_root = self.root.resolve()
        resolved_parent = self.run_dir.parent.resolve()
        if resolved_parent != resolved_root:
            raise InvalidRunIdError(
                f"run_id {self.run_id!r} escapes the configured logs "
                f"root: run_dir parent resolves to {resolved_parent}, "
                f"expected {resolved_root}"
            )
        # Detect a symlink *at* run_dir before any mkdir. ``is_symlink``
        # returns True for dangling symlinks too, which is what we want:
        # even a broken symlink shouldn't be followed by ``touch`` /
        # ``_atomic_write_json`` further down.
        if self.run_dir.is_symlink():
            try:
                target = os.readlink(self.run_dir)
            except OSError:  # pragma: no cover - race between stat/readlink
                target = "<unreadable>"
            raise InvalidRunIdError(
                f"run_id {self.run_id!r} already exists as a symbolic "
                f"link under {self.root} (points to {target!r}). "
                "Refusing to write through it: a run directory must be "
                "a real directory directly under the configured logs "
                "root, not a symlink that could redirect artifacts "
                "outside the tree. Remove the link or pick a different "
                "--run-id."
            )
        # Final resolution invariant: catch stacked symlinks anywhere
        # along the path (e.g. a symlinked intermediate the sanitizer
        # wouldn't see). ``Path.resolve`` on a non-existent leaf is
        # well-defined on Python 3.6+: it returns the canonical
        # absolute path the entry *would* have, with any existing
        # symlink components followed. So this check is meaningful
        # both before and after the run directory is created.
        try:
            resolved_run_dir = self.run_dir.resolve()
        except OSError as exc:  # pragma: no cover - exotic FS only
            raise InvalidRunIdError(
                f"run_id {self.run_id!r}: cannot canonicalize "
                f"{self.run_dir}: {exc}"
            ) from exc
        expected_run_dir = resolved_root / self.run_id
        if resolved_run_dir != expected_run_dir:
            raise InvalidRunIdError(
                f"run_id {self.run_id!r} escapes the configured logs "
                f"root after path resolution: run_dir resolves to "
                f"{resolved_run_dir}, expected {expected_run_dir}. "
                "This usually indicates a symlink along the path."
            )

        self.events_path = self.run_dir / EVENTS_FILENAME
        self.state_path = self.run_dir / STATE_FILENAME
        self.index_path = self.root / INDEX_FILENAME

        # Reject run-id reuse outright: merging a new run into a
        # populated directory produces duplicate ``seq`` values
        # (``[1, 2, 1, 2]`` was the reproducer) and makes
        # ``index.jsonl`` ambiguous. Auto-generated timestamp+uuid ids
        # are collision-free in practice, so this branch only fires on
        # an explicit ``--run-id`` the caller reused. An empty
        # pre-existing dir is fine (operators may pre-create paths).
        if self.events_path.exists() or self.state_path.exists():
            raise FileExistsError(
                f"run_id {self.run_id!r} already has log artifacts in "
                f"{self.run_dir} ({EVENTS_FILENAME} / {STATE_FILENAME}). "
                "Refusing to merge a new run into an existing one; "
                "pick a different --run-id."
            )

        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._tail_size = tail_size
        self._tail = _TailBuffer(maxlen=tail_size)
        self._lock = threading.Lock()
        self._seq = 0

        self._state: dict[str, Any] = self._initial_state()
        # Create an empty events.jsonl and an initial state.json up front
        # so a concurrent reader (TUI, sibling agent) can open the run
        # directory immediately without racing the first emit.
        self.events_path.touch(exist_ok=False)
        _atomic_write_json(self.state_path, self._state)

    # ── Public lifecycle helpers ───────────────────────────────────────────

    def start(self, *, config: dict) -> None:
        """Record ``run_started`` in both per-run and root index logs.

        ``config.started_at`` (when provided) is promoted to the
        canonical per-run ``started_at`` *before* anything is written,
        so ``index.jsonl.run_started.started_at`` and
        ``state.json.started_at`` end up agreeing on the exact same
        ISO-8601 string. The old ordering wrote the index entry from
        the constructor-time default and only then let the
        ``run_started`` event handler overwrite ``state.started_at``
        from the config payload — reviewer reproduced a ~2ms drift
        between the two surfaces on a clean run, which is an avoidable
        inconsistency in the machine-readable contract.

        When ``config.started_at`` is absent (e.g. tests that omit it),
        the constructor-time default is used uniformly in both files.
        """
        safe_config = _scrub_config(config)
        # Promote config.started_at to the canonical value first so
        # every downstream write observes the same timestamp. Only a
        # *truthy* value wins — a stray empty string from a scrubber
        # bug would otherwise silently blank out the constructor-time
        # default that tests rely on.
        cfg_started_at = safe_config.get("started_at")
        if cfg_started_at:
            self._state["started_at"] = cfg_started_at
        self._state["config"] = safe_config
        # Expose a small, well-known subset of config as flags that a TUI
        # may want to highlight without re-reading the whole config blob.
        self._state["flags"] = {
            "skip_impl": bool(safe_config.get("skip_impl", False)),
        }
        self._state["max_outer"] = safe_config.get("max_outer")
        self._state["max_inner"] = safe_config.get("max_inner")

        _append_jsonl(
            self.index_path,
            {
                "ts": _now_utc_iso(),
                "type": "run_started",
                "run_id": self.run_id,
                "run_dir": self.run_id,
                "started_at": self._state["started_at"],
                "prompt_preview": _truncate(safe_config.get("prompt", "") or "", 200),
                "workspace": safe_config.get("workspace"),
                "impl_model": safe_config.get("impl_model"),
                "fix_model": safe_config.get("fix_model"),
                "reviewer_model": safe_config.get("reviewer_model"),
            },
        )
        self.emit("run_started", {"config": safe_config})

    def finish(
        self,
        *,
        approved: bool,
        exit_code: int,
        total_reviews: Optional[int] = None,
        outer_loops: Optional[int] = None,
    ) -> None:
        """Record ``run_finished`` in both per-run and root index logs.

        ``total_reviews`` and ``outer_loops`` default to ``None`` and fall
        back to the logger's live in-memory state when omitted. That
        matters most on the crash path: if ``_run_loop()`` emits real
        progress events (``outer_started``, ``review_started``, …) and
        then raises before the caller learns the final counters, the
        caller's locals are still stale (``0``, ``0``). Passing those
        zeros here would overwrite the already-correct live values in
        both ``state.json`` and ``index.jsonl`` — the reviewer's
        reproducer showed ``state.total_reviews=0`` and
        ``index.run_finished.outer_loops=0`` after a mid-loop crash
        that had already emitted two reviews.

        The live state is derived from the same ordered event stream
        that drives ``events.jsonl``, so it is always a consistent
        terminal-state source. Callers that do have authoritative
        counts (happy path, deterministic tests) can still pass them
        explicitly and the explicit values win.
        """
        if total_reviews is None:
            live_total = self._state.get("total_reviews")
            total_reviews = int(live_total) if isinstance(live_total, int) else 0
        if outer_loops is None:
            # ``state["outer"]`` tracks the *current* outer index, which
            # equals "how many outer loops have started" for this run.
            # That is exactly the number we want to report in the final
            # summary. ``None`` means no outer_started / inner_started
            # event ever landed, which maps to zero outer loops.
            live_outer = self._state.get("outer")
            outer_loops = int(live_outer) if isinstance(live_outer, int) else 0

        finished_at = _now_utc_iso()
        summary = {
            "approved": approved,
            "exit_code": exit_code,
            "total_reviews": total_reviews,
            "outer_loops": outer_loops,
            "finished_at": finished_at,
        }
        self.emit("run_finished", summary)
        _append_jsonl(
            self.index_path,
            {
                "ts": finished_at,
                "type": "run_finished",
                "run_id": self.run_id,
                "run_dir": self.run_id,
                **summary,
            },
        )

    # ── Generic emit ───────────────────────────────────────────────────────

    def emit(self, event_type: str, payload: Optional[dict] = None) -> dict:
        """Append an event, apply it to state, atomically rewrite state.json."""
        payload = dict(payload) if payload else {}
        with self._lock:
            self._seq += 1
            seq = self._seq
            ts = _now_utc_iso()
            record = {"seq": seq, "ts": ts, "type": event_type, **payload}
            _append_jsonl(self.events_path, record)
            self._apply_event_to_state(event_type, payload, ts, seq)
            _atomic_write_json(self.state_path, self._state)
            return record

    def agent_event_sink(self) -> EventSink:
        """Return a callable suitable for ``run_agent(event_sink=…)``.

        The sink just forwards ``(event_type, payload)`` to :meth:`emit`,
        so every stream event observed by ``_StreamReader`` lands in the
        same ordered ``events.jsonl`` as orchestration events.
        """
        return self.emit

    # ── Orchestration convenience helpers ─────────────────────────────────
    #
    # These are thin wrappers around emit() so callers don't have to
    # remember stringly-typed event names. The underlying events.jsonl is
    # the stable contract; these helpers are sugar.

    def implementation_started(self, *, model: str, tag: str) -> None:
        self.emit("implementation_started", {"model": model, "tag": tag})

    def implementation_finished(self, *, rc: int) -> None:
        self.emit("implementation_finished", {"rc": rc})

    def outer_started(self, *, outer: int, max_outer: int) -> None:
        self.emit("outer_started", {"outer": outer, "max_outer": max_outer})

    def outer_finished(self, *, outer: int, approved: bool, inner_iterations: int) -> None:
        self.emit("outer_finished", {
            "outer": outer,
            "approved": approved,
            "inner_iterations": inner_iterations,
        })

    def inner_started(self, *, outer: int, inner: int, max_inner: int, tag: str) -> None:
        self.emit("inner_started", {
            "outer": outer, "inner": inner, "max_inner": max_inner, "tag": tag,
        })

    def inner_finished(self, *, outer: int, inner: int, verdict: str) -> None:
        self.emit("inner_finished", {
            "outer": outer, "inner": inner, "verdict": verdict,
        })

    def review_started(self, *, model: str, tag: str) -> None:
        self.emit("review_started", {"model": model, "tag": tag})

    def review_finished(self, *, verdict: str, rc: int, tag: str) -> None:
        self.emit("review_finished", {"verdict": verdict, "rc": rc, "tag": tag})

    def fix_started(self, *, model: str, tag: str) -> None:
        self.emit("fix_started", {"model": model, "tag": tag})

    def fix_finished(self, *, rc: int, tag: str) -> None:
        self.emit("fix_finished", {"rc": rc, "tag": tag})

    def guidance_received(self, *, text: str, source: str = "operator") -> None:
        """Reserved hook for future operator-interception work."""
        self.emit("guidance_received", {"text": text, "source": source})

    # ── State derivation ──────────────────────────────────────────────────

    def _initial_state(self) -> dict[str, Any]:
        now = _now_utc_iso()
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "finished": False,
            "approved": None,
            "exit_code": None,
            "phase": PHASE_INIT,
            "step_tag": None,
            "outer": None,
            "inner": None,
            "max_outer": None,
            "max_inner": None,
            "total_reviews": 0,
            "last_verdict": None,
            "config": None,
            "agent": {
                "model": None,
                "session_active": False,
                "attempt": 0,
                "max_resume_attempts": None,
                "resume_in_progress": False,
                "pending_tools": 0,
                "last_tool": None,
                "saw_result": False,
                "stream_closed": False,
                "last_rc": None,
                "last_rc_display": None,
                "last_error": None,
            },
            # Token/cost metadata is explicitly unknown until an agent
            # stream event surfaces it. ``tokens`` / ``cost`` are the
            # *running* totals across every ``agent_result`` observed this
            # run; ``last_tokens`` / ``last_cost`` expose the most recent
            # sample for at-a-glance displays. Everything starts ``None``
            # so a future consumer can tell "not yet observed" from
            # "observed a zero usage" — we never fabricate numbers.
            "usage": {
                "tokens": None,
                "cost": None,
                "last_tokens": None,
                "last_cost": None,
                "results_observed": 0,
            },
            "tail": [],
            "pending_guidance": None,
            "flags": {},
            # Populated from ``run_error`` events so a crash is visible in
            # the snapshot alongside the real ``exit_code``.
            "error": None,
            "last_event": None,
        }

    def _apply_event_to_state(
        self,
        et: str,
        payload: dict,
        ts: str,
        seq: int,
    ) -> None:
        """Fold *event* into the in-memory ``state`` snapshot.

        Unknown event types are accepted without error: they still update
        ``last_event`` and ``updated_at`` and bump the tail if they carry
        text, leaving room for future event kinds (e.g. operator
        interception) to be added without a schema break.
        """
        st = self._state
        st["updated_at"] = ts
        st["last_event"] = {"seq": seq, "type": et, "ts": ts}

        if et == "run_started":
            st["phase"] = PHASE_INIT
            st["started_at"] = payload.get("config", {}).get("started_at", st["started_at"])
            return

        if et == "run_finished":
            st["phase"] = PHASE_FINISHED
            st["finished"] = True
            st["approved"] = payload.get("approved")
            # ``exit_code`` is the *real* terminal exit status. Callers are
            # responsible for passing what the OS will observe (0 / 1 /
            # 130 / SystemExit.code) — see ``review-loop.main`` — so we
            # do not invent a value here when it's explicitly ``None``.
            st["exit_code"] = payload.get("exit_code")
            st["finished_at"] = payload.get("finished_at", ts)
            if payload.get("total_reviews") is not None:
                st["total_reviews"] = payload["total_reviews"]
            return

        if et == "run_error":
            # Captured alongside ``run_finished``/``exit_code`` so a
            # consumer tailing state.json can distinguish a clean exit
            # from a crash without parsing traceback text.
            st["error"] = {
                "type": payload.get("error_type"),
                "message": payload.get("message"),
                "traceback": payload.get("traceback"),
            }
            return

        if et == "implementation_started":
            st["phase"] = PHASE_IMPLEMENTATION
            st["step_tag"] = payload.get("tag")
            return

        if et == "implementation_finished":
            st["step_tag"] = None
            return

        if et == "outer_started":
            st["outer"] = payload.get("outer")
            st["inner"] = None
            return

        if et == "outer_finished":
            # Retain outer index so a snapshot taken between outers still
            # tells a reader which outer just ran.
            return

        if et == "inner_started":
            st["outer"] = payload.get("outer", st["outer"])
            st["inner"] = payload.get("inner")
            st["step_tag"] = payload.get("tag")
            return

        if et == "inner_finished":
            return

        if et == "review_started":
            st["phase"] = PHASE_REVIEW
            st["step_tag"] = payload.get("tag", st["step_tag"])
            st["total_reviews"] = (st.get("total_reviews") or 0) + 1
            return

        if et == "review_finished":
            st["last_verdict"] = payload.get("verdict")
            return

        if et == "fix_started":
            st["phase"] = PHASE_FIX
            st["step_tag"] = payload.get("tag", st["step_tag"])
            return

        if et == "fix_finished":
            return

        if et == "guidance_received":
            st["pending_guidance"] = {
                "text": payload.get("text"),
                "source": payload.get("source", "operator"),
                "received_at": ts,
            }
            return

        # ── Agent-session events (from run_agent + _StreamReader) ─────
        agent = st["agent"]

        if et == "agent_session_started":
            agent["session_active"] = True
            agent["model"] = payload.get("model", agent["model"])
            agent["attempt"] = payload.get("attempt", 0)
            agent["max_resume_attempts"] = payload.get("max_resume_attempts")
            agent["resume_in_progress"] = False
            # Reset per-session counters. ``last_tool`` is deliberately
            # retained across sessions so a TUI still sees the most recent
            # observed tool call even after a resume with no further tool
            # activity.
            agent["pending_tools"] = 0
            agent["saw_result"] = False
            agent["stream_closed"] = False
            agent["last_rc"] = None
            agent["last_rc_display"] = None
            agent["last_error"] = None
            # Close out any in-flight partial line from the previous
            # session so the tail doesn't concatenate a dangling "worl"
            # with the new session's first delta.
            self._tail.session_reset()
            st["tail"] = self._tail.snapshot()
            return

        if et == "agent_session_finished":
            agent["session_active"] = False
            agent["last_rc"] = payload.get("rc", agent["last_rc"])
            agent["last_rc_display"] = payload.get("rc_display", agent["last_rc_display"])
            agent["pending_tools"] = payload.get(
                "pending_tools", agent["pending_tools"],
            )
            agent["saw_result"] = payload.get("saw_result", agent["saw_result"])
            agent["stream_closed"] = payload.get(
                "stream_closed", agent["stream_closed"],
            )
            return

        if et == "agent_resume_started":
            agent["resume_in_progress"] = True
            agent["attempt"] = payload.get("attempt", agent["attempt"])
            agent["max_resume_attempts"] = payload.get(
                "max_resume_attempts", agent["max_resume_attempts"],
            )
            return

        if et == "agent_exit_abnormal":
            agent["last_rc"] = payload.get("rc", agent["last_rc"])
            agent["last_rc_display"] = payload.get(
                "rc_display", agent["last_rc_display"],
            )
            agent["last_error"] = payload.get("message")
            return

        if et == "agent_resume_giveup":
            agent["resume_in_progress"] = False
            agent["last_error"] = payload.get("message", agent["last_error"])
            return

        # ── Stream events (assistant text, tool calls, result) ────────

        if et == "tool_call_started":
            agent["pending_tools"] += 1
            agent["last_tool"] = {
                "phase": "started",
                "name": payload.get("name"),
                "summary": payload.get("summary"),
            }
            return

        if et == "tool_call_completed":
            agent["pending_tools"] = max(0, agent["pending_tools"] - 1)
            agent["last_tool"] = {
                "phase": "completed",
                "name": payload.get("name"),
                "summary": payload.get("summary"),
            }
            return

        if et == "assistant_partial":
            self._tail.on_partial(payload.get("text", "") or "")
            st["tail"] = self._tail.snapshot()
            return

        if et == "assistant_message":
            self._tail.on_assistant_message(payload.get("text", "") or "")
            st["tail"] = self._tail.snapshot()
            return

        if et == "agent_result":
            agent["saw_result"] = True
            usage = payload.get("usage")
            cost = payload.get("cost")
            u = st["usage"]
            if usage is not None:
                u["tokens"] = _merge_tokens(u.get("tokens"), usage)
                u["last_tokens"] = usage
            if cost is not None:
                u["cost"] = _add_cost(u.get("cost"), cost)
                u["last_cost"] = cost
            if usage is not None or cost is not None:
                u["results_observed"] = int(u.get("results_observed") or 0) + 1
            # ``result.text`` is usually a near-copy of what partials
            # already produced. Only add it to the tail if the session
            # never surfaced assistant text, so we don't duplicate.
            self._tail.on_aux_text(payload.get("text") or "")
            st["tail"] = self._tail.snapshot()
            return

        if et == "stream_closed":
            agent["stream_closed"] = True
            self._tail.on_aux_text(payload.get("text") or "")
            st["tail"] = self._tail.snapshot()
            return

        # Fall-through: unknown event type. We still updated last_event +
        # updated_at above, and we keep the in-memory state stable.


# ── Helpers ───────────────────────────────────────────────────────────────


_CONFIG_ALLOWLIST = {
    "prompt",
    "context",
    "impl_model",
    "fix_model",
    "reviewer_model",
    "max_outer",
    "max_inner",
    "workspace",
    "skip_impl",
    "extra_flags",
    "agent_cmd",
    "started_at",
}


def _scrub_config(config: dict) -> dict:
    """Return a JSON-serializable copy of *config* limited to known keys.

    Anything that is not JSON-friendly (e.g. a ``Path``) is stringified.
    Unknown keys are dropped to keep ``state.json`` schema-stable.
    """
    out: dict[str, Any] = {}
    for key in _CONFIG_ALLOWLIST:
        if key not in config:
            continue
        val = config[key]
        if isinstance(val, (list, tuple)):
            out[key] = [str(v) for v in val]
        elif isinstance(val, (str, int, float, bool)) or val is None:
            out[key] = val
        else:
            out[key] = str(val)
    return out
