"""Review-loop runner — events + control + the original implement/review/fix flow.

This module owns the semantic loop: *implement* → *outer loop* → *inner
loop of review + fix* → *fresh-eyes validation* → *summary*. On top of
the original behaviour it now:

* emits a ``events.jsonl`` stream for ``ai tail``;
* refreshes ``state.json`` after every meaningful transition;
* drains operator control files at ``inner_started`` boundaries only (so
  we never reach into a live agent stream);
* supports a ``rewind`` intent that truncates history and re-enters the
  loop at a specified ``(outer, inner, phase)``;
* pauses at boundaries when ``control/pause`` is present;
* keeps a heartbeat file ticking so ``ai ls`` can separate ``stuck`` from
  ``crashed``.

When no control files are ever dropped and the runs-dir is a throwaway
tmpdir, the resulting console output is behaviourally identical to the
pre-filesystem review-loop — the event/state/control machinery is pure
side-band state."""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import signal
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .control import RewindIntent, drain_control, wait_while_paused
from .events import EventLog, RunState
from .feature.config import RunConfig
from .feature.steps import run_fix, run_implementation, run_review
from .heartbeat import Heartbeat
from .logging import banner, log, make_tag, ok, section, summary, warn
from .meta import update_meta
from .run_dir import (
    RunPaths,
    append_jsonl,
    atomic_write_json,
    now_iso,
)


# ── Run-dir bootstrap & teardown ─────────────────────────────────────────────


def cfg_to_spec(cfg: RunConfig, *, agent_type: str = "review-loop") -> dict:
    """Serialize a RunConfig into the ``spec.json`` payload.

    ``spec.json`` is write-once at startup and drives ``ai restart``, so
    it must be a lossless snapshot of *everything* a fresh runner needs
    to repeat this run."""
    d = asdict(cfg)
    # tuples -> lists for JSON, but keep the shape otherwise identical.
    d["extra_flags"] = list(d.get("extra_flags", []))
    d["agent_type"] = agent_type
    return d


def spec_to_cfg(spec: dict) -> RunConfig:
    """Reverse of :func:`cfg_to_spec`; tolerant of missing optional fields."""
    kwargs = {
        "task": spec["task"],
        "impl_model": spec["impl_model"],
        "fix_model": spec["fix_model"],
        "reviewer_model": spec["reviewer_model"],
        "max_outer": int(spec["max_outer"]),
        "max_inner": int(spec["max_inner"]),
        "workspace": spec["workspace"],
        "skip_impl": bool(spec.get("skip_impl", False)),
        "extra_flags": tuple(spec.get("extra_flags", [])),
        "agent_cmd": spec.get("agent_cmd", "agent"),
        "backend": spec.get("backend", "cursor"),
    }
    return RunConfig(**kwargs)


def bootstrap_run(
    paths: RunPaths,
    cfg: RunConfig,
    *,
    pid: int,
    agent_type: str = "review-loop",
) -> None:
    """Write the initial ``meta.json`` / ``spec.json`` and index entry.

    Called *before* the main loop starts, by ``ai run --foreground``, by
    the foreground review-loop wrapper, and by ``ai run``'s detached
    parent (which invokes this itself; the detached child re-stamps its
    real pid from ``_main_from_run_dir`` rather than calling back here).
    ``spec.json`` is rewritten every call — callers that need write-once
    semantics gate on ``paths.spec.exists()`` themselves."""
    atomic_write_json(paths.spec, cfg_to_spec(cfg, agent_type=agent_type))
    update_meta(
        paths,
        run_id=paths.run_id,
        pid=pid,
        status="running",
        started_at=now_iso(),
        finished_at=None,
        workspace=cfg.workspace,
        agent_type=agent_type,
        heartbeat_at=now_iso(),
    )
    append_jsonl(paths.index, {
        "event": "run_started",
        "timestamp": now_iso(),
        "run_id": paths.run_id,
        "workspace": cfg.workspace,
        "agent_type": agent_type,
        "pid": pid,
    })


def finalize_run(
    paths: RunPaths,
    *,
    exit_code: int,
    status: str,
    approved: bool,
) -> None:
    """Mark the run terminated in ``meta.json`` and ``index.jsonl``.

    Invoked from three paths: the runner's normal exit, its top-level
    exception handler, and the signal / atexit handlers in
    ``_install_signal_handlers``. All callers gate on
    ``ReviewLoopRunner._finalized`` so re-entry is a no-op — the first
    writer wins and subsequent paths bail out cleanly."""
    update_meta(
        paths,
        status=status,
        finished_at=now_iso(),
        exit_code=exit_code,
        approved=approved,
    )
    append_jsonl(paths.index, {
        "event": "run_finished",
        "timestamp": now_iso(),
        "run_id": paths.run_id,
        "status": status,
        "exit_code": exit_code,
        "approved": approved,
    })


# ── The loop itself ──────────────────────────────────────────────────────────


class ReviewLoopRunner:
    """Implements the review-loop semantics with event + control awareness."""

    def __init__(
        self,
        cfg: RunConfig,
        paths: RunPaths,
        *,
        agent_type: str = "review-loop",
    ) -> None:
        self.cfg = cfg
        self.paths = paths
        self.agent_type = agent_type
        self.state = RunState(
            run_id=paths.run_id,
            prompt=cfg.task,
            workspace=cfg.workspace,
        )
        self.log = EventLog(paths, self.state)
        self.heartbeat = Heartbeat(
            paths.heartbeat,
            on_beat=lambda ts: update_meta(paths, heartbeat_at=ts),
        )
        self._finalized = False

    # ── Finalization ─────────────────────────────────────────────────

    def finalize(self, exit_code: int, status: str) -> None:
        """Emit ``run_finished`` + write terminal meta. Idempotent."""
        if self._finalized:
            return
        self._finalized = True
        self.state.exit_code = exit_code
        self.state.finished = True
        self.state.finished_at = now_iso()
        self.log.emit(
            "run_finished",
            approved=self.state.approved,
            exit_code=exit_code,
            status=status,
            total_reviews=self.state.total_reviews,
            outer=self.state.outer,
            inner=self.state.inner,
        )
        finalize_run(
            self.paths,
            exit_code=exit_code,
            status=status,
            approved=self.state.approved,
        )
        try:
            self.heartbeat.stop()
        except Exception:
            pass

    # ── Main entry ───────────────────────────────────────────────────

    async def run(self) -> int:
        self.heartbeat.start()
        self.log.emit(
            "run_started",
            run_id=self.paths.run_id,
            workspace=self.cfg.workspace,
            backend=self.cfg.backend,
            agent_cmd=self.cfg.agent_cmd,
            agent_type=self.agent_type,
            prompt_preview=(self.cfg.task or "")[:200],
        )
        banner("Review Loop", self.cfg.banner_items())

        exit_status = "exited"
        try:
            if not self.cfg.skip_impl:
                await self._run_implementation()
            else:
                log("Skipping implementation (--skip-impl)")
                print()
            self.state.phase = "after_impl"
            self.log.refresh_snapshot()

            await self._run_outer_loop()

            if not self.state.approved:
                if self.state.last_verdict == "APPROVED":
                    warn(
                        f"Inner loop converged but MAX_OUTER ({self.cfg.max_outer}) "
                        "exhausted without a clean fresh-eyes pass"
                    )
                else:
                    warn(
                        f"Exhausted {self.cfg.max_outer} outer loop(s) "
                        "without reaching approval"
                    )
        except Exception as exc:  # pragma: no cover - safety net
            exit_status = "crashed"
            warn(f"Runner hit an unexpected exception: {exc!r}")
            self.log.emit("control_rejected", kind="runner", reason=repr(exc))
            self.finalize(exit_code=2, status="crashed")
            return 2

        summary(
            approved=self.state.approved,
            total_reviews=self.state.total_reviews,
            outer_loops=self.state.outer,
            max_outer=self.cfg.max_outer,
            max_inner=self.cfg.max_inner,
        )
        exit_code = 0 if self.state.approved else 1
        self.finalize(exit_code=exit_code, status=exit_status)
        return exit_code

    # ── Implementation phase ─────────────────────────────────────────

    async def _run_implementation(self) -> None:
        self.state.phase = "impl"
        self.log.emit("impl_started", model=self.cfg.impl_model)
        # The pre-existing step emits its own console banner; we keep
        # that exactly so console output stays unchanged for humans.
        await run_implementation(self.cfg)
        self.log.emit("impl_finished")

    # ── Outer loop ───────────────────────────────────────────────────

    async def _run_outer_loop(self) -> None:
        """Drive fresh-eyes retries with ``rewind`` awareness.

        A rewind intent drained at ``inner_started`` may target either
        the current outer iteration (handled by ``_run_inner_loop``) or
        a *different* outer iteration (bubbled up here). This method
        restarts the correct outer iteration with fresh history."""
        self.state.outer = 0
        while self.state.outer < self.cfg.max_outer:
            self.state.outer += 1
            self.state.history = []
            self.state.inner = 0
            self.state.phase = "outer_started"
            self.log.emit(
                "outer_started",
                outer=self.state.outer,
                max_outer=self.cfg.max_outer,
            )
            section(
                f"Outer Loop {self.state.outer}/{self.cfg.max_outer} — fresh context"
            )

            rewind = await self._run_inner_loop()
            if rewind is not None:
                self._apply_cross_outer_rewind(rewind)
                continue

            self.log.emit(
                "outer_finished",
                outer=self.state.outer,
                verdict=self.state.last_verdict,
                inner=self.state.inner,
            )

            if self.state.last_verdict != "APPROVED":
                warn(
                    f"Inner loop did not reach approval after {self.cfg.max_inner} "
                    "iteration(s) — outer loop will retry with fresh context",
                )
                print()
                continue

            if self.state.inner == 1:
                self.state.approved = True
                self.log.refresh_snapshot()
                ok(
                    "Approved on first pass" if self.state.outer == 1
                    else f"Fresh-eyes review approved on outer loop {self.state.outer}"
                )
                break

            ok(
                f"Inner loop converged after {self.state.inner} iteration(s) — "
                "starting fresh-eyes validation in next outer loop",
            )
            print()

    def _apply_cross_outer_rewind(self, rewind: RewindIntent) -> None:
        """Reset state so the next outer iteration targets ``rewind``.

        For ``after_impl`` we drop back to ``outer=0`` so the next
        iteration increments to 1 with a clean history (matches what the
        initial bootstrap does). For a cross-outer jump we position
        ``outer`` one below the target so the top of the loop advances
        into it with fresh state.

        ``rewind.inner`` and ``rewind.phase`` are intentionally discarded
        here: a cross-outer jump means "start the target outer with fresh
        history", and the inner loop always opens with a review at
        inner=1 — there's nowhere to latch ``phase="fix"`` or ``inner>1``
        onto. We emit ``rewind_narrowed`` so the discard isn't silent."""
        if rewind.phase == "after_impl":
            self.state.outer = 0
            self.state.inner = 0
            self.state.history = []
            self.state.phase = "after_impl"
            self.log.refresh_snapshot()
            return
        target = max(1, min(rewind.outer, self.cfg.max_outer))
        if rewind.inner != 1 or rewind.phase != "review":
            self.log.emit(
                "rewind_narrowed",
                requested={
                    "outer": rewind.outer,
                    "inner": rewind.inner,
                    "phase": rewind.phase,
                },
                applied={"outer": target, "inner": 1, "phase": "review"},
                reason="cross-outer rewinds start with fresh history",
            )
        self.state.outer = target - 1
        self.state.inner = 0
        self.state.history = []
        self.state.phase = "rewinding"
        self.log.refresh_snapshot()

    # ── Inner loop ───────────────────────────────────────────────────

    async def _run_inner_loop(self) -> Optional[RewindIntent]:
        """Run the review/fix inner cycle for the current outer iteration.

        Returns a ``RewindIntent`` if the operator asked to jump to a
        *different* outer iteration; the caller handles that. Returns
        ``None`` once the inner loop ends normally (approved, exhausted,
        or same-outer rewind consumed internally)."""
        self.state.inner = 0
        while self.state.inner < self.cfg.max_inner:
            self.state.inner += 1
            self.state.phase = "review"
            self.state.total_reviews += 1
            self.log.emit(
                "inner_started",
                outer=self.state.outer,
                inner=self.state.inner,
                max_inner=self.cfg.max_inner,
            )

            # Boundary: pause, then drain control intents.
            wait_while_paused(self.paths, self.state, self.log)
            rewind = drain_control(self.paths, self.state, self.log)
            if rewind is not None:
                result = self._apply_same_outer_rewind(rewind)
                if result is not None:
                    # Cross-outer rewind — bubble up to _run_outer_loop.
                    return result
                # Same-outer rewind consumed — restart the inner loop
                # iteration with the adjusted state.
                continue

            tag = make_tag(self.state.outer, self.state.inner)

            # Review.
            guidance = list(self.state.guidance_queue)
            self.log.emit(
                "review_started",
                outer=self.state.outer,
                inner=self.state.inner,
                model=self.cfg.reviewer_model,
                guidance_count=len(guidance),
            )
            verdict, review_text = await run_review(
                self.cfg,
                self.state.history,
                tag,
                task=self.state.prompt,
                guidance=guidance,
            )
            # Guidance is one-shot — clear after folding in.
            self.state.guidance_queue = []
            self.state.last_verdict = verdict
            print()
            self.log.emit(
                "review_finished",
                outer=self.state.outer,
                inner=self.state.inner,
                verdict=verdict,
                review_chars=len(review_text),
            )

            if verdict == "APPROVED":
                ok("Reviewer approved", tag)
                return None

            if self.state.inner >= self.cfg.max_inner:
                warn(
                    f"Inner loop exhausted ({self.cfg.max_inner} iterations)", tag,
                )
                return None

            # Fix.
            self.state.phase = "fix"
            self.log.emit(
                "fix_started",
                outer=self.state.outer,
                inner=self.state.inner,
                model=self.cfg.fix_model,
            )
            rc, fix_text = await run_fix(
                self.cfg,
                self.state.history,
                tag,
                task=self.state.prompt,
            )
            print()
            self.log.emit(
                "fix_finished",
                outer=self.state.outer,
                inner=self.state.inner,
                rc=rc,
                fix_chars=len(fix_text),
            )
        return None

    def _apply_same_outer_rewind(self, rewind: RewindIntent) -> Optional[RewindIntent]:
        """Apply a rewind intent; bubble up if it targets a different outer.

        Same-outer rewinds truncate history and reposition the inner
        counter *one below* the target so the enclosing ``while`` loop
        advances into the target iteration cleanly. Fix-phase rewinds
        keep the matching review in history so the agent has the verdict
        it's nominally fixing against."""
        if rewind.phase == "after_impl" or rewind.outer != self.state.outer:
            return rewind
        # Clamp inner to a valid range.
        target_inner = max(1, min(rewind.inner, self.cfg.max_inner))
        if rewind.phase == "review":
            # Truncate: keep 2*(target-1) entries — all prior rev/fix pairs.
            keep = 2 * (target_inner - 1)
            self.state.history = self.state.history[:keep]
            self.state.inner = target_inner - 1
        elif rewind.phase == "fix":
            # Keep 2*(target-1) + 1 entries — prior pairs plus this
            # round's review, so the fix agent sees what it's fixing.
            keep = 2 * (target_inner - 1) + 1
            if len(self.state.history) < keep:
                # We don't have the required review in history; rewind
                # falls back to the review phase at the same inner.
                keep = 2 * (target_inner - 1)
                self.state.history = self.state.history[:keep]
                self.state.inner = target_inner - 1
            else:
                self.state.history = self.state.history[:keep]
                # Reposition to *this* inner's fix — the enclosing while
                # will re-increment to target_inner on the next pass.
                self.state.inner = target_inner - 1
        self.state.total_reviews = max(0, self.state.total_reviews - 1)
        self.state.phase = "rewinding"
        self.log.refresh_snapshot()
        return None


# ── Synchronous entry points + __main__ spawn target ─────────────────────────


def _install_signal_handlers(runner: "ReviewLoopRunner") -> None:
    """Finalize cleanly on SIGTERM / SIGINT so ``meta.json`` reflects reality.

    Without this a ``kill`` from ``ai kill`` leaves ``meta.status`` at
    ``running`` forever and readers have to wait for the heartbeat to go
    stale before they realise the run is gone. With this, ``meta.status``
    flips to ``killed`` immediately and the index records the transition."""

    def handler(signum, _frame):
        if runner._finalized:
            return
        # Ctrl-C is an operator-initiated kill just like SIGTERM; surfacing
        # it as "exited" would mask the cancellation in ``ai ls``.
        status = "killed" if signum in (
            signal.SIGTERM, signal.SIGHUP, signal.SIGINT,
        ) else "exited"
        try:
            runner.finalize(exit_code=128 + signum, status=status)
        finally:
            # Re-raise the default signal so the process actually dies.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            # ValueError: signal not installable from a thread.
            # OSError: signal not available on this platform.
            pass

    def at_exit():
        if runner._finalized:
            return
        # Reaching atexit without an explicit finalize means the loop
        # exited via an uncaught exception or sys.exit(): not a clean run.
        exit_code = runner.state.exit_code if runner.state.exit_code is not None else 1
        runner.finalize(exit_code=exit_code, status="crashed")

    atexit.register(at_exit)


def run_review_loop_sync(
    cfg: RunConfig,
    paths: RunPaths,
    *,
    agent_type: str = "review-loop",
    install_signal_handlers: bool = True,
) -> int:
    """Spin up the runner synchronously and return the exit code.

    This is the shared entry point used by ``ai run`` (detached child),
    ``ai run --foreground``, and the legacy ``review-loop.py`` wrapper.
    Signal handlers are installed by default so an operator's ``ai kill``
    actually transitions ``meta.json`` to ``killed``; tests that don't
    want process-wide handlers installed can pass ``False``."""
    runner = ReviewLoopRunner(cfg, paths, agent_type=agent_type)
    if install_signal_handlers:
        _install_signal_handlers(runner)
    return asyncio.run(runner.run())


# ``python -m auto_iterator.runner <run_dir>`` — the spawn target of ai run.
def _main_from_run_dir(run_dir: str) -> int:
    p = Path(run_dir).resolve()
    runs_dir = p.parent
    paths = RunPaths(runs_dir=runs_dir, run_id=p.name)
    spec = json.loads(paths.spec.read_text(encoding="utf-8"))
    cfg = spec_to_cfg(spec)
    agent_type = spec.get("agent_type", "review-loop")
    # Re-stamp meta with our actual pid in case ``ai run`` seeded a
    # placeholder (or if spec existed but meta didn't).
    update_meta(paths, pid=os.getpid(), status="running")
    return run_review_loop_sync(cfg, paths, agent_type=agent_type)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m auto_iterator.runner <run_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(_main_from_run_dir(sys.argv[1]))
