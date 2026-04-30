"""RunConfig dataclass — single object for all loop configuration.

Model defaults are owned by each backend (see
``auto_iterator.backends.cursor.CursorBackend`` /
``auto_iterator.backends.claude_code.ClaudeCodeBackend``), not by this
module, because each CLI speaks to a different set of model names.

Per-phase backends
------------------
The loop has three roles — ``impl`` (initial implementation), ``fix``
(post-review fix-up), and ``reviewer`` (the diff inspector). Each role
can be pinned to a different CLI backend so an operator can mix and
match: e.g. Claude Code as the implementer/fixer with Codex as a
fresh-eyes reviewer.

The optional ``{impl,fix,reviewer}_backend`` and matching
``{impl,fix,reviewer}_agent_cmd`` fields override the global ``backend``
/ ``agent_cmd`` for the corresponding role. They default to ``None``,
which means "fall back to the global backend / cmd" — keeping single-
backend setups byte-identical to the pre-mixed-backend behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


# Phase names used by ``backend_for`` / ``agent_cmd_for`` /
# ``agent_kw_for``. Kept as a tuple of literals so callers can lint
# against typos and downstream code can iterate them.
_PHASES = ("impl", "fix", "reviewer")


@dataclass(frozen=True)
class RunConfig:
    task: str
    impl_model: str
    fix_model: str
    reviewer_model: str
    max_outer: int
    max_inner: int
    workspace: str
    skip_impl: bool
    extra_flags: tuple[str, ...]
    agent_cmd: str = "agent"
    backend: str = "cursor"
    use_worktree: bool = True
    # Set by the runner once the worktree is created; agents are launched
    # with this as their cwd. ``workspace`` keeps pointing at the source
    # workspace so spec.json is restartable from the user's vantage point.
    worktree_path: str | None = None

    # ── Per-phase backend / cmd overrides ─────────────────────────────
    # ``None`` means "use the global ``backend`` / ``agent_cmd``"; any
    # non-None value pins that phase to a different CLI. The CLI / TUI
    # are responsible for validating the chosen backend names against
    # ``BACKENDS`` so callers here don't have to import the registry.
    impl_backend: str | None = None
    fix_backend: str | None = None
    reviewer_backend: str | None = None
    impl_agent_cmd: str | None = None
    fix_agent_cmd: str | None = None
    reviewer_agent_cmd: str | None = None

    @property
    def effective_workspace(self) -> str:
        """The path agents actually run inside.

        Falls back to ``workspace`` whenever no worktree was created — for
        ``--no-worktree`` runs, non-git workspaces, or runs that haven't
        finished bootstrapping yet."""
        return self.worktree_path or self.workspace

    # ── Per-phase resolution helpers ──────────────────────────────────

    def backend_for(self, phase: str) -> str:
        """Return the backend name for *phase* (``impl``/``fix``/``reviewer``).

        Falls back to the global ``backend`` whenever the phase override
        is ``None`` — which is the common single-backend case."""
        if phase not in _PHASES:
            raise ValueError(f"unknown phase '{phase}'; expected one of {_PHASES}")
        override = {
            "impl": self.impl_backend,
            "fix": self.fix_backend,
            "reviewer": self.reviewer_backend,
        }[phase]
        return override or self.backend

    def agent_cmd_for(self, phase: str) -> str:
        """Return the CLI binary name to invoke for *phase*.

        Falls back to the global ``agent_cmd`` whenever the phase
        override is ``None``. A per-phase backend without an explicit
        cmd is the CLI's responsibility to resolve (it sets the default
        cmd from the chosen backend at parse time so spec.json is a
        complete snapshot)."""
        if phase not in _PHASES:
            raise ValueError(f"unknown phase '{phase}'; expected one of {_PHASES}")
        override = {
            "impl": self.impl_agent_cmd,
            "fix": self.fix_agent_cmd,
            "reviewer": self.reviewer_agent_cmd,
        }[phase]
        return override or self.agent_cmd

    def agent_kw_for(self, phase: str) -> dict:
        """Keyword arguments forwarded to ``run_agent`` for *phase*.

        Replaces the legacy ``agent_kw`` property at every loop call
        site so a phase pinned to a different backend (e.g. Codex as
        the reviewer with Claude Code as the implementer/fixer) gets
        its own CLI binary and stream-json adapter."""
        return dict(
            workspace=self.effective_workspace,
            agent_cmd=self.agent_cmd_for(phase),
            extra_flags=list(self.extra_flags),
            backend=self.backend_for(phase),
        )

    @property
    def agent_kw(self) -> dict:
        """Legacy alias — keyword arguments under the global backend.

        New call sites should use :meth:`agent_kw_for` so the right
        backend is chosen per phase. This property is kept for back-
        compat with single-backend callers (notably the experiment
        loop, which still uses one backend across all roles)."""
        return dict(
            workspace=self.effective_workspace,
            agent_cmd=self.agent_cmd,
            extra_flags=list(self.extra_flags),
            backend=self.backend,
        )

    @property
    def has_mixed_backends(self) -> bool:
        """True iff any phase pins a backend different from the global one."""
        return any(
            self.backend_for(p) != self.backend for p in _PHASES
        ) or any(
            self.agent_cmd_for(p) != self.agent_cmd for p in _PHASES
        )

    def validate(self) -> str | None:
        """Return an error message if invalid, *None* if OK."""
        if self.max_outer < 1:
            return f"--max-outer must be a positive integer (got '{self.max_outer}')"
        if self.max_inner < 1:
            return f"--max-inner must be a positive integer (got '{self.max_inner}')"
        return None

    def banner_items(self) -> dict[str, object]:
        """Ordered dict of label→value pairs for the startup banner."""
        items: dict[str, object] = {
            "task": self.task,
            "impl_model": self.impl_model,
            "fix_model": self.fix_model,
            "reviewer_model": self.reviewer_model,
            "max_outer": self.max_outer,
            "max_inner": self.max_inner,
            "workspace": self.workspace,
            "skip_impl": self.skip_impl,
            "backend": self.backend,
            "agent_cmd": self.agent_cmd,
            "use_worktree": self.use_worktree,
        }
        # Surface per-phase overrides only when they differ from the
        # global backend — otherwise the banner stays single-line for
        # the common case.
        if self.has_mixed_backends:
            items["impl_backend"] = self.backend_for("impl")
            items["impl_agent_cmd"] = self.agent_cmd_for("impl")
            items["fix_backend"] = self.backend_for("fix")
            items["fix_agent_cmd"] = self.agent_cmd_for("fix")
            items["reviewer_backend"] = self.backend_for("reviewer")
            items["reviewer_agent_cmd"] = self.agent_cmd_for("reviewer")
        if self.worktree_path:
            items["worktree_path"] = self.worktree_path
        items["started_at"] = datetime.now(timezone.utc).isoformat()
        return items
