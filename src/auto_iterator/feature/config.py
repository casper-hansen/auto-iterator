"""RunConfig dataclass — single object for all loop configuration.

Model defaults are owned by each backend (see
``auto_iterator.backends.cursor.CursorBackend`` /
``auto_iterator.backends.claude_code.ClaudeCodeBackend``), not by this
module, because each CLI speaks to a different set of model names.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


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

    @property
    def effective_workspace(self) -> str:
        """The path agents actually run inside.

        Falls back to ``workspace`` whenever no worktree was created — for
        ``--no-worktree`` runs, non-git workspaces, or runs that haven't
        finished bootstrapping yet."""
        return self.worktree_path or self.workspace

    @property
    def agent_kw(self) -> dict:
        """Keyword arguments forwarded to every ``run_agent`` call."""
        return dict(
            workspace=self.effective_workspace,
            agent_cmd=self.agent_cmd,
            extra_flags=list(self.extra_flags),
            backend=self.backend,
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
        if self.worktree_path:
            items["worktree_path"] = self.worktree_path
        items["started_at"] = datetime.now(timezone.utc).isoformat()
        return items
