"""RunConfig dataclass — single object for all loop configuration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

DEFAULT_IMPL_MODEL = "claude-4.6-opus-max"
DEFAULT_FIX_MODEL = "claude-4.6-opus-max"
DEFAULT_REVIEWER_MODEL = "gpt-5.4-xhigh"


@dataclass(frozen=True)
class RunConfig:
    prompt: str
    context: str
    impl_model: str
    fix_model: str
    reviewer_model: str
    max_outer: int
    max_inner: int
    workspace: str
    skip_impl: bool
    extra_flags: tuple[str, ...]
    agent_cmd: str = "agent"

    @property
    def agent_kw(self) -> dict:
        """Keyword arguments forwarded to every ``run_agent`` call."""
        return dict(
            workspace=self.workspace,
            agent_cmd=self.agent_cmd,
            extra_flags=list(self.extra_flags),
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
        return {
            "prompt": self.prompt,
            "context": self.context,
            "impl_model": self.impl_model,
            "fix_model": self.fix_model,
            "reviewer_model": self.reviewer_model,
            "max_outer": self.max_outer,
            "max_inner": self.max_inner,
            "workspace": self.workspace,
            "skip_impl": self.skip_impl,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
