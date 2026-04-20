"""ExperimentConfig dataclass — single object for all experiment loop configuration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

DEFAULT_EXPERIMENTER_MODEL = "claude-opus-4-7-thinking-max"
DEFAULT_ADJUSTER_MODEL = "claude-opus-4-7-thinking-max"
DEFAULT_ANALYST_MODEL = "gpt-5.4-xhigh"


@dataclass(frozen=True)
class ExperimentConfig:
    hypothesis: str
    success_criteria: str
    baseline_config: str
    experiment_config: str
    training_cmd: str
    experimenter_model: str
    adjuster_model: str
    analyst_model: str
    max_iterations: int
    workspace: str
    skip_baseline: bool
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
        if self.max_iterations < 1:
            return f"--max-iterations must be >= 1 (got '{self.max_iterations}')"
        if not self.hypothesis:
            return "hypothesis is required"
        if not self.success_criteria:
            return "success criteria are required"
        return None

    def banner_items(self) -> dict[str, object]:
        """Ordered dict of label→value pairs for the startup banner."""
        return {
            "hypothesis": self.hypothesis,
            "success_criteria": self.success_criteria,
            "baseline_config": self.baseline_config,
            "experiment_config": self.experiment_config,
            "training_cmd": self.training_cmd,
            "experimenter": self.experimenter_model,
            "adjuster": self.adjuster_model,
            "analyst": self.analyst_model,
            "max_iterations": self.max_iterations,
            "workspace": self.workspace,
            "skip_baseline": self.skip_baseline,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
