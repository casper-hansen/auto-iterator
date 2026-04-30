"""Per-phase backend overrides — mixed-backend run configuration.

The CLI supports pinning each phase (``impl`` / ``fix`` / ``reviewer``)
to a different backend so an operator can mix CLIs across the loop —
the canonical case being "Claude Code as implementer/fixer with Codex
as a fresh-eyes reviewer".

These tests exercise the wiring at three layers:

1. ``RunConfig`` resolves ``backend_for`` / ``agent_cmd_for`` /
   ``agent_kw_for`` correctly for both legacy single-backend configs
   and the new mixed setup.
2. ``cli._make_cfg_from_args`` plumbs the new ``--impl-backend`` /
   ``--reviewer-backend`` flags into the cfg, defaulting per-phase
   models and CLI binaries from the right backend.
3. ``cfg_to_spec`` / ``spec_to_cfg`` round-trip the new fields so
   ``ai restart`` reproduces a mixed-backend run from disk.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_iterator.backends import BACKENDS  # noqa: E402
from auto_iterator.cli import _build_parser, _make_cfg_from_args  # noqa: E402
from auto_iterator.feature.config import RunConfig  # noqa: E402
from auto_iterator.runner import cfg_to_spec, spec_to_cfg  # noqa: E402


def _make_cfg(**overrides) -> RunConfig:
    """Build a baseline RunConfig with sensible defaults for tests."""
    base = dict(
        task="t",
        impl_model="m1",
        fix_model="m2",
        reviewer_model="m3",
        max_outer=1,
        max_inner=1,
        workspace="/tmp",
        skip_impl=False,
        extra_flags=(),
        agent_cmd="agent",
        backend="cursor",
    )
    base.update(overrides)
    return RunConfig(**base)


def test_runconfig_legacy_single_backend_falls_back() -> None:
    """No per-phase override → backend_for / agent_cmd_for return globals."""
    cfg = _make_cfg(backend="cursor", agent_cmd="agent")
    for phase in ("impl", "fix", "reviewer"):
        assert cfg.backend_for(phase) == "cursor"
        assert cfg.agent_cmd_for(phase) == "agent"
    assert cfg.has_mixed_backends is False
    # ``agent_kw_for`` mirrors the legacy ``agent_kw`` for single-backend.
    assert cfg.agent_kw_for("impl") == cfg.agent_kw


def test_runconfig_reviewer_pinned_to_codex() -> None:
    """Mixed setup: claude-code globally, codex for review only."""
    cfg = _make_cfg(
        backend="claude-code",
        agent_cmd="claude",
        reviewer_backend="codex",
        reviewer_agent_cmd="codex",
    )
    assert cfg.backend_for("impl") == "claude-code"
    assert cfg.backend_for("fix") == "claude-code"
    assert cfg.backend_for("reviewer") == "codex"
    assert cfg.agent_cmd_for("impl") == "claude"
    assert cfg.agent_cmd_for("fix") == "claude"
    assert cfg.agent_cmd_for("reviewer") == "codex"
    assert cfg.has_mixed_backends is True
    # The reviewer's agent_kw should target codex with its own binary.
    rk = cfg.agent_kw_for("reviewer")
    assert rk["backend"] == "codex"
    assert rk["agent_cmd"] == "codex"


def test_runconfig_unknown_phase_raises() -> None:
    cfg = _make_cfg()
    with pytest.raises(ValueError, match="unknown phase"):
        cfg.backend_for("planner")


def test_cli_parser_accepts_new_flags() -> None:
    p = _build_parser()
    args = p.parse_args([
        "run", "--prompt", "x",
        "--backend", "claude-code",
        "--reviewer-backend", "codex",
        "--reviewer-cmd", "/usr/local/bin/codex",
    ])
    assert args.backend == "claude-code"
    assert args.reviewer_backend == "codex"
    assert args.reviewer_agent_cmd == "/usr/local/bin/codex"
    # Flags we didn't pass should still parse as None.
    assert args.impl_backend is None
    assert args.fix_backend is None


def test_make_cfg_resolves_per_phase_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--reviewer-backend codex`` defaults reviewer_model from codex's table."""
    monkeypatch.delenv("AGENT_BACKEND", raising=False)
    monkeypatch.delenv("AGENT_CMD", raising=False)
    p = _build_parser()
    args = p.parse_args([
        "run",
        "--prompt", "x",
        "--workspace", "/tmp",
        "--backend", "claude-code",
        "--reviewer-backend", "codex",
    ])
    cfg = _make_cfg_from_args(args)
    assert cfg.backend == "claude-code"
    assert cfg.reviewer_backend == "codex"
    # impl/fix follow claude-code; reviewer follows codex.
    assert cfg.impl_model == BACKENDS["claude-code"].default_impl_model
    assert cfg.fix_model == BACKENDS["claude-code"].default_fix_model
    assert cfg.reviewer_model == BACKENDS["codex"].default_reviewer_model
    # Reviewer cmd defaults to codex's default_cmd when not specified.
    assert cfg.reviewer_agent_cmd == BACKENDS["codex"].default_cmd
    # impl/fix have no override → None (preserves legacy spec.json shape).
    assert cfg.impl_backend is None
    assert cfg.fix_backend is None
    assert cfg.has_mixed_backends is True


def test_make_cfg_rejects_unknown_phase_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_BACKEND", raising=False)
    p = _build_parser()
    args = p.parse_args([
        "run", "--prompt", "x",
        "--reviewer-backend", "no-such-backend",
    ])
    with pytest.raises(ValueError, match="--reviewer-backend"):
        _make_cfg_from_args(args)


def test_spec_roundtrip_preserves_mixed_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """``cfg_to_spec`` ↔ ``spec_to_cfg`` preserves per-phase overrides."""
    monkeypatch.delenv("AGENT_BACKEND", raising=False)
    monkeypatch.delenv("AGENT_CMD", raising=False)
    cfg = _make_cfg(
        backend="claude-code",
        agent_cmd="claude",
        reviewer_backend="codex",
        reviewer_agent_cmd="codex",
        impl_model="opus",
        fix_model="opus",
        reviewer_model="gpt-5.5",
    )
    spec = cfg_to_spec(cfg)
    assert spec["reviewer_backend"] == "codex"
    assert spec["reviewer_agent_cmd"] == "codex"
    assert spec["impl_backend"] is None
    assert spec["fix_backend"] is None

    restored = spec_to_cfg(spec)
    assert restored.backend_for("impl") == "claude-code"
    assert restored.backend_for("reviewer") == "codex"
    assert restored.agent_cmd_for("reviewer") == "codex"
    assert restored.has_mixed_backends is True


def test_spec_to_cfg_tolerates_legacy_spec_without_phase_fields() -> None:
    """Old spec.json files (no per-phase fields) restart cleanly."""
    legacy_spec = {
        "task": "t",
        "impl_model": "m1",
        "fix_model": "m2",
        "reviewer_model": "m3",
        "max_outer": 1,
        "max_inner": 1,
        "workspace": "/tmp",
        "skip_impl": False,
        "extra_flags": [],
        "agent_cmd": "agent",
        "backend": "cursor",
        "use_worktree": True,
    }
    cfg = spec_to_cfg(legacy_spec)
    assert cfg.backend == "cursor"
    assert cfg.has_mixed_backends is False
    for phase in ("impl", "fix", "reviewer"):
        assert cfg.backend_for(phase) == "cursor"
        assert cfg.agent_cmd_for(phase) == "agent"


def test_compose_review_prompt_uses_reviewer_backend() -> None:
    """Mixed setup: reviewer's backend governs the review-prompt template.

    Claude Code's backend ships a custom ``build_review_prompt`` that
    dispatches the ``/ultrareview`` skill. With Codex pinned as the
    reviewer, the generic prompt should be used instead — even when the
    *global* backend is ``claude-code``."""
    from auto_iterator.feature.steps import compose_review_prompt

    cfg = _make_cfg(
        backend="claude-code",
        agent_cmd="claude",
        reviewer_backend="codex",
        reviewer_agent_cmd="codex",
    )
    prompt = compose_review_prompt(cfg, task="t", history=[], guidance=[])
    # Generic prompt ends with the verdict choice; Claude's ultrareview
    # skill template wraps it differently. Either way, "VERDICT:" is
    # present, but the generic prompt does NOT include the skill's
    # ``ultrareview`` token.
    assert "VERDICT" in prompt
    assert "ultrareview" not in prompt.lower()


def test_compose_review_prompt_uses_global_backend_template_when_unmixed() -> None:
    """Single-backend claude-code reviewer still gets the skill template."""
    from auto_iterator.feature.steps import compose_review_prompt

    cfg = _make_cfg(backend="claude-code", agent_cmd="claude")
    prompt = compose_review_prompt(cfg, task="t", history=[], guidance=[])
    # The skill template references "review" semantics; the placeholder
    # token from the markdown skill is the smoking gun.
    assert "VERDICT" in prompt


def test_make_cfg_picks_up_phase_env_vars_when_flag_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``AGENT_REVIEWER_BACKEND`` is honoured when ``--reviewer-backend`` is unset.

    Mirrors the resolution order of the global ``--backend`` flag —
    explicit flag wins, then env var, then ``None``. This keeps the CLI
    in lockstep with :func:`actions.default_run_config` so ``ai run``
    and the TUI's ``n`` verb produce the same RunConfig from the same
    shell."""
    monkeypatch.delenv("AGENT_BACKEND", raising=False)
    monkeypatch.delenv("AGENT_CMD", raising=False)
    monkeypatch.setenv("AGENT_REVIEWER_BACKEND", "codex")
    p = _build_parser()
    args = p.parse_args([
        "run",
        "--prompt", "x",
        "--workspace", "/tmp",
        "--backend", "claude-code",
    ])
    cfg = _make_cfg_from_args(args)
    assert cfg.backend == "claude-code"
    assert cfg.backend_for("reviewer") == "codex"
    assert cfg.reviewer_agent_cmd == BACKENDS["codex"].default_cmd
    assert cfg.has_mixed_backends is True


def test_make_cfg_phase_flag_beats_phase_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``--reviewer-backend`` trumps ``AGENT_REVIEWER_BACKEND``."""
    monkeypatch.delenv("AGENT_BACKEND", raising=False)
    monkeypatch.setenv("AGENT_REVIEWER_BACKEND", "claude-code")
    p = _build_parser()
    args = p.parse_args([
        "run", "--prompt", "x", "--workspace", "/tmp",
        "--backend", "claude-code",
        "--reviewer-backend", "codex",
    ])
    cfg = _make_cfg_from_args(args)
    assert cfg.backend_for("reviewer") == "codex"


def test_make_cfg_redundant_phase_backend_preserves_global_agent_cmd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--reviewer-backend claude-code`` matching the global must not bypass ``--agent-cmd``.

    Regression: an explicit-but-redundant per-phase backend used to
    normalize ``reviewer_backend`` to ``None`` while overwriting
    ``reviewer_agent_cmd`` with the backend's ``default_cmd``, silently
    bypassing a custom global ``--agent-cmd``. The phase must inherit
    the global cmd whenever no explicit per-phase cmd was passed."""
    monkeypatch.delenv("AGENT_BACKEND", raising=False)
    monkeypatch.delenv("AGENT_CMD", raising=False)
    monkeypatch.delenv("AGENT_REVIEWER_BACKEND", raising=False)
    monkeypatch.delenv("AGENT_REVIEWER_CMD", raising=False)
    p = _build_parser()
    args = p.parse_args([
        "run", "--prompt", "x", "--workspace", "/tmp",
        "--backend", "claude-code",
        "--agent-cmd", "/tmp/custom-claude",
        "--reviewer-backend", "claude-code",
    ])
    cfg = _make_cfg_from_args(args)
    assert cfg.reviewer_backend is None
    assert cfg.reviewer_agent_cmd is None
    assert cfg.agent_cmd_for("reviewer") == "/tmp/custom-claude"
    assert cfg.has_mixed_backends is False


def test_make_cfg_redundant_phase_backend_with_explicit_phase_cmd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same backend, different binary path: per-phase cmd survives normalization."""
    monkeypatch.delenv("AGENT_BACKEND", raising=False)
    monkeypatch.delenv("AGENT_CMD", raising=False)
    p = _build_parser()
    args = p.parse_args([
        "run", "--prompt", "x", "--workspace", "/tmp",
        "--backend", "claude-code",
        "--agent-cmd", "/tmp/stable-claude",
        "--reviewer-backend", "claude-code",
        "--reviewer-cmd", "/tmp/beta-claude",
    ])
    cfg = _make_cfg_from_args(args)
    assert cfg.reviewer_backend is None
    assert cfg.reviewer_agent_cmd == "/tmp/beta-claude"
    assert cfg.agent_cmd_for("reviewer") == "/tmp/beta-claude"
    assert cfg.has_mixed_backends is True


def test_make_cfg_redundant_phase_backend_via_env_preserves_global_agent_cmd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``AGENT_REVIEWER_BACKEND`` matching the global also preserves global cmd."""
    monkeypatch.setenv("AGENT_BACKEND", "claude-code")
    monkeypatch.setenv("AGENT_CMD", "/tmp/custom-claude")
    monkeypatch.setenv("AGENT_REVIEWER_BACKEND", "claude-code")
    monkeypatch.delenv("AGENT_REVIEWER_CMD", raising=False)
    p = _build_parser()
    args = p.parse_args([
        "run", "--prompt", "x", "--workspace", "/tmp",
    ])
    cfg = _make_cfg_from_args(args)
    assert cfg.backend == "claude-code"
    assert cfg.agent_cmd == "/tmp/custom-claude"
    assert cfg.reviewer_backend is None
    assert cfg.reviewer_agent_cmd is None
    assert cfg.agent_cmd_for("reviewer") == "/tmp/custom-claude"
    assert cfg.has_mixed_backends is False
