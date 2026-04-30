"""Unit tests for :func:`auto_iterator.actions.default_run_config`.

The TUI's "new run" verb and any future env-driven caller need a
single helper that mirrors the CLI's backend / ``agent_cmd``
resolution order. These tests pin that contract directly so a
regression doesn't have to wait for the TUI smoke tests to surface
it."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_iterator import actions  # noqa: E402
from auto_iterator.backends import BACKENDS  # noqa: E402


def _clear_agent_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip ``AGENT_*`` env vars so each test starts from a known baseline."""
    for name in (
        "AGENT_BACKEND", "AGENT_CMD",
        "AGENT_IMPL_BACKEND", "AGENT_FIX_BACKEND", "AGENT_REVIEWER_BACKEND",
        "AGENT_IMPL_CMD", "AGENT_FIX_CMD", "AGENT_REVIEWER_CMD",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_run_config_uses_cursor_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No env vars → cursor backend with its default model fingerprints."""
    _clear_agent_env(monkeypatch)
    cfg = actions.default_run_config(task="t", workspace="/tmp/ws")
    be = BACKENDS["cursor"]
    assert cfg.backend == "cursor"
    assert cfg.agent_cmd == be.default_cmd
    assert cfg.impl_model == be.default_impl_model
    assert cfg.fix_model == be.default_fix_model
    assert cfg.reviewer_model == be.default_reviewer_model


def test_default_run_config_respects_agent_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``$AGENT_BACKEND`` selects the backend, mirroring ``_make_cfg_from_args``."""
    _clear_agent_env(monkeypatch)
    monkeypatch.setenv("AGENT_BACKEND", "claude-code")
    cfg = actions.default_run_config(task="t", workspace="/tmp/ws")
    be = BACKENDS["claude-code"]
    assert cfg.backend == "claude-code"
    # Model fingerprints come from the resolved backend, not cursor.
    assert cfg.impl_model == be.default_impl_model
    assert cfg.agent_cmd == be.default_cmd


def test_default_run_config_respects_agent_cmd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``$AGENT_CMD`` overrides the backend's default binary name."""
    _clear_agent_env(monkeypatch)
    monkeypatch.setenv("AGENT_CMD", "/opt/bin/my-agent")
    cfg = actions.default_run_config(task="t", workspace="/tmp/ws")
    assert cfg.agent_cmd == "/opt/bin/my-agent"
    # Backend defaults still come from cursor (no $AGENT_BACKEND).
    assert cfg.backend == "cursor"


def test_default_run_config_explicit_args_beat_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit kwargs trump env vars (matches ``args.backend or os.environ.get``)."""
    monkeypatch.setenv("AGENT_BACKEND", "cursor")
    monkeypatch.setenv("AGENT_CMD", "ignored")
    cfg = actions.default_run_config(
        task="t",
        workspace="/tmp/ws",
        backend="codex",
        agent_cmd="explicit-bin",
    )
    assert cfg.backend == "codex"
    assert cfg.agent_cmd == "explicit-bin"


def test_default_run_config_unknown_backend_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bad backend names surface as ValueError so callers can show a toast."""
    _clear_agent_env(monkeypatch)
    monkeypatch.setenv("AGENT_BACKEND", "no-such-backend")
    with pytest.raises(ValueError, match="unknown backend"):
        actions.default_run_config(task="t", workspace="/tmp/ws")


def test_default_run_config_validates_max_outer_inner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helper runs ``RunConfig.validate`` so callers don't have to."""
    _clear_agent_env(monkeypatch)
    with pytest.raises(ValueError, match="max-outer"):
        actions.default_run_config(
            task="t", workspace="/tmp/ws", max_outer=0,
        )


# ── Mixed-backend resolution (TUI ``n`` verb path) ─────────────────────────


def test_default_run_config_kwargs_select_mixed_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit per-phase kwargs route impl/fix vs reviewer to different CLIs.

    This is the canonical "Claude Code impl/fix + Codex reviewer" mix,
    plumbed through the TUI helper for callers that already know the
    desired layout."""
    _clear_agent_env(monkeypatch)
    cfg = actions.default_run_config(
        task="t", workspace="/tmp/ws",
        backend="claude-code",
        reviewer_backend="codex",
    )
    assert cfg.backend == "claude-code"
    assert cfg.backend_for("impl") == "claude-code"
    assert cfg.backend_for("fix") == "claude-code"
    assert cfg.backend_for("reviewer") == "codex"
    # Reviewer cmd defaults to codex's default_cmd; impl/fix follow claude.
    assert cfg.agent_cmd_for("reviewer") == BACKENDS["codex"].default_cmd
    assert cfg.agent_cmd_for("impl") == BACKENDS["claude-code"].default_cmd
    # Reviewer model fingerprint comes from codex, not claude.
    assert cfg.reviewer_model == BACKENDS["codex"].default_reviewer_model
    assert cfg.impl_model == BACKENDS["claude-code"].default_impl_model
    assert cfg.has_mixed_backends is True


def test_default_run_config_env_vars_select_mixed_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-phase env vars trigger a mixed setup from the TUI's ``n`` verb.

    The TUI has no argparse namespace; the only way to get a mixed
    Claude/Codex run from pressing ``n`` is via the environment. This
    test pins that contract directly."""
    _clear_agent_env(monkeypatch)
    monkeypatch.setenv("AGENT_BACKEND", "claude-code")
    monkeypatch.setenv("AGENT_REVIEWER_BACKEND", "codex")
    cfg = actions.default_run_config(task="t", workspace="/tmp/ws")
    assert cfg.backend == "claude-code"
    assert cfg.backend_for("reviewer") == "codex"
    assert cfg.backend_for("impl") == "claude-code"
    assert cfg.agent_cmd_for("reviewer") == BACKENDS["codex"].default_cmd
    assert cfg.reviewer_model == BACKENDS["codex"].default_reviewer_model
    assert cfg.has_mixed_backends is True


def test_default_run_config_phase_cmd_env_var_without_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``AGENT_REVIEWER_CMD`` alone overrides the binary, keeps backend."""
    _clear_agent_env(monkeypatch)
    monkeypatch.setenv("AGENT_BACKEND", "claude-code")
    monkeypatch.setenv("AGENT_REVIEWER_CMD", "/opt/bin/claude-reviewer")
    cfg = actions.default_run_config(task="t", workspace="/tmp/ws")
    assert cfg.backend_for("reviewer") == "claude-code"
    assert cfg.agent_cmd_for("reviewer") == "/opt/bin/claude-reviewer"
    assert cfg.has_mixed_backends is True  # cmd diverges from global


def test_default_run_config_kwargs_beat_phase_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit kwargs trump per-phase env vars, mirroring the global flag's pattern."""
    _clear_agent_env(monkeypatch)
    monkeypatch.setenv("AGENT_REVIEWER_BACKEND", "claude-code")
    cfg = actions.default_run_config(
        task="t", workspace="/tmp/ws",
        backend="claude-code",
        reviewer_backend="codex",
    )
    assert cfg.backend_for("reviewer") == "codex"


def test_default_run_config_unknown_phase_backend_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bad per-phase backend name surfaces as ValueError."""
    _clear_agent_env(monkeypatch)
    monkeypatch.setenv("AGENT_REVIEWER_BACKEND", "no-such-backend")
    with pytest.raises(ValueError, match="reviewer-backend"):
        actions.default_run_config(task="t", workspace="/tmp/ws")


def test_default_run_config_redundant_phase_backend_keeps_global_cmd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A redundant per-phase backend equal to the global must not bypass ``agent_cmd``.

    Regression: previously
    ``--reviewer-backend claude-code`` matching ``--backend claude-code``
    would normalize ``reviewer_backend`` to ``None`` but leave
    ``reviewer_agent_cmd`` populated with ``BACKENDS["claude-code"].default_cmd``
    — silently overwriting a custom global ``agent_cmd`` like
    ``/tmp/custom-claude``. The phase must inherit the global cmd
    unless an explicit per-phase cmd was given."""
    _clear_agent_env(monkeypatch)
    cfg = actions.default_run_config(
        task="t", workspace="/tmp/ws",
        backend="claude-code",
        agent_cmd="/tmp/custom-claude",
        reviewer_backend="claude-code",
    )
    assert cfg.reviewer_backend is None
    assert cfg.reviewer_agent_cmd is None
    assert cfg.agent_cmd_for("reviewer") == "/tmp/custom-claude"
    assert cfg.has_mixed_backends is False


def test_default_run_config_redundant_phase_backend_via_env_keeps_global_cmd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same regression via ``AGENT_REVIEWER_BACKEND`` env var (TUI path)."""
    _clear_agent_env(monkeypatch)
    monkeypatch.setenv("AGENT_BACKEND", "claude-code")
    monkeypatch.setenv("AGENT_CMD", "/tmp/custom-claude")
    monkeypatch.setenv("AGENT_REVIEWER_BACKEND", "claude-code")
    cfg = actions.default_run_config(task="t", workspace="/tmp/ws")
    assert cfg.reviewer_backend is None
    assert cfg.reviewer_agent_cmd is None
    assert cfg.agent_cmd_for("reviewer") == "/tmp/custom-claude"
    assert cfg.has_mixed_backends is False


def test_default_run_config_redundant_phase_backend_with_explicit_phase_cmd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit per-phase cmd is preserved even when the backend matches the global.

    The fix must not regress the "same backend, different binary path"
    use case: e.g. impl/fix on stable claude, reviewer on a beta build."""
    _clear_agent_env(monkeypatch)
    cfg = actions.default_run_config(
        task="t", workspace="/tmp/ws",
        backend="claude-code",
        agent_cmd="/tmp/stable-claude",
        reviewer_backend="claude-code",
        reviewer_agent_cmd="/tmp/beta-claude",
    )
    # Backend is collapsed to None (same as global) but the explicit
    # phase cmd survives so the reviewer hits the beta binary.
    assert cfg.reviewer_backend is None
    assert cfg.reviewer_agent_cmd == "/tmp/beta-claude"
    assert cfg.agent_cmd_for("reviewer") == "/tmp/beta-claude"
    assert cfg.has_mixed_backends is True
