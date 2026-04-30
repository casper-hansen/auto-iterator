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
    monkeypatch.delenv("AGENT_BACKEND", raising=False)
    monkeypatch.delenv("AGENT_CMD", raising=False)


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
