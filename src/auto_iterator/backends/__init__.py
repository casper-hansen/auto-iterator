"""Pluggable CLI backends for ``run_agent``.

Each backend knows how to (a) build a shell command that drives its CLI in
headless stream-json mode and (b) translate the stream-json events that
come back into the generic ``_StreamReader`` state the loop cares about
(text, tool start/stop, final ``result`` marker).
"""

from __future__ import annotations

from .claude_code import ClaudeCodeBackend
from .codex import CodexBackend
from .cursor import CursorBackend

BACKENDS = {
    "cursor": CursorBackend(),
    "claude-code": ClaudeCodeBackend(),
    "codex": CodexBackend(),
}


def get_backend(name: str):
    try:
        return BACKENDS[name]
    except KeyError:
        valid = ", ".join(sorted(BACKENDS))
        raise ValueError(
            f"Unknown agent backend '{name}'. Valid options: {valid}"
        ) from None


__all__ = [
    "BACKENDS",
    "ClaudeCodeBackend",
    "CodexBackend",
    "CursorBackend",
    "get_backend",
]
