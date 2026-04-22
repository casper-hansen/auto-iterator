"""OutputFormatter — streams agent output, dimming reasoning blocks.

The formatter is invoked from the PTY reader thread inside
:class:`_StreamReader`. A broken stdout at that point must not crash
the reader — the caller still needs a final outcome to decide whether
to auto-resume — and it must not leak ``OSError`` up into unrelated
code paths. Every write here therefore routes through
:func:`iterator_loop.logging._safe_print`, which swaps ``fd 1`` to
``/dev/null`` on the first failure (see the module docstring on
``logging.py`` for the full rationale).
"""

from __future__ import annotations

import re

from .colors import DIM, NC
from .logging import _safe_print, _ts

_THINKING_OPEN = re.compile(r"<(?:antml:)?thinking>")
_THINKING_CLOSE = re.compile(r"</(?:antml:)?thinking>")
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\].*?\x07")


class OutputFormatter:
    """Streams agent output, dimming reasoning and printing messages plainly."""

    THINKING = "thinking"
    ASSISTANT = "assistant"

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self._state = self.ASSISTANT

    def feed(self, line: str) -> None:
        ts = f"{DIM}{_ts()}{NC}"

        if _THINKING_OPEN.search(line):
            self._state = self.THINKING
            after = _THINKING_OPEN.sub("", line).strip()
            if after:
                _safe_print(f"{ts} {self.tag}   {DIM}{after}{NC}", flush=True)
            return

        if _THINKING_CLOSE.search(line):
            before = _THINKING_CLOSE.sub("", line).strip()
            if before:
                _safe_print(f"{ts} {self.tag}   {DIM}{before}{NC}", flush=True)
            self._state = self.ASSISTANT
            return

        if self._state == self.THINKING:
            _safe_print(f"{ts} {self.tag}   {DIM}{line}{NC}", flush=True)
        else:
            _safe_print(f"{ts} {self.tag}   {line}", flush=True)

    def feed_tool(self, line: str) -> None:
        """Print a tool-call line, always dimmed to reduce visual noise."""
        ts = f"{DIM}{_ts()}{NC}"
        _safe_print(f"{ts} {self.tag}     {DIM}{line}{NC}", flush=True)

    def flush(self) -> None:
        self._state = self.ASSISTANT
