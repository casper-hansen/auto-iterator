"""OutputFormatter — streams agent output, dimming reasoning blocks."""

from __future__ import annotations

import re

from .colors import DIM, NC
from .logging import _ts

_THINKING_OPEN = re.compile(r"<(?:antml:)?thinking>")
_THINKING_CLOSE = re.compile(r"</(?:antml:)?thinking>")

# CSI: ESC [ <params> <intermediate> <final>, where each class uses its
# spec byte range. The digits/`;?` original covered standard SGR, but
# Claude Code emits private-parameter codes like \x1b[>4m and \x1b[<u
# (params in 0x30-0x3f including `<` `>` `=`), which the old charset
# missed and left visible as "[>4m[<u". The OSC branch (ESC ] ... BEL)
# is unchanged.
_ANSI_RE = re.compile(
    r"\x1b\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]|\x1b\].*?\x07"
)


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
                print(f"{ts} {self.tag}   {DIM}{after}{NC}", flush=True)
            return

        if _THINKING_CLOSE.search(line):
            before = _THINKING_CLOSE.sub("", line).strip()
            if before:
                print(f"{ts} {self.tag}   {DIM}{before}{NC}", flush=True)
            self._state = self.ASSISTANT
            return

        if self._state == self.THINKING:
            print(f"{ts} {self.tag}   {DIM}{line}{NC}", flush=True)
        else:
            print(f"{ts} {self.tag}   {line}", flush=True)

    def feed_tool(self, line: str) -> None:
        """Print a tool-call line, always dimmed to reduce visual noise."""
        ts = f"{DIM}{_ts()}{NC}"
        print(f"{ts} {self.tag}     {DIM}{line}{NC}", flush=True)

    def flush(self) -> None:
        self._state = self.ASSISTANT
