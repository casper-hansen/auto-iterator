"""ANSI color constants, auto-disabled when stdout is not a TTY."""

from __future__ import annotations

import os
import sys

_USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ

BOLD = "\033[1m" if _USE_COLOR else ""
DIM = "\033[2m" if _USE_COLOR else ""
RED = "\033[0;31m" if _USE_COLOR else ""
GREEN = "\033[0;32m" if _USE_COLOR else ""
YELLOW = "\033[0;33m" if _USE_COLOR else ""
BLUE = "\033[0;34m" if _USE_COLOR else ""
CYAN = "\033[0;36m" if _USE_COLOR else ""
MAGENTA = "\033[0;35m" if _USE_COLOR else ""
NC = "\033[0m" if _USE_COLOR else ""
