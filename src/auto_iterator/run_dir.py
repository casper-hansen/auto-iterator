"""Filesystem layout for a single auto-iterator run.

Every intent an operator can express — start, steer, inspect, kill — is a
file drop under a per-run directory. This module owns the paths, the id
scheme, and the atomic write primitives; higher-level modules (events,
control, ls) build on it.

Design notes
------------
* The runs-dir is a *per-user* root; the default is ``~/.auto-iterator/runs``
  but every entry point accepts ``--runs-dir`` so tests can substitute a
  tmpdir.
* Permissions are tight by default (``0700`` for dirs, ``0600`` for files)
  because prompts and context can contain sensitive material.
* Writes that matter (``meta.json``, ``state.json``, ``spec.json``,
  ``control/rewind.json``) go through :func:`atomic_write_json` which uses
  tmp + ``os.rename`` so a reader never sees a half-written file.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RUNS_DIR_ENV = "AUTO_ITERATOR_RUNS_DIR"
DEFAULT_RUNS_DIR = "~/.auto-iterator/runs"

# File layout constants (one place to rename them later).
META_FILE = "meta.json"
SPEC_FILE = "spec.json"
STATE_FILE = "state.json"
EVENTS_FILE = "events.jsonl"
CONTROL_APPLIED_FILE = "control-applied.jsonl"
HEARTBEAT_FILE = "heartbeat"
INDEX_FILE = "index.jsonl"
CONTROL_DIR = "control"
LOGS_DIR = "logs"
AGENT_LOG_FILE = "agent.log"

# Control file names (under ``<run_dir>/control/``).
CTL_GUIDANCE = "guidance.txt"
CTL_REWIND = "rewind.json"
CTL_PROMPT = "prompt.txt"
CTL_CONTEXT = "context.txt"
CTL_PAUSE = "pause"


def resolve_runs_dir(override: str | None) -> Path:
    """Pick the runs-dir: CLI override > env var > user default."""
    if override:
        root = Path(override).expanduser().resolve()
    else:
        env = os.environ.get(DEFAULT_RUNS_DIR_ENV, "").strip()
        root = Path(env or DEFAULT_RUNS_DIR).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    return root


def new_run_id() -> str:
    """Generate ``<UTC compact timestamp>-<6 hex>`` — sortable + unique."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{secrets.token_hex(3)}"


def now_iso() -> str:
    """ISO-8601 UTC timestamp with microseconds; matches what events carry."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RunPaths:
    """All the paths a runner or CLI subcommand needs, in one bundle."""

    runs_dir: Path
    run_id: str

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.run_id

    @property
    def meta(self) -> Path:
        return self.run_dir / META_FILE

    @property
    def spec(self) -> Path:
        return self.run_dir / SPEC_FILE

    @property
    def state(self) -> Path:
        return self.run_dir / STATE_FILE

    @property
    def events(self) -> Path:
        return self.run_dir / EVENTS_FILE

    @property
    def control_applied(self) -> Path:
        return self.run_dir / CONTROL_APPLIED_FILE

    @property
    def heartbeat(self) -> Path:
        return self.run_dir / HEARTBEAT_FILE

    @property
    def control_dir(self) -> Path:
        return self.run_dir / CONTROL_DIR

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / LOGS_DIR

    @property
    def agent_log(self) -> Path:
        return self.logs_dir / AGENT_LOG_FILE

    @property
    def index(self) -> Path:
        return self.runs_dir / INDEX_FILE

    def control_file(self, name: str) -> Path:
        return self.control_dir / name


def create_run_dir(runs_dir: Path, run_id: str) -> RunPaths:
    """Create the per-run directory tree with tight permissions.

    Directories are created ``0700``, enforced with ``os.chmod`` (to defeat
    any inherited umask). Callers write files into this tree through
    ``atomic_write_*`` which enforces ``0600`` on the final file.
    """
    paths = RunPaths(runs_dir=runs_dir, run_id=run_id)
    paths.run_dir.mkdir(parents=True, exist_ok=False)
    paths.control_dir.mkdir(exist_ok=True)
    paths.logs_dir.mkdir(exist_ok=True)
    for d in (paths.run_dir, paths.control_dir, paths.logs_dir):
        try:
            os.chmod(d, 0o700)
        except OSError:
            pass
    return paths


def atomic_write_bytes(path: Path, data: bytes, mode: int = 0o600) -> None:
    """Write *data* atomically: tmp file in the same dir, then ``os.rename``.

    Same-directory rename is atomic on POSIX, so readers only ever see the
    old contents or the new contents — never a truncated frame.
    """
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{secrets.token_hex(3)}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    os.rename(tmp, path)


def atomic_write_text(path: Path, text: str, mode: int = 0o600) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


def atomic_write_json(path: Path, obj: Any, mode: int = 0o600) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"),
        mode=mode,
    )


def append_jsonl(path: Path, obj: Any, mode: int = 0o600) -> None:
    """Append one JSON object as a line. Uses ``O_APPEND`` so concurrent
    appenders interleave safely (each ``write`` is atomic for lines that
    fit in ``PIPE_BUF``, which ours comfortably do)."""
    line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    fd = os.open(path, flags, mode)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)
    # A newly created file may have a broader umask mode; re-apply desired.
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def touch(path: Path, mode: int = 0o600) -> None:
    """Ensure *path* exists (create empty if missing) and bump mtime to now."""
    flags = os.O_WRONLY | os.O_CREAT
    fd = os.open(path, flags, mode)
    os.close(fd)
    os.utime(path, None)


def read_json(path: Path) -> Any:
    """Read a JSON file; raises FileNotFoundError / JSONDecodeError to caller."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_last_jsonl(path: Path, max_bytes: int = 65_536) -> dict | None:
    """Return the last complete JSON object in *path*, or ``None`` if empty.

    Reads only the tail (``max_bytes``) so this is cheap even for long event
    logs. If the tail does not contain a complete final line, ``None`` is
    returned rather than guessing."""
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size == 0:
        return None
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
            f.readline()  # drop partial leading line
        tail = f.read()
    if not tail:
        return None
    lines = [ln for ln in tail.splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1].decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None


def iter_run_dirs(runs_dir: Path):
    """Yield ``RunPaths`` for every per-run subdirectory of *runs_dir*.

    Ignores files (``index.jsonl``) and hidden dotfiles; nonexistent
    ``runs_dir`` returns an empty iterator."""
    if not runs_dir.exists():
        return
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        yield RunPaths(runs_dir=runs_dir, run_id=entry.name)


def pid_alive(pid: int) -> bool:
    """``kill -0`` check: is *pid* a running process this user can signal?"""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — still alive.
        return True
    except OSError:
        return False
    return True
