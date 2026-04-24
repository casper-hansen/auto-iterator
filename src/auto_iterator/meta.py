"""Reading/writing the small-but-hot ``meta.json`` file.

``meta.json`` is the cheap index record for each run: pid, workspace,
started_at, heartbeat_at, and a best-effort ``status``. ``ai ls`` reads
one of these per run, so keeping it tiny means ``ls`` stays cheap even
with hundreds of historical runs.

``status`` here is a *hint* — the authoritative status is computed at
read time by ``ls.py`` which reconciles pid liveness, heartbeat mtime,
and the event-log tail. We still write ``running`` / ``exited`` /
``killed`` on the hot path so single-run inspection tools don't need to
run the full reconciliation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .run_dir import RunPaths, atomic_write_json


def write_meta(paths: RunPaths, meta: dict[str, Any]) -> None:
    atomic_write_json(paths.meta, meta)


def read_meta(paths: RunPaths) -> Optional[dict[str, Any]]:
    try:
        return json.loads(paths.meta.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return None


def update_meta(paths: RunPaths, **fields: Any) -> dict[str, Any]:
    """Merge *fields* into ``meta.json`` and rewrite atomically.

    Idempotent: if meta doesn't exist yet, we seed it from ``fields``.
    Callers should always include ``run_id`` in the seed case."""
    current = read_meta(paths) or {}
    current.update(fields)
    write_meta(paths, current)
    return current
