"""Background heartbeat: the runner's liveness signal for ``ai ls``.

A tiny daemon thread touches ``<run_dir>/heartbeat`` every few seconds.
``ai ls`` reads the file's mtime to tell ``running`` apart from ``stuck``
(pid alive, heartbeat stale > 30s) and ``crashed`` (pid dead entirely).

We deliberately do *not* route this through ``EventLog`` — heartbeats are
noise that would dwarf the real event stream. They live as a side-band
signal that costs almost nothing to read and write."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from .run_dir import now_iso, touch


# Heartbeat cadence: 5s keeps the staleness window tight without burning
# measurable IO. 30s is the threshold used by ``ls`` to flip ``running`` →
# ``stuck`` (six missed beats), so callers can tune either in concert.
DEFAULT_INTERVAL = 5.0
STUCK_THRESHOLD_SECONDS = 30.0


class Heartbeat:
    """Start/stop a thread that pings the heartbeat file at a fixed cadence.

    Also drives the ``heartbeat_at`` field on ``meta.json`` (via the
    supplied writer callback) so an ``ai ls --json`` consumer that prefers
    ISO timestamps over filesystem mtime has a parallel source of truth."""

    def __init__(
        self,
        heartbeat_path: Path,
        on_beat=None,
        *,
        interval: float = DEFAULT_INTERVAL,
    ) -> None:
        self._path = heartbeat_path
        self._on_beat = on_beat
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                touch(self._path)
                if self._on_beat is not None:
                    self._on_beat(now_iso())
            except OSError:
                # Disk gone / run-dir deleted — silently back off; the
                # runner's own shutdown will handle diagnostics.
                pass
            self._stop.wait(self._interval)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="ai-heartbeat", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 2)
            self._thread = None

    def __enter__(self) -> "Heartbeat":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
