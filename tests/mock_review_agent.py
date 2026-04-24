#!/usr/bin/env python3
"""Deterministic mock of an agent CLI that the review-loop can drive.

Environment variables (all optional):

* ``MOCK_VERDICTS`` — comma-separated verdicts cycled through on each
  invocation. Each entry is one of ``APPROVED`` / ``CHANGES_NEEDED`` /
  ``SLEEP:<seconds>`` (emit text and block before exiting). Default
  ``APPROVED``.
* ``MOCK_CAPTURE_DIR`` — directory where each invocation drops a
  ``call-<N>.json`` file containing its argv + prompt. Tests read these
  to assert that operator guidance / prompt replacements reached the
  agent.
* ``MOCK_STATE_FILE`` — opaque marker file the mock creates on first
  invocation so subsequent ``--continue`` calls can distinguish "initial"
  from "resumed" sessions.

The mock speaks the Cursor ``stream-json`` dialect (which ``run_agent``
tolerates via the Cursor backend). It emits a single ``assistant`` text
message carrying the verdict line and, for review invocations, the
expected ``VERDICT: …`` tail the runner parses."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _emit_assistant_text(text: str) -> None:
    """Emit a single ``assistant`` stream-json event carrying *text*."""
    obj = {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": text}]},
        "model_call_id": f"mock-{os.getpid()}-{int(time.time() * 1000)}",
    }
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _emit_result(text: str) -> None:
    sys.stdout.write(json.dumps({
        "type": "result",
        "subtype": "success",
        "result": text,
    }) + "\n")
    sys.stdout.flush()


def _capture(prompt: str, argv: list[str]) -> None:
    cap_dir = os.environ.get("MOCK_CAPTURE_DIR")
    if not cap_dir:
        return
    p = Path(cap_dir)
    p.mkdir(parents=True, exist_ok=True)
    n = len(list(p.glob("call-*.json")))
    (p / f"call-{n:03d}.json").write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "argv": argv,
        "prompt": prompt,
    }, indent=2), encoding="utf-8")


def _next_verdict() -> str:
    """Pop the next verdict from the env var, advancing the index in place."""
    verdicts = os.environ.get("MOCK_VERDICTS", "APPROVED").split(",")
    idx_file = os.environ.get("MOCK_VERDICT_IDX_FILE")
    idx = 0
    if idx_file and Path(idx_file).exists():
        try:
            idx = int(Path(idx_file).read_text().strip() or "0")
        except ValueError:
            idx = 0
    verdict = verdicts[min(idx, len(verdicts) - 1)].strip()
    if idx_file:
        Path(idx_file).write_text(str(idx + 1), encoding="utf-8")
    return verdict


def main() -> int:
    # Accept any subset of the cursor-style flags we care about, collect
    # the positional prompt (the last non-flag argv element).
    argv = sys.argv[1:]
    # Prompt is the final positional; flags we know take values eat the
    # next arg so we just take the tail.
    prompt = argv[-1] if argv else ""
    _capture(prompt, argv)

    verdict = _next_verdict()
    if verdict.startswith("SLEEP:"):
        # SLEEP:<seconds>[:verdict] — useful for integration tests that
        # need a stable boundary to drop control files in.
        parts = verdict.split(":", 2)
        try:
            dur = float(parts[1])
        except (IndexError, ValueError):
            dur = 1.0
        tail_verdict = parts[2] if len(parts) >= 3 else "APPROVED"
        _emit_assistant_text("Starting review...")
        time.sleep(dur)
        _emit_assistant_text(f"Review complete.\nVERDICT: {tail_verdict}")
        _emit_result(f"VERDICT: {tail_verdict}")
        return 0

    # Plain verdict or arbitrary text
    if verdict in ("APPROVED", "CHANGES_NEEDED"):
        text = f"Reviewing diff...\nLooks {'good' if verdict == 'APPROVED' else 'needs work'}.\nVERDICT: {verdict}"
    else:
        text = verdict  # raw emission (used by impl/fix phases)

    _emit_assistant_text(text)
    _emit_result(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
