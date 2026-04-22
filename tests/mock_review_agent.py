#!/usr/bin/env python3
"""Deterministic mock review agent for orchestration-level tests.

Emits one ``assistant`` stream-json event whose text is pulled in order
from a pipe-delimited schedule in ``MOCK_SCHEDULE``. Call number is
persisted across invocations in ``MOCK_STATE_FILE`` so the review loop
can cycle through the schedule naturally.

A single optional ``MOCK_TOOL_CALL=1`` will also emit a short
tool_call_started/completed pair before the assistant message — useful
for tests that want to prove tool-call events are captured in
``events.jsonl`` alongside orchestration events.
"""

from __future__ import annotations

import json
import os
import sys


def _load_idx(state_path: str) -> int:
    try:
        with open(state_path, "r", encoding="utf-8") as fp:
            return int(fp.read().strip() or "0")
    except (OSError, ValueError):
        return 0


def main() -> int:
    state_path = os.environ["MOCK_STATE_FILE"]
    schedule = os.environ["MOCK_SCHEDULE"].split("|")
    idx = _load_idx(state_path)
    text = schedule[idx] if idx < len(schedule) else "VERDICT: APPROVED"

    with open(state_path, "w", encoding="utf-8") as fp:
        fp.write(str(idx + 1))

    if os.environ.get("MOCK_TOOL_CALL") == "1":
        started = {
            "type": "tool_call",
            "subtype": "started",
            "tool_call": {
                "shellToolCall": {
                    "args": {"command": "ls /tmp", "description": "probe"},
                },
            },
        }
        completed = {
            "type": "tool_call",
            "subtype": "completed",
            "tool_call": {
                "shellToolCall": {
                    "args": {"command": "ls /tmp"},
                    "result": {"success": {"exitCode": 0, "stdout": "probe"}},
                },
            },
        }
        print(json.dumps(started), flush=True)
        print(json.dumps(completed), flush=True)

    evt = {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": text}]},
        "model_call_id": f"m-{idx}",
    }
    print(json.dumps(evt), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
