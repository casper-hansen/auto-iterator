"""End-to-end test: run_agent auto-resumes after server-side session kills.

Covers both observed failure modes:
  1. Explicit "WritableIterable is closed" message + non-zero exit
  2. Silent non-zero exit while a tool call is still in-flight

Uses a mock agent script that dies on the first call and succeeds
on the second (--continue) call.
"""

from __future__ import annotations

import asyncio
import os
import stat
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iterator_loop.agent import run_agent  # noqa: E402

MOCK_SCRIPT = Path(__file__).resolve().parent / "mock_agent.sh"


async def _run_scenario(failure_mode: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state_file = os.path.join(tmp, "mock_state")

        mock_path = str(MOCK_SCRIPT)
        os.chmod(mock_path, os.stat(mock_path).st_mode | stat.S_IEXEC)

        env_backup = os.environ.copy()
        os.environ["MOCK_STATE_FILE"] = state_file
        os.environ["MOCK_FAILURE_MODE"] = failure_mode

        try:
            rc, text = await run_agent(
                model="test-model",
                prompt="Run the experiment",
                tag=f"[{failure_mode}]",
                workspace="/tmp",
                agent_cmd=mock_path,
                extra_flags=[],
            )
        finally:
            os.environ.clear()
            os.environ.update(env_backup)

    print(f"  exit code : {rc}")
    print(f"  text len  : {len(text)} chars")
    print(f"  captured  : {text[:120]}…" if len(text) > 120 else f"  captured  : {text}")

    assert rc == 0, f"Expected exit code 0, got {rc}"
    assert "Resumed after interruption" in text, "Resume text not found"
    assert "Task complete" in text, "Completion text not found"
    print(f"  PASS\n")


async def _main() -> None:
    print("=" * 60)
    print("Test 1: stream_closed (WritableIterable is closed)")
    print("-" * 60)
    await _run_scenario("stream_closed")

    print("=" * 60)
    print("Test 2: silent_kill (non-zero exit, pending tool call)")
    print("-" * 60)
    await _run_scenario("silent_kill")

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(_main())
