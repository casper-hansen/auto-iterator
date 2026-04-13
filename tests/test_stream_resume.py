"""End-to-end test: run_agent auto-resumes after WritableIterable stream closure.

Uses a mock agent script that:
  1st call  → emits stream-json, then prints "S: WritableIterable is closed", exits 1
  2nd call  → (with --continue) emits stream-json, exits 0

Verifies that run_agent transparently retries and returns the combined text.
"""

from __future__ import annotations

import asyncio
import os
import stat
import sys
import tempfile
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iterator_loop.agent import run_agent  # noqa: E402


MOCK_SCRIPT = Path(__file__).resolve().parent / "mock_agent.sh"


async def _test_auto_resume() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state_file = os.path.join(tmp, "mock_state")

        mock_path = str(MOCK_SCRIPT)
        os.chmod(mock_path, os.stat(mock_path).st_mode | stat.S_IEXEC)

        env_backup = os.environ.copy()
        os.environ["MOCK_STATE_FILE"] = state_file

        try:
            rc, text = await run_agent(
                model="test-model",
                prompt="Run the experiment",
                tag="[TEST]",
                workspace="/tmp",
                agent_cmd=mock_path,
                extra_flags=[],
            )
        finally:
            os.environ.clear()
            os.environ.update(env_backup)

    print()
    print("=" * 60)
    print(f"Exit code : {rc}")
    print(f"Text len  : {len(text)} chars")
    print(f"State file existed (retry happened): {os.path.exists(state_file)}")
    print()
    print("Captured text:")
    print("-" * 60)
    print(text)
    print("-" * 60)

    assert rc == 0, f"Expected exit code 0, got {rc}"
    assert "Resumed after stream closure" in text, "Resume text not found"
    assert "Task complete" in text, "Completion text not found"
    print()
    print("PASS — auto-resume worked correctly")


if __name__ == "__main__":
    asyncio.run(_test_auto_resume())
