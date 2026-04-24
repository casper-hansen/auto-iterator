"""End-to-end test: run_agent auto-resumes after server-side session kills.

Covers the observed failure modes:
  1. Explicit "WritableIterable is closed" message + non-zero exit
  2. Silent non-zero exit while a tool call is still in-flight
  3. Silent non-zero exit *between* tool calls (matches straw-fireworks-backend
     log crash — tool completes, then the CLI dies with no result event)
  4. SIGKILL between tool calls (OOM-kill shape)

Also verifies the retry cap: a permanently-crashing mock bubbles up a non-zero
rc and does not loop forever.

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

from auto_iterator.agent import run_agent  # noqa: E402

MOCK_SCRIPT = Path(__file__).resolve().parent / "mock_agent.sh"


async def _run_scenario(failure_mode: str, *, expect_rc: int = 0) -> tuple[int, str]:
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

    if expect_rc == 0:
        assert rc == 0, f"Expected exit code 0, got {rc}"
        assert "Resumed after interruption" in text, "Resume text not found"
        assert "Task complete" in text, "Completion text not found"
    else:
        assert rc != 0, f"Expected non-zero rc for {failure_mode}, got {rc}"

    print("  PASS\n")
    return rc, text


async def _run_retry_cap_scenario() -> None:
    """With max_resume_attempts=0, a failing mock must surface its rc instead
    of silently succeeding on a resume."""
    with tempfile.TemporaryDirectory() as tmp:
        state_file = os.path.join(tmp, "mock_state")

        mock_path = str(MOCK_SCRIPT)
        os.chmod(mock_path, os.stat(mock_path).st_mode | stat.S_IEXEC)

        env_backup = os.environ.copy()
        os.environ["MOCK_STATE_FILE"] = state_file
        os.environ["MOCK_FAILURE_MODE"] = "silent_between_tools"

        try:
            rc, text = await run_agent(
                model="test-model",
                prompt="Run the experiment",
                tag="[retry-cap]",
                workspace="/tmp",
                agent_cmd=mock_path,
                extra_flags=[],
                max_resume_attempts=0,
            )
        finally:
            os.environ.clear()
            os.environ.update(env_backup)

    print(f"  exit code : {rc}")
    print(f"  text len  : {len(text)} chars")
    assert rc != 0, f"Expected non-zero rc with no retries, got {rc}"
    assert "Resumed after interruption" not in text, (
        "Should not have resumed when max_resume_attempts=0"
    )
    print("  PASS\n")


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
    print("Test 3: silent_between_tools (rc=1, no pending tool, no result event)")
    print("-" * 60)
    await _run_scenario("silent_between_tools")

    print("=" * 60)
    print("Test 4: sigkill_between_tools (killed by SIGKILL between tool calls)")
    print("-" * 60)
    await _run_scenario("sigkill_between_tools")

    print("=" * 60)
    print("Test 5: retry cap is honoured (max_resume_attempts=0 surfaces rc)")
    print("-" * 60)
    await _run_retry_cap_scenario()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(_main())
