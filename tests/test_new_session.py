"""Verify the agent subprocess is spawned in its own session.

When ``_run_once`` creates a subprocess it must pass ``start_new_session=True``
so the child is unreachable by SIGHUP from the launching terminal.  This test
spawns a tiny mock that prints its own PID / SID / PGID and asserts:

    child_sid  != parent_sid
    child_pgid != parent_pgid
    child_pid == child_sid == child_pgid  (child is its own session & group leader)

If this invariant ever regresses, long-running runs will start dying again on
ssh disconnect / terminal close.
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


MOCK = r"""#!/usr/bin/env bash
# Emit pid/sid/pgid as a single JSON assistant message, then exit cleanly.
pid=$$
sid=$(ps -o sid= -p "$pid" | tr -d ' ')
pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
printf '{"type":"assistant","message":{"content":[{"type":"text","text":"pid=%s sid=%s pgid=%s"}]},"model_call_id":"m1"}\n' "$pid" "$sid" "$pgid"
exit 0
"""


async def _main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "sid_probe.sh"
        script.write_text(MOCK)
        script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IREAD)

        rc, text = await run_agent(
            model="test-model",
            prompt="probe",
            tag="[new-session]",
            workspace="/tmp",
            agent_cmd=str(script),
            extra_flags=[],
        )

    assert rc == 0, f"Probe exited rc={rc}"
    print(f"  captured  : {text}")

    parts = dict(kv.split("=", 1) for kv in text.split() if "=" in kv)
    child_pid = int(parts["pid"])
    child_sid = int(parts["sid"])
    child_pgid = int(parts["pgid"])

    parent_sid = os.getsid(0)
    parent_pgid = os.getpgid(0)

    print(f"  parent    : sid={parent_sid} pgid={parent_pgid}")
    print(f"  child     : pid={child_pid} sid={child_sid} pgid={child_pgid}")

    assert child_sid != parent_sid, (
        f"Child shares the parent's session (sid={child_sid}); "
        "start_new_session=True is not taking effect and SIGHUP from the "
        "launching terminal can still kill it."
    )
    assert child_pgid != parent_pgid, (
        f"Child shares the parent's process group (pgid={child_pgid})."
    )
    assert child_pid == child_sid == child_pgid, (
        f"Child should be the leader of its own session and group "
        f"(pid={child_pid} sid={child_sid} pgid={child_pgid})."
    )

    print("  PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Test: agent subprocess is in its own session (start_new_session)")
    print("-" * 60)
    asyncio.run(_main())
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
