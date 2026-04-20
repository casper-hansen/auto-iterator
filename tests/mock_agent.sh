#!/usr/bin/env bash
# Mock agent CLI that simulates server-side session kills.
#
# MOCK_FAILURE_MODE controls which failure to simulate:
#   "stream_closed"        — explicit "WritableIterable is closed" message
#   "silent_kill"          — non-zero exit while a tool call is in-flight
#   "silent_between_tools" — non-zero exit AFTER a tool completes, with no
#                            stream_closed message and no `result` event —
#                            matches the straw-fireworks-backend log crash
#   "sigkill_between_tools"— same shape but killed by SIGKILL (rc=-9)
#
# MOCK_STATE_FILE tracks invocation count so the second call (--continue) succeeds.

STATE_FILE="${MOCK_STATE_FILE:?MOCK_STATE_FILE must be set}"
FAILURE_MODE="${MOCK_FAILURE_MODE:-stream_closed}"

has_continue=false
for arg in "$@"; do
    if [[ "$arg" == "--continue" ]]; then
        has_continue=true
        break
    fi
done

if [[ "$has_continue" == true && -f "$STATE_FILE" ]]; then
    # Resumed session — succeed
    echo '{"type":"assistant","message":{"content":[{"type":"text","text":"Resumed after interruption. Continuing monitoring..."}]},"model_call_id":"resume-1"}'
    sleep 0.3
    echo '{"type":"assistant","message":{"content":[{"type":"text","text":"Task complete. Training finished successfully."}]},"model_call_id":"resume-2"}'
    exit 0
fi

# First session — work for a bit, then die
touch "$STATE_FILE"
echo '{"type":"assistant","message":{"content":[{"type":"text","text":"Starting experiment monitoring..."}]},"model_call_id":"call-1"}'
sleep 0.3
echo '{"type":"tool_call","subtype":"started","tool_call":{"name":"Shell","parameters":{"command":"sleep 3600 && tail -1 /tmp/training.log"}}}'
sleep 0.3

if [[ "$FAILURE_MODE" == "stream_closed" ]]; then
    echo 'S: WritableIterable is closed'
    exit 1
elif [[ "$FAILURE_MODE" == "silent_kill" ]]; then
    # Server kills the session silently — tool call never completes
    exit 1
elif [[ "$FAILURE_MODE" == "silent_between_tools" ]]; then
    # The exact shape from the straw-fireworks-backend.20260420_182159 crash:
    # a tool completes cleanly, then the CLI exits non-zero with no result event
    # and no pending tool calls, no stream-closed message.
    echo '{"type":"tool_call","subtype":"completed","tool_call":{"shellToolCall":{"args":{"command":"sleep 3600 && tail -1 /tmp/training.log"},"result":{"success":{"exitCode":0,"stdout":"line"}}}}}'
    sleep 0.1
    exit 1
elif [[ "$FAILURE_MODE" == "sigkill_between_tools" ]]; then
    # Same shape but killed by SIGKILL (what an OOM kill looks like).
    echo '{"type":"tool_call","subtype":"completed","tool_call":{"shellToolCall":{"args":{"command":"sleep 3600 && tail -1 /tmp/training.log"},"result":{"success":{"exitCode":0,"stdout":"line"}}}}}'
    sleep 0.1
    kill -9 $$
else
    echo "Unknown MOCK_FAILURE_MODE: $FAILURE_MODE" >&2
    exit 2
fi
