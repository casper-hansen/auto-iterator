#!/usr/bin/env bash
# Mock agent CLI that simulates the "WritableIterable is closed" error.
#
# Behaviour is controlled by a state file:
#   - First invocation (no --continue): emit stream-json, then die with the error
#   - Second invocation (--continue present): emit stream-json, exit 0
#
# The state file path is passed via MOCK_STATE_FILE env var.

STATE_FILE="${MOCK_STATE_FILE:?MOCK_STATE_FILE must be set}"

has_continue=false
for arg in "$@"; do
    if [[ "$arg" == "--continue" ]]; then
        has_continue=true
        break
    fi
done

if [[ "$has_continue" == true && -f "$STATE_FILE" ]]; then
    # Resumed session — succeed
    echo '{"type":"assistant","message":{"content":[{"type":"text","text":"Resumed after stream closure. Continuing monitoring..."}]},"model_call_id":"resume-1"}'
    sleep 0.3
    echo '{"type":"assistant","message":{"content":[{"type":"text","text":"Task complete. Training finished successfully."}]},"model_call_id":"resume-2"}'
    exit 0
else
    # First session — work for a bit, then die
    touch "$STATE_FILE"
    echo '{"type":"assistant","message":{"content":[{"type":"text","text":"Starting experiment monitoring..."}]},"model_call_id":"call-1"}'
    sleep 0.3
    echo '{"type":"tool_call","subtype":"started","tool_call":{"name":"Shell","parameters":{"command":"sleep 3600 && tail -1 /tmp/training.log"}}}'
    sleep 0.5
    # Server-side gRPC stream timeout — exact error from production logs
    echo 'S: WritableIterable is closed'
    exit 1
fi
