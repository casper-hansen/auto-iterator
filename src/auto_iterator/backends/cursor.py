"""Cursor agent CLI backend — original behavior of the iterator loop."""

from __future__ import annotations

from ..tool_formatter import tool_summary

_RESUME_PROMPT = (
    "Your previous session ended unexpectedly before completing the task. "
    "Please continue where you left off."
)


class CursorBackend:
    name = "cursor"
    default_cmd = "agent"
    display_name = "Cursor agent CLI"
    install_hint = "Install it with: curl https://cursor.com/install -fsSL | bash"

    # Cursor runs both Claude and non-Claude models; "-thinking-max" is
    # Cursor's way of pinning max reasoning effort on a model.
    default_impl_model = "claude-opus-4-7-thinking-max"
    default_fix_model = "claude-opus-4-7-thinking-max"
    default_reviewer_model = "gpt-5.4-xhigh"
    default_experimenter_model = "claude-opus-4-7-thinking-max"
    default_adjuster_model = "claude-opus-4-7-thinking-max"
    default_analyst_model = "gpt-5.4-xhigh"

    def build_initial_cmd(
        self,
        agent_cmd: str,
        model: str,
        prompt: str,
        workspace: str,
        extra_flags: list[str],
    ) -> list[str]:
        cmd = [
            agent_cmd, "-p",
            "--output-format", "stream-json",
            "--stream-partial-output",
            "--model", model,
            "--workspace", workspace,
            "--trust",
            "--force",
        ]
        cmd.extend(extra_flags)
        cmd.append(prompt)
        return cmd

    def build_continue_cmd(
        self,
        agent_cmd: str,
        model: str,
        workspace: str,
        extra_flags: list[str],
    ) -> list[str]:
        cmd = [
            agent_cmd, "-p",
            "--output-format", "stream-json",
            "--stream-partial-output",
            "--model", model,
            "--workspace", workspace,
            "--trust",
            "--force",
            "--continue",
        ]
        cmd.extend(extra_flags)
        cmd.append(_RESUME_PROMPT)
        return cmd

    def handle_event(self, evt: dict, reader) -> None:
        etype = evt.get("type", "")

        if etype == "assistant":
            content_parts = evt.get("message", {}).get("content", [])
            ts_ms = evt.get("timestamp_ms")
            is_partial = ts_ms is not None and "model_call_id" not in evt

            if is_partial:
                for part in content_parts:
                    if part.get("type") == "text" and part["text"]:
                        reader._text_buf.append(part["text"])
                reader._flush_complete_lines()
            else:
                reader._flush_text()
                assembled = "".join(
                    p.get("text", "")
                    for p in content_parts
                    if p.get("type") == "text"
                )
                if assembled:
                    reader._full_text.append(assembled)

        elif etype == "tool_call":
            reader._flush_text()
            sub = evt.get("subtype", "")
            tc = evt.get("tool_call", {})
            if sub == "started":
                reader.pending_tools += 1
                reader._fmt.feed_tool(f"→ {tool_summary(tc)}")
            elif sub == "completed":
                reader.pending_tools = max(0, reader.pending_tools - 1)
                reader._fmt.feed_tool(f"← {tool_summary(tc, completed=True)}")

        elif etype == "result":
            reader._flush_text()
            reader.saw_result = True
            result_text = evt.get("result", "")
            if result_text and not reader._full_text:
                reader._full_text.append(result_text)
