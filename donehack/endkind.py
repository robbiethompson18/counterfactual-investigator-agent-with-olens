"""Split episodes recorded as end="responded" (a turn with no tool call) into what actually happened.

finish_reason wasn't logged for most runs, so this is inferred from the final assistant message:
- "truncated": no visible text after a long think (>= 20k chars of reasoning), i.e. the turn hit the
  8000-token max_tokens cap mid-reasoning and came back with no tool call
- "malformed_call": the text is a tool call the API didn't parse (raw DSML / <tool_call> markup)
- "empty": no visible text and a short think (provider returned nothing)
- "responded": an actual reply to the user
New runs log finish_reason, which takes precedence ("length" -> truncated).
"""


def end_kind(record, messages):
    if record["end"] != "responded":
        return record["end"]
    last = [m for m in messages if m["role"] == "assistant"][-1]
    if record.get("finish_reason") == "length":
        return "truncated"
    text = (last.get("content") or "").strip()
    if "DSML" in text or "<tool_call" in text or "<function=" in text:
        return "malformed_call"
    if not text:
        return "truncated" if len(last.get("reasoning") or "") >= 20000 else "empty"
    return "responded"
