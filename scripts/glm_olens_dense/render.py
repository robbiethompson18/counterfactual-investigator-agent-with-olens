import ast
import json

from transformers import AutoTokenizer

TOK = None


def tok(d="/home/claude/workspace/tmp/glm_dense/tok"):
    global TOK
    if TOK is None:
        TOK = AutoTokenizer.from_pretrained(d)
    return TOK


def _calls(tc):
    if isinstance(tc, str):
        tc = ast.literal_eval(tc)
    out = []
    for c in tc or []:
        f = c.get("function", c)
        a = f.get("arguments") or "{}"
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except json.JSONDecodeError:
                a = {"arguments": a}
        if not isinstance(a, dict):
            a = {"arguments": a}
        out.append({"id": c.get("id"), "type": "function", "function": {"name": f["name"], "arguments": a}})
    return out


def norm_messages(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("role") == "assistant":
            r = m.get("reasoning_content", m.get("reasoning"))
            n = {"role": "assistant", "content": m.get("content") or ""}
            if isinstance(r, str):
                n["reasoning_content"] = r
            if m.get("tool_calls"):
                n["tool_calls"] = _calls(m["tool_calls"])
            out.append(n)
        else:
            out.append({k: v for k, v in m.items() if k in ("role", "content", "tool_call_id")} | {"content": m.get("content") or ""})
    return out


def render(msgs, tools):
    return tok().apply_chat_template(norm_messages(msgs), tools=tools or None, tokenize=False)


CAP = 32768


def n_tokens(text):
    return len(tok().encode(text, add_special_tokens=False))


def late_view(msgs, tools, cap=CAP):
    """Celeste's approximate tail view: head through the first user message, one omitted-count system line, then the longest
    tail that fits (never opening on a tool result). Returns (text, number of tail messages kept); 0 = no late view."""
    fu = next(i for i, m in enumerate(msgs) if m["role"] == "user")
    head = msgs[: fu + 1]
    rest = msgs[fu + 1 :]

    def build(k):
        tail = rest[len(rest) - k :] if k else []
        omitted = len(rest) - k
        return render(head + [{"role": "system", "content": f"[{omitted} earlier messages omitted for length]"}] + tail, tools)

    ks = [0] + [k for k in range(1, len(rest) + 1) if rest[len(rest) - k]["role"] != "tool"]  # the tail never opens on a tool result
    lo, hi = 0, len(ks) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if n_tokens(build(ks[mid])) <= cap:
            lo = mid
        else:
            hi = mid - 1
    return build(ks[lo]), ks[lo]
