"""Chat-template rendering of OpenAI-style messages, to find token positions in what the model saw (O-lens reads, viewers).

    from counterfactual_investigator_agent_with_olens.render import chat_ids, chat_text, glm_raw_prompt
    text = chat_text(tok, messages, tools=schemas, generation_prompt=True)   # as vLLM renders a chat request
    ids = chat_ids(tok, messages, generation_prompt=True, thinking=False)    # the same, tokenized
    prompt = glm_raw_prompt(messages)                                         # evalcue's GLM 5.3 raw-prefill prompt

Keep this module free of third-party imports (the tokenizer is an argument): the olens-read pod imports it with
PYTHONPATH=olens/src:src (runpod/olens_reads.sh) and the transcript-viewer venv through sys.path (transcript-viewer/web/build.py).
"""

import copy
import itertools
import json


def tool_args_as_dicts(messages: list[dict]) -> list[dict]:
    """A deep copy with each tool call's `arguments` parsed from JSON text to a dict, as vLLM does before templating:
    the Qwen3.6 template iterates `tool_call.arguments|items`, so a JSON string would render wrong."""
    msgs = copy.deepcopy(messages)
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            if isinstance(tc["function"]["arguments"], str):
                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
    return msgs


def chat_text(tok, messages: list[dict], *, tools: list[dict] | None = None, generation_prompt: bool = False, thinking: bool = True) -> str:
    """`tok.apply_chat_template` as vLLM renders a chat request. `generation_prompt` appends the open assistant header
    (`<|im_start|>assistant\\n<think>\\n`, or the closed empty think block when `thinking` is off; Qwen3.6's template reads
    `enable_thinking` only there, so `thinking` changes nothing without `generation_prompt`). Past `reasoning_content` renders
    only on assistant turns after the last user query, as at generation time (no `preserve_thinking`)."""
    return tok.apply_chat_template(
        tool_args_as_dicts(messages), tools=tools or None, tokenize=False, add_generation_prompt=generation_prompt, enable_thinking=thinking
    )


def chat_ids(tok, messages: list[dict], **kw) -> list[int]:
    """Token ids of chat_text (`add_special_tokens=False`: the template carries every special token)."""
    return list(tok(chat_text(tok, messages, **kw), add_special_tokens=False).input_ids)


# zai-org/GLM-5.3 chat_template.jinja by hand: its prefix with add_generation_prompt and the default reasoning effort, the
# system/user turns, then the "</think>" it puts after "<think>" on an assistant turn that had no reasoning
GLM_RAW_PREFIX = "[gMASK]<sop><|system|>Reasoning Effort: Max"
GLM_RAW_TURN = {"system": "<|system|>", "user": "<|user|>"}
GLM_RAW_EMPTY_THINK = "<|assistant|><think></think>"


def glm_raw_prompt(messages: list[dict]) -> str:
    """The raw completions prompt evalcue sends to run GLM 5.3 with thinking off (EVALCUE_RAW_PREFILL in the deleted evalcue/harness.py):
    the hand-rendered template with an empty think block prefilled. Single turn: system/user messages only, no tools."""
    assert all(m["role"] in GLM_RAW_TURN for m in messages), "raw prefill: system/user turns only"
    return GLM_RAW_PREFIX + "".join(GLM_RAW_TURN[m["role"]] + m["content"] for m in messages) + GLM_RAW_EMPTY_THINK


BASH_TOOL = {  # evalcue/harness.py; copied so this runs on a pod without evalcue's deps
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command in the working directory. Returns stdout+stderr (truncated) and the exit code.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}


def render_request(tok, row: dict) -> tuple[list[dict], list[int], list[tuple]]:
    """One olens_read_requests.py request row (messages, tools, generation_prompt) rendered as Qwen saw it -> (messages, token
    ids, per-token (msg_index, field, char_offset)). The batch readers and the transcripts backfill share it."""
    msgs = tool_args_as_dicts(row["messages"])
    tools = [BASH_TOOL] if row.get("tools") == "evalcue.harness.BASH_TOOL" else row.get("tools")
    text = chat_text(tok, msgs, tools=tools, generation_prompt=bool(row.get("generation_prompt")))
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    # Find each message's fields in the rendered text, in order; a token belongs to the last message whose first
    # field starts at or before it (so template tokens around a message go with it), -1 before the first.
    fields, cur = [], 0  # (start, end, msg_index, field)
    for i, m in enumerate(msgs):
        parts = [("reasoning_content", m.get("reasoning_content")), ("content", m.get("content"))]
        parts += [("tool_call", v) for tc in m.get("tool_calls") or [] for v in tc["function"]["arguments"].values()]
        for name, s in parts:
            s = s.strip() if isinstance(s, str) else None
            if not s:
                continue
            j = text.find(s, cur)
            if j < 0:
                continue
            fields.append((j, j + len(s), i, name))
            cur = j + len(s)
    starts = {}
    for a, _, i, _ in fields:
        starts.setdefault(i, a)
    where = []
    for a, _ in enc.offset_mapping:
        msg = max((i for i, s in starts.items() if s <= a), default=-1)
        f = next(((name, a - s) for s, e, i, name in fields if s <= a < e), (None, None))
        where.append((msg, *f))
    return msgs, enc.input_ids, where


ASSISTANT_SECTION = {"reasoning_content": "reasoning", "tool_call": "assistant tool call"}  # by render_request's field


def transcript_line(row: dict, msgs: list[dict], toks: list[str], where: list[tuple]) -> dict:
    """The `.transcripts.jsonl` line a batch reader writes beside its reads (docs/olens-reads-format.md): the request's
    metadata, every token as text and the chat sections (one per <|im_start|> block, an assistant block split into
    reasoning / tool call / reply), so a viewer needs neither the tokenizer nor the chat template."""
    starts = [i for i, t in enumerate(toks) if t == "<|im_start|>"]
    bounds = sorted({0, *starts, len(toks)})
    sections = []
    for a, b in itertools.pairwise(bounds):
        msg = next((w[0] for w in where[a:b] if w[1]), None)  # the message whose text is in this block
        role = msgs[msg]["role"] if msg is not None else toks[a + 1].strip() if toks[a] == "<|im_start|>" and a + 1 < b else "system"
        first = len(sections)  # sections merge only within a block
        for i in range(a, b):
            name = ASSISTANT_SECTION.get(where[i][1], "assistant") if role == "assistant" else role
            if len(sections) > first and sections[-1][0] == name:
                sections[-1][2] = i + 1
            else:
                sections.append([name, i, i + 1])
    meta = {k: v for k, v in row.items() if k not in ("messages", "tools")}
    return {**meta, "tokens": toks, "sections": sections}
