"""Chat-template rendering of OpenAI-style messages, to find token positions in what the model saw (O-lens reads, viewers).

    from counterfactual_investigator_agent_with_olens.render import chat_ids, chat_text, glm_raw_prompt
    text = chat_text(tok, messages, tools=schemas, generation_prompt=True)   # as vLLM renders a chat request
    ids = chat_ids(tok, messages, generation_prompt=True, thinking=False)    # the same, tokenized
    prompt = glm_raw_prompt(messages)                                         # evalcue's GLM 5.3 raw-prefill prompt

Keep this module free of third-party imports (the tokenizer is an argument): the olens-read pod imports it with
PYTHONPATH=olens/src:src (runpod/olens_reads.sh) and the transcript-viewer venv through sys.path (transcript-viewer/web/build.py).
"""

import copy
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
