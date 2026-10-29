"""O-lens read requests (scripts/olens_read_requests.py's input) for sim_or_real's Qwen (thinking off, vLLM) runs.

    PYTHONPATH=src uv run python -m environments.sim_or_real.olens_requests OUT.jsonl

One episode per (arm, item), sample 0 (bare and opencode: every sample), from the live arms of: the single-turn ladder
(`replace` rungs), `after_trace` (one per trace), `bare`, opencode `ask` and `after_bump` (thinking off). Read positions,
as explicit token indices into the rendering the reader makes (render_request with the Qwen3.6-27B tokenizer): the
question's punctuation / delimiter / newline tokens (in the last user turn), every token from the end of that turn
through the assistant header (<|im_end|>, <|im_start|>, "assistant", "\\n", the empty think block), and the reply's first
content token (the answer letter, or "(" / " " before it, so the next token too).
"""

import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

from counterfactual_investigator_agent_with_olens.render import render_request
from environments._lib.core import load_env
from environments._lib.runfolder import LOCAL, read_episodes

PUNCT = re.compile(r"^\s*[.,;:!?]+[\"')\]*]*\s*$")  # as scripts/olens_read_requests.py
DELIM = re.compile(r"^\s*[^\w\s]+\s*$|^\s*\n\s*$")
RUNS = {  # run -> which arms, which samples
    "2026-10-29_qwen-vllm-ladder1t": (lambda a: a.startswith(("replace/ladder/", "after_trace/")), {0}),
    "2026-10-29_qwen-vllm-bare": (lambda a: a == "bare", None),
    "2026-10-29_qwen-vllm-opencode-ask": (lambda a: a == "opencode/admete/ask", None),
    "2026-10-29_qwen-vllm-opencode-bump": (lambda a: a == "opencode/admete/after_bump", None),
}


def positions(tok, row) -> list[int]:
    msgs, ids, where = render_request(tok, row)
    toks = [tok.decode([i]) for i in ids]
    q = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    a = q + 1  # the reply
    in_q = [i for i, w in enumerate(where) if w[0] == q and w[1] == "content"]
    out = {i for i in in_q if PUNCT.match(toks[i]) or DELIM.match(toks[i])}
    first_reply = min(i for i, w in enumerate(where) if w[0] == a and w[1] == "content")
    out |= set(range(in_q[-1] + 1, first_reply))  # end of the question's turn through the assistant header
    out |= {first_reply, first_reply + 1}  # the answer letter (or the "(" before it)
    return sorted(out)


def main(out: Path):
    env = load_env("sim_or_real")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    rows = []
    for run, (keep, samples) in RUNS.items():
        for ep in read_episodes(LOCAL / "sim_or_real" / run):
            if ep.get("error") or not keep(ep["arm"]) or ep["arm"] in env.RETIRED or (samples and ep["sample"] not in samples):
                continue
            m = env.measure(ep)
            if m["positive"] is None:
                continue
            msgs = ep["messages"][: max(i for i, x in enumerate(ep["messages"]) if x["role"] == "user") + 2]  # through the answer
            spec = ep.get("episode_spec") or {}
            row = {"id": f"sim_or_real/{run}:{ep['id']}", "env": "sim_or_real", "arm": ep["arm"], "item": ep["item"]}
            row |= {"positive": m["positive"], "thinking": False, "priority": 1, "messages": msgs}
            row["tools"] = None if spec.get("tools_in_prompt") else (ep.get("tools") or None)  # tools-in-prompt rungs: in the text
            row["positions"] = positions(tok, row)
            rows.append(row)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} requests, {sum(len(r['positions']) for r in rows)} positions -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
