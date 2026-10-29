"""Align GLM-5.3 prompt tokens with Qwen3.6-27B's tokens of the same prompt, so a Qwen-on-Qwen O-lens run can be shown at the GLM
tokens (to_reads.py --qwen-run): for each GLM token inside the system or user content, the Qwen token holding its last character
(the one whose activation has just read it; `exact` when both tokens end at that character). Template tokens map only where
they correspond: GLM's real <|system|> / <|user|> / <|assistant|> -> the last token of Qwen's "<|im_start|>system\\n" / "user\\n" /
"assistant\\n" header, <think> and </think> to Qwen's own. GLM's "Reasoning Effort: Max" system block has no Qwen counterpart.

    PYTHONPATH=.:src uv run --with transformers --with jinja2 python scripts/glm_qwen_reader/qwen_align.py \\
        GLM_REQUESTS.jsonl QWEN_RUN OUT.jsonl     # {id, glm_pos, qwen_pos, exact} per matched GLM token
"""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from counterfactual_investigator_agent_with_olens.render import glm_raw_prompt, render_request

glm_req = [json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines()]
qwen_req = {r["id"]: r for r in map(json.loads, (Path(sys.argv[2]) / "requests.jsonl").read_text().splitlines())}
gtok, qtok = AutoTokenizer.from_pretrained("zai-org/GLM-5.3"), AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")


def spans(tok, text):
    return tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]


def content_char(text, offs, contents):
    """token index -> (message index, char span within that message's content) for tokens inside a content span. Qwen's template
    strips the content, so the stripped text is what's searched, with offsets kept relative to the unstripped content."""
    out, cur = {}, 0
    for m, c in enumerate(contents):
        lead, c = len(c) - len(c.lstrip()), c.strip()
        s = text.index(c, cur)
        cur = s + len(c)
        for i, (a, b) in enumerate(offs):
            if s <= a and b <= cur and b > a:
                out[i] = (m, a - s + lead, b - s + lead)
    return out


n_exact = n_cont = n_tmpl = n_glm = 0
with open(sys.argv[3], "w") as f:
    for g in glm_req:
        q = qwen_req[g["id"]]
        msgs = q["messages"][:2]
        assert [m["role"] for m in msgs] == ["system", "user"]
        gtext = glm_raw_prompt(msgs)
        assert gtok(gtext, add_special_tokens=False).input_ids == g["prompt_ids"], g["id"]
        _, qids, _ = render_request(qtok, q)
        qtext = qtok.decode(qids)
        goffs, qoffs = spans(gtok, gtext), spans(qtok, qtext)
        assert len(qoffs) == len(qids)
        contents = [m["content"] for m in msgs]
        gc, qc = content_char(gtext, goffs, contents), content_char(qtext, qoffs, contents)
        qby = {}  # (message, char) -> qwen token index holding it
        for i, (m, a, b) in qc.items():
            for ch in range(a, b):
                qby[(m, ch)] = (i, b)
        pairs = []
        for i, (m, a, b) in gc.items():
            if (m, b - 1) in qby:
                j, qend = qby[(m, b - 1)]
                pairs.append((i, j, qend == b))
        qt = [qtok.decode([t]) for t in qids]
        gt = g["tokens"]
        sys_start = qtext.index(contents[0].strip())
        user_start = qtext.index(contents[1].strip())
        q_hdr = {
            "system": max(j for j, (a, b) in enumerate(qoffs) if b <= sys_start),
            "user": max(j for j, (a, b) in enumerate(qoffs) if b <= user_start),
        }
        q_asst = next(j for j in range(len(qt) - 1, -1, -1) if qt[j] == "assistant")
        tmpl = {"<|system|>": q_hdr["system"], "<|user|>": q_hdr["user"], "<|assistant|>": q_asst + 1}
        tmpl |= {
            "<think>": max(j for j, t in enumerate(qt) if t == "<think>"),
            "</think>": max(j for j, t in enumerate(qt) if t == "</think>"),
        }
        last_sys = max(i for i, t in enumerate(gt) if t == "<|system|>")
        for t, j in tmpl.items():
            i = last_sys if t == "<|system|>" else max(i for i, x in enumerate(gt) if x == t)
            pairs.append((i, j, False))
            n_tmpl += 1
        read = set(g["positions"]) | {d[0] for d in g["dups"]}
        for i, j, exact in sorted(pairs):
            if i in read:
                f.write(json.dumps({"id": g["id"], "glm_pos": i, "qwen_pos": j, "qwen_token": qt[j], "exact": exact}) + "\n")
                n_exact += exact
                n_cont += not exact
        n_glm += len(read)
print(f"{n_glm} GLM read positions: {n_exact} exact end matches, {n_cont} containing-token or template matches ({n_tmpl} template)")
