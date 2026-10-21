"""O-lens reads for combo_google.

header: greedy reads at the 4 non-newline header tokens of the final user turn (<|im_end|>, <|im_start|>,
        assistant, <think>) x layers 28..52, for each condition x prompt.
cot:    for sampled replies, the 5 tokens right before the first mention of the vendor the answer picked, in the
        reasoning (x layers 28/36/44/52; not for pair probes) and in the answer (x layers 28/40/52).

uv run --with openai --with anthropic --with transformers --with jinja2 python scripts/identity/combo_google/lens.py header|cot
"""

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/combo_google")
from conds import BELIEF, CONDS, PAIR_PROBES, PROBES, SPEC, is_toolcall, msgs
from qwen import header_positions, read, render

OUT = Path("data/runs/identity/2026-10-21_combo_google")
HEADER_LAYERS = tuple(range(28, 53, 4))
COT_LAYERS = {"cot": (28, 36, 44, 52), "answer": (28, 40, 52)}
PP = SPEC["pair_products"]
HEADER_PROMPTS = {
    "assistant_for_mom": PROBES["assistant_for_mom"],
    "pair_personal|google>anthropic": PAIR_PROBES["pair_personal"].format(a=PP["google"], b=PP["anthropic"]),
    "pair_personal|openai>google": PAIR_PROBES["pair_personal"].format(a=PP["openai"], b=PP["google"]),
    "trustworthy_lab": PROBES["trustworthy_lab"],
    "belief_direct": BELIEF["direct"],
}
# name as it appears when the model starts writing it in its reasoning
NAME_RE = {
    "google": r"Gemini|Google",
    "anthropic": r"Claude|Anthropic",
    "openai": r"ChatGPT|GPT|OpenAI",
    "alibaba": r"Qwen|Alibaba",
    "meta": r"Llama|Meta",
    "microsoft": r"Copilot|Microsoft",
}


def header():
    path = OUT / "lens_header.jsonl"
    done = {json.loads(line)["key"] for line in path.open()} if path.exists() else set()
    for cond in CONDS:
        for pk, q in HEADER_PROMPTS.items():
            key = f"{cond}|{pk}"
            if key in done:
                continue
            ids, toks = render(msgs(cond, q))
            pos = [p for p in header_positions(ids) if toks[p] != "\n"]
            items = [(p, L) for p in pos for L in HEADER_LAYERS]
            res = read(ids, items)
            with path.open("a") as f:
                for (p, L), r in zip(items, res):
                    f.write(
                        json.dumps(
                            {"key": key, "cond": cond, "prompt": pk, "pos": p, "tok": toks[p], "layer": L, **r[0]}, ensure_ascii=False
                        )
                        + "\n"
                    )
            print(key, len(items), flush=True)


API_Q = {p["probe"]: p["prompt"] for p in json.load(open("data/runs/identity/2026-10-21_beh_mine/probes.json"))["probes"]}
# second pass: all 6 header tokens (incl. the newlines; env_google found the newline after `assistant` at L56 the
# most Google-readable spot) x layers 44-56, for env_google's cue vs its persona-only control and the API probes
HEADER2 = [(c, pk) for c in ("fnfirst", "support") for pk in ("assistant_for_mom", "pair_personal|google>anthropic", "auth_header")]
HEADER2 += [(c, "auth_header") for c in ("TM_uswest", "TM_hangzhou", "TM", "none")] + [
    (c, "assistant_for_mom") for c in ("TM_uswest", "none")
]
HEADER2_LAYERS = (44, 48, 52, 56)


def header2():
    path = OUT / "lens_header2.jsonl"
    done = {json.loads(line)["key"] for line in path.open()} if path.exists() else set()
    for cond, pk in HEADER2:
        key = f"{cond}|{pk}"
        if key in done:
            continue
        ids, toks = render(msgs(cond, HEADER_PROMPTS.get(pk) or API_Q[pk]))
        items = [(p, L) for p in header_positions(ids) for L in HEADER2_LAYERS]
        res = read(ids, items)
        with path.open("a") as f:
            for (p, L), r in zip(items, res):
                f.write(
                    json.dumps({"key": key, "cond": cond, "prompt": pk, "pos": p, "tok": toks[p], "layer": L, **r[0]}, ensure_ascii=False)
                    + "\n"
                )
        print(key, len(items), flush=True)


def first_mention_tok(toks, vendor, site):
    """Token index where the picked vendor is first named inside the reasoning (site="cot") or the answer
    (site="answer"); None if never."""
    joined, starts = "", []
    for s in toks:
        starts.append(len(joined))
        joined += s
    a = joined.rfind("<think>")
    b = joined.find("</think>", a)
    if site == "answer":
        a, b = b, len(joined)
    m = re.search(NAME_RE[vendor], joined[a:b])
    if not m:
        return None
    c = a + m.start()
    return max(i for i, s in enumerate(starts) if s <= c)


def cot(cells, per_cell, seed=0):
    rows = [json.loads(line) for line in (OUT / "samples.jsonl").open()]
    path = OUT / "lens_cot.jsonl"
    done = {(json.loads(line)["sid"], json.loads(line).get("site", "cot")) for line in path.open()} if path.exists() else set()
    rng = random.Random(seed)
    for cond, probe in cells:
        rs = [r for r in rows if r["cond"] == cond and r["probe"] == probe and r["top"] in NAME_RE and not is_toolcall(r["content"])]
        rng.shuffle(rs)
        g = [r for r in rs if r["top"] == "google"]
        o = [r for r in rs if r["top"] != "google"]
        pick = g[: per_cell // 2] + o[: per_cell - min(len(g), per_cell // 2)]
        for r in pick:
            # pair probes restate both options at the top of the CoT, so only the answer site is informative there
            sites = [x for x in (("answer",) if probe.startswith("pair_") else ("cot", "answer")) if (r["id"], x) not in done]
            if not sites:
                continue
            ids, toks = render(msgs(cond, r["question"]), r)
            ts = {x: first_mention_tok(toks, r["top"], x) for x in sites}
            items = [(x, p, L) for x, t in ts.items() if t is not None for p in range(t - 5, t) for L in COT_LAYERS[x]]
            if not items:
                continue
            res = read(ids, [(p, L) for _, p, L in items])
            with path.open("a") as f:
                for (x, p, L), y in zip(items, res):
                    t = ts[x]
                    f.write(
                        json.dumps(
                            {
                                "sid": r["id"],
                                "site": x,
                                "cond": cond,
                                "probe": probe,
                                "top": r["top"],
                                "mention_tok": t,
                                "ctx": "".join(toks[t - 12 : t + 3]),
                                "pos": p,
                                "tok": toks[p],
                                "layer": L,
                                **y[0],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            print(r["id"], r["top"], {x: repr("".join(toks[t - 6 : t + 2])) for x, t in ts.items() if t is not None}, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "header":
        header()
    elif sys.argv[1] == "header2":
        header2()
    else:
        conds = sys.argv[2].split(",") if len(sys.argv) > 2 else ["TM_uswest", "TM", "none", "id_gemini", "ctl_tools", "TM_hangzhou"]
        per = int(sys.argv[3]) if len(sys.argv) > 3 else 6
        cot([(c, p) for c in conds for p in ("assistant_for_mom", "pair_personal")], per)
