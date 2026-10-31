"""Batch 3 of the blog-author eval (Kylie, 2026-10-31: scale toward 1,000 posts, flat 15% rule). Scores batch 3's deletions
(2026-10-30_ablate-loo-c: the abandoned run's partial log-probs, both option orders, plus any single-order shards finished
later), labels each case positive when p0 - p1 > 0.15 on Qwen's answer (flat, no median term), keeps confident and stable
(median deletion TV < 0.05) pairs with at least one positive, adds 1:1 random non-significant sentences of the same post, and
writes the new cases plus the O-lens/NLA read requests and the L48 probe capture inputs for those pairs.

    uv run --with numpy --with transformers --with jinja2 python -m scripts.author_probe.blog_c RUN [SHARD_LOGP_DIR...]
"""

import json
import random
import sys
from pathlib import Path

import numpy as np

from scripts.author_probe.bank import LETTERS, MODEL, read_positions
from scripts.author_probe.blog import ATTRS, BANK, END, sentences
from scripts.author_probe.blog import message as qf_message
from scripts.author_probe.period_probe import sites

ABL = Path("data/runs/author_mcq_blog/2026-10-30_ablate-loo-c")
ANSWERS = Path("data/runs/author_mcq_blog/2026-10-30_qwen36-qfirst-c/qwen_answers.jsonl")
THR = 0.15


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in Path(p).read_text().splitlines()]


def main(run: Path, shard_dirs: list[str]):
    from transformers import AutoTokenizer

    order_of = {r["id"]: r["order"] for r in load(ABL / "inputs_del_answer.jsonl")}
    logp = {r["id"]: r["logp"] for r in load(ABL / "answer_logp_partial.jsonl")}
    for d in shard_dirs:  # later single-order shards (hs_at_positions logp.jsonl parts)
        for f in Path(d).glob("**/logp.jsonl"):
            logp |= {r["id"]: r["logp"] for r in load(f)}

    def pmf(key):
        ps = []
        for o in (0, 1):
            i = f"{key}:o{o}"
            if i in logp:
                lp = np.array(logp[i])
                p = np.exp(lp - lp.max())
                c = np.zeros(len(p))
                c[order_of[i]] = p / p.sum()
                ps.append(c)
        return (sum(ps) / len(ps), len(ps)) if ps else (None, 0)

    passages = {p["id"]: p for p in load(BANK / "passages.jsonl")}
    answers = {r["id"]: r for r in load(ANSWERS)}
    rng, cases, stats = random.Random(0), [], {}
    for pr in load(ABL / "pairs.jsonl"):
        full, nf = pmf(f"{pr['id']}~-1")
        dels = [pmf(f"{pr['id']}~{i}") for i in range(pr["n"])]
        if full is None or any(d[0] is None for d in dels):
            stats["incomplete"] = stats.get("incomplete", 0) + 1
            continue
        orders = min([nf] + [n for _, n in dels])  # 2 = both option orders, 1 = one
        ans = int(full.argmax())
        version = "qwen_wrong" if answers[pr["id"]]["answer"] != answers[pr["id"]]["label"] else "qwen_right"
        tv = [0.5 * float(np.abs(d - full).sum()) for d, _ in dels]
        if np.median(tv) >= 0.05:
            continue
        sents = sentences(passages[pr["passage"]]["text"])
        drops = [float(full[ans] - d[ans]) for d, _ in dels]
        pos = [i for i, x in enumerate(drops) if x > THR]
        neg = [i for i, x in enumerate(drops) if x <= THR]
        if not pos:
            continue
        stats[version] = stats.get(version, 0) + 1
        opts = ATTRS[pr["attr"]]
        for i in pos + rng.sample(neg, min(len(pos), len(neg))):
            d = dels[i][0]
            cases.append({"version": version, "item": f"{pr['id']}#{i}", "pair": pr["id"], "attr": pr["attr"], "comment": sents[i],
                          "positive": drops[i] > THR, "threshold": THR, "answer": LETTERS[ans],
                          "label": LETTERS[answers[pr["id"]]["label"]], "p0": float(full[ans]), "p1": float(d[ans]),
                          "drop": drops[i], "tv": tv[i], "flip": int(d.argmax()) != ans, "orders": orders, "batch": 2,
                          "prompt": qf_message(passages[pr["passage"]]["text"], pr["attr"], opts)})  # fmt: skip
    run.mkdir(parents=True, exist_ok=True)
    (run / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))
    pairs = sorted({c["pair"] for c in cases})
    tok = AutoTokenizer.from_pretrained(MODEL)
    req, hs = [], []
    for pair in pairs:
        pid, attr = pair.split(":")
        user = qf_message(passages[pid]["text"], attr)
        msgs = [{"role": "user", "content": user}]
        ids, pos = read_positions(tok, tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False), user)
        req.append({"id": pair, "messages": msgs, "tools": None, "prompt_ids": ids, "positions": pos, "env": "author_blog", "arm": attr})
        hids, hpos, kinds = sites(tok, passages[pid]["text"], attr)
        hs.append(
            ({"id": pair, "ids": hids, "sites": {"h": hpos}, "cand": []}, {"id": pair, "split": "eval", "kinds": kinds, "positions": hpos})
        )
    (run / "read_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in req))
    (run / "hs_inputs.jsonl").write_text("".join(json.dumps(a) + "\n" for a, _ in hs))
    (run / "index.jsonl").write_text("".join(json.dumps(b) + "\n" for _, b in hs))
    assert END in req[0]["messages"][0]["content"]
    print(stats, f"{len(cases)} cases ({sum(c['positive'] for c in cases)} positive) from {len(pairs)} pairs")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
