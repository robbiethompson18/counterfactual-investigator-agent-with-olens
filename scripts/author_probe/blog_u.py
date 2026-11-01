"""The uniform version of the blog-author eval (Kylie, 2026-10-31: "run through a big pile of blog posts, filter on there is
evidence and qwen is confident"). No stratification, no Qwen-right cap, no weights:

  1. a seeded uniform draw of N posts from every eligible post in the bank (blog.py build's length / ASCII / urlLink filters,
     one post per blogger; all four batches, so the natural sex x age mix of the eligible pool);
  2. both questions; keep the pairs Qwen answers confidently (c >= 0.5) with both option orders agreeing;
  3. every confident pair's leave-one-sentence-out deletions, reusing earlier runs' log-probs where the same prompt exists
     (batches 1-2 both orders, batch 3 partly) and running the rest in one option order (o0);
  4. a pair is in the eval if some deletion drops Qwen's probability on its answer by more than 0.15; every such sentence is a
     positive, plus as many seeded random other sentences of the post as negatives; stable pairs only (median deletion TV
     < 0.05; Kylie: "stability filter is good").

The run actually used (2026-10-31_uniform-sample) replaces steps 1-3's fresh draw with cell-level rejection sampling of the
pairs batches 1-3 had already deleted plus 461 new Qwen-right pairs drawn uniformly within their cells (its plan.json);
`cases` reads that run's pairs.jsonl either way.

    uv run --with numpy --with transformers --with jinja2 python -m scripts.author_probe.blog_u prep RUN [--n 2000]
    uv run --with numpy --with transformers --with jinja2 python -m scripts.author_probe.blog_u cases RUN LOGP_DIR...
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

from scripts.author_probe.bank import LETTERS, MODEL, orders, read_positions
from scripts.author_probe.blog import ATTRS, BANK, _row, conf, join, sentences
from scripts.author_probe.blog import message as qf_message
from scripts.author_probe.period_probe import sites

ROOT = Path("data/runs/author_mcq_blog")
ANSWERS = [ROOT / f"2026-10-30_qwen36-qfirst{s}/qwen_answers.jsonl" for s in ("", "-b", "-c")]
OLD_INPUTS = [ROOT / f"2026-10-30_ablate-loo{s}/inputs_del_answer.jsonl" for s in ("", "-b", "-c")]
OLD_LOGP = [Path("data/probe_cache/2026-10-30_ablate-loo"), Path("data/probe_cache/2026-10-30_ablate-loo-b"),
            ROOT / "2026-10-30_ablate-loo-c/answer_logp_partial.jsonl"]  # fmt: skip
THR = 0.15


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in Path(p).read_text().splitlines()]


def logps(extra: list[str]) -> dict[str, list[float]]:
    out = {}
    for p in [*OLD_LOGP, *map(Path, extra)]:
        files = [p] if p.is_file() else [Path(d) / "logp.jsonl" for d, _, fs in os.walk(p, followlinks=True) if "logp.jsonl" in fs]
        for f in files:
            out |= {r["id"]: r["logp"] for r in load(f)}
    return out


def prep(a):
    from transformers import AutoTokenizer

    run, tok = Path(a.run), AutoTokenizer.from_pretrained(MODEL)
    passages = load(BANK / "passages.jsonl")
    pick = {p["id"] for p in random.Random(a.seed).sample(passages, a.n)}
    answers = {r["id"]: r for f in ANSWERS for r in load(f)}
    pairs = [r for r in answers.values() if r["passage"] in pick and conf(r["pmf"]) >= 0.5 and r["order_agree"]]
    have = logps([])
    by = {p["id"]: p for p in passages}
    todo, n_reused = [], 0
    for r in pairs:
        sents = by[r["passage"]]["sentences"]
        for i in range(-1, len(sents)):
            rid = f"{r['id']}~{i}:o0"
            if rid in have:
                n_reused += 1
                continue
            text = join([s for j, s in enumerate(sents) if j != i])
            todo.append(_row(tok, rid, text, r["attr"], orders(len(ATTRS[r["attr"]]))[0]))
    (run / "pairs.jsonl").write_text("".join(json.dumps({"id": r["id"], "passage": r["passage"], "attr": r["attr"],
                                                         "n": len(by[r["passage"]]["sentences"])}) + "\n" for r in pairs))  # fmt: skip
    (run / "inputs_del_answer.jsonl").write_text("".join(json.dumps(r) + "\n" for r in todo))
    print(f"{len(pick)} posts, {len(pairs)} confident pairs; {n_reused} o0 deletion prompts reused, {len(todo)} to run")


def cases(a):
    run = Path(a.run)
    order_of = {r["id"]: r["order"] for f in [*OLD_INPUTS, run / "inputs_del_answer.jsonl"] for r in load(f)}
    lp = logps(a.logp)

    def pmf(key):
        ps = []
        for o in (0, 1):
            i = f"{key}:o{o}"
            if i in lp:
                x = np.exp(np.array(lp[i]) - max(lp[i]))
                c = np.zeros(len(x))
                c[order_of[i]] = x / x.sum()
                ps.append(c)
        return sum(ps) / len(ps) if ps else None

    passages = {p["id"]: p for p in load(BANK / "passages.jsonl")}
    answers = {r["id"]: r for f in ANSWERS for r in load(f)}
    out, stats = [], {"pairs": 0, "missing": 0, "with_positive": 0}
    for pr in load(run / "pairs.jsonl"):
        stats["pairs"] += 1
        full = pmf(f"{pr['id']}~-1")
        dels = [pmf(f"{pr['id']}~{i}") for i in range(pr["n"])]
        if full is None or any(d is None for d in dels):
            stats["missing"] += 1
            continue
        ans = int(full.argmax())
        drops = [float(full[ans] - d[ans]) for d in dels]
        tv = [0.5 * float(np.abs(d - full).sum()) for d in dels]
        pos = [i for i, x in enumerate(drops) if x > THR]
        if not pos or np.median(tv) >= 0.05:  # stability filter (Kylie, 2026-10-31: "stability filter is good")
            continue
        stats["with_positive"] += 1
        q = answers[pr["id"]]
        version = "qwen_wrong" if q["answer"] != q["label"] else "qwen_right"
        neg = [i for i, x in enumerate(drops) if x <= THR]
        sents = sentences(passages[pr["passage"]]["text"])
        for i in pos + random.Random(pr["id"]).sample(neg, min(len(pos), len(neg))):
            out.append({"version": version, "item": f"{pr['id']}#{i}", "pair": pr["id"], "attr": pr["attr"], "comment": sents[i],
                        "positive": drops[i] > THR, "threshold": THR, "answer": LETTERS[ans], "label": LETTERS[q["label"]],
                        "p0": float(full[ans]), "p1": float(dels[i][ans]), "drop": drops[i], "tv": tv[i],
                        "flip": int(dels[i].argmax()) != ans, "stable": bool(np.median(tv) < 0.05), "batch": "uniform",
                        "prompt": qf_message(passages[pr["passage"]]["text"], pr["attr"], ATTRS[pr["attr"]])})  # fmt: skip
    (run / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in out))
    print(stats, f"{len(out)} cases ({sum(c['positive'] for c in out)} positive), {len({c['pair'] for c in out})} pairs, "
          f"{len({c['pair'].split(':')[0] for c in out})} posts, stable {sum(c['stable'] for c in out)}")  # fmt: skip


def reads_inputs(a):
    """O-lens read requests and probe/NLA capture inputs for the eval pairs in RUN/cases.jsonl that SKIP's runs don't cover."""
    from transformers import AutoTokenizer

    run, tok = Path(a.run), AutoTokenizer.from_pretrained(MODEL)
    have = {r["id"] for d in a.skip for r in load(Path(d) / "requests.jsonl")}
    passages = {p["id"]: p for p in load(BANK / "passages.jsonl")}
    pairs = sorted({c["pair"] for c in load(run / "cases.jsonl")} - have)
    req, hs, idx = [], [], []
    for pair in pairs:
        pid, attr = pair.split(":")
        user = qf_message(passages[pid]["text"], attr)
        msgs = [{"role": "user", "content": user}]
        ids, pos = read_positions(tok, tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False), user)
        req.append({"id": pair, "messages": msgs, "tools": None, "prompt_ids": ids, "positions": pos, "env": "author_blog", "arm": attr})
        hids, hpos, kinds = sites(tok, passages[pid]["text"], attr)
        hs.append({"id": pair, "ids": hids, "sites": {"h": hpos}, "cand": []})
        idx.append({"id": pair, "split": "eval", "kinds": kinds, "positions": hpos})
    w = lambda name, rows: (run / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    w("read_requests.jsonl", req)
    w("nla_inputs.jsonl", [{"id": r["id"], "ids": r["prompt_ids"], "sites": {"h": r["positions"]}, "cand": []} for r in req])
    w("hs_inputs.jsonl", hs)
    w("index.jsonl", idx)
    print(f"{len(pairs)} pairs need reads ({sum(len(r['positions']) for r in req)} read positions)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("run")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("cases")
    c.add_argument("run")
    c.add_argument("logp", nargs="*")
    r = sub.add_parser("reads-inputs")
    r.add_argument("run")
    r.add_argument("skip", nargs="*")
    a = ap.parse_args()
    {"prep": prep, "cases": cases, "reads-inputs": reads_inputs}[a.cmd](a)


if __name__ == "__main__":
    main()
