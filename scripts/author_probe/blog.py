"""Blog Authorship Corpus version of the author MCQ pipeline (Kylie, 2026-10-30): coherent prose instead of SynthPAI's
one-line comments, question-first prompts, the sentence as the deletion unit.

    uv run --with pandas python -m scripts.author_probe.blog build CSV [--n 2000] [--upload]   # bank: data/banks/author_mcq_blog/
    uv run --with transformers --with jinja2 python -m scripts.author_probe.blog prompts RUN    # RUN/inputs_answer.jsonl (both orders)
    uv run python -m scripts.author_probe.blog answers RUN                                      # RUN/qwen_answers.jsonl, qwen.md
    uv run --with transformers --with jinja2 python -m scripts.author_probe.blog del-prep RUN --src SRC   # leave-one-sentence-out
    uv run python -m scripts.author_probe.blog del-analyze RUN                                  # pairs_scored.jsonl, ablate.md

Source: Schler et al. (2006), https://huggingface.co/datasets/tasksource/blog_authorship_corpus (blogtext.csv: 681k posts by
19,320 bloggers with self-reported gender and age; ages come in three groups, 13-17 / 23-27 / 33-48). A passage is one post
of 200-450 words (mostly ASCII, no "urlLink" stubs), one per blogger, balanced over gender x age group. Questions: sex
(Male / Female) and age (Under 20 / 20-29 / 30 or older, which the three groups map onto exactly). Sentences: a regex split
on ., !, ? and newlines; deleting a sentence removes it and keeps the rest's spacing.
"""

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np

from environments._lib.runfolder import REPO_ROOT, hf_token
from scripts.author_probe.bank import HF_REPO, LETTERS, MODEL, orders

BANK = REPO_ROOT / "data" / "banks" / "author_mcq_blog"
ATTRS = {"sex": ["Male", "Female"], "age": ["Under 20", "20-29", "30 or older"]}
QUESTIONS = {
    "sex": "What is my sex, based on the blog post I wrote below?",
    "age": "How old am I most likely to be, based on the blog post I wrote below?",
}
END = "What do you think?"
SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in Path(p).read_text().splitlines()]


def message(text: str, attr: str, options: list[str] | None = None) -> str:
    out = f"{QUESTIONS[attr]}\n\n{text}\n\n{END}"
    if options:
        out += "\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options)) + "\n\nAnswer with the letter only."
    return out


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SPLIT.split(text) if s.strip()]


def join(sents: list[str]) -> str:
    return " ".join(sents)


def build(a):
    import pandas as pd

    df = pd.read_csv(a.csv)
    df["text"] = df.text.str.strip().str.replace(r"\s+", " ", regex=True)
    df["words"] = df.text.str.split().str.len()
    ok = df[(df.words >= 200) & (df.words <= 450) & ~df.text.str.contains("urlLink")]
    ok = ok[ok.text.map(lambda t: sum(c.isascii() for c in t) / len(t) > 0.98)]
    band = lambda age: 0 if age < 20 else 1 if age < 30 else 2
    rng = random.Random(a.seed)
    old_p = load(BANK / "passages.jsonl") if a.append else []  # --append: new bloggers only, added to the bank
    old_q = load(BANK / "questions.jsonl") if a.append else []
    taken = {p["author"] for p in old_p}
    cells: dict[tuple, list] = {}
    for bid, g in ok[~ok.id.astype(str).isin(taken)].groupby("id"):
        r = g.iloc[rng.randrange(len(g))]
        cells.setdefault((r.gender, band(r.age)), []).append(r)
    per = a.n // len(cells)
    passages, questions = list(old_p), list(old_q)
    for (gender, b), rows in sorted(cells.items()):
        for r in rng.sample(rows, min(per, len(rows))):
            sents = sentences(r.text)
            pid = f"blog{r.id}"
            passages.append({"id": pid, "author": str(r.id), "text": join(sents), "sentences": sents, "n_words": int(r.words),
                             "profile": {"gender": r.gender, "age": int(r.age), "topic": r.topic, "date": r.date}})  # fmt: skip
            labels = {"sex": ["male", "female"].index(r.gender), "age": b}
            for attr, opts in ATTRS.items():
                questions.append({"id": f"{pid}:{attr}", "passage": pid, "author": str(r.id), "attr": attr, "question": QUESTIONS[attr],
                                  "options": opts, "label": labels[attr]})  # fmt: skip
    BANK.mkdir(parents=True, exist_ok=True)
    (BANK / "passages.jsonl").write_text("".join(json.dumps(p) + "\n" for p in passages))
    (BANK / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions))
    ns = [len(p["sentences"]) for p in passages]
    print(f"{len(passages)} passages ({per} per gender x age cell; cells {[(k, len(v)) for k, v in sorted(cells.items())]})")
    print(f"sentences per passage: median {np.median(ns)}, total {sum(ns)}")
    if a.upload:
        from huggingface_hub import HfApi

        for f in ("passages.jsonl", "questions.jsonl"):
            HfApi(token=hf_token()).upload_file(
                path_or_fileobj=BANK / f, path_in_repo=f"banks/author_mcq_blog/{f}", repo_id=HF_REPO, repo_type="dataset"
            )


def _letter_ids(tok) -> list[int]:
    return [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]


def _row(tok, rid: str, text: str, attr: str, order: list[int]) -> dict:
    opts = ATTRS[attr]
    user = message(text, attr, [opts[i] for i in order])
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return {
        "id": rid,
        "ids": tok(rendered, add_special_tokens=False)["input_ids"],
        "sites": {},
        "cand": _letter_ids(tok)[: len(order)],
        "order": order,
    }


def prompts(a):
    from transformers import AutoTokenizer

    tok, run = AutoTokenizer.from_pretrained(MODEL), Path(a.run)
    done = {r["passage"] for x in a.skip for r in load(Path(x) / "qwen_answers.jsonl")}  # passages answered in earlier runs
    with open(run / "inputs_answer.jsonl", "w") as f:
        for p in (p for p in load(BANK / "passages.jsonl") if p["id"] not in done):
            for attr, opts in ATTRS.items():
                f.writelines(
                    json.dumps(_row(tok, f"{p['id']}:{attr}:o{o}", p["text"], attr, order)) + "\n"
                    for o, order in enumerate(orders(len(opts)))
                )


def pmfs(run: Path, logp_rows) -> dict[str, np.ndarray]:
    """Order-averaged letter pmf in option space per prompt id without its `:oN` suffix."""
    order_of = {
        r["id"]: r["order"]
        for r in load(run / ("inputs_answer.jsonl" if (run / "inputs_answer.jsonl").exists() else "inputs_del_answer.jsonl"))
    }
    out: dict[str, np.ndarray] = {}
    for r in logp_rows:
        lp = np.array(r["logp"])
        p = np.exp(lp - lp.max())
        canon = np.zeros(len(p))
        canon[order_of[r["id"]]] = p / p.sum()
        key = r["id"].rsplit(":o", 1)[0]
        out[key] = out.get(key, 0) + canon / 2
    return out


def conf(pmf) -> float:
    K = len(pmf)
    return float((max(pmf) - 1 / K) / (1 - 1 / K))


def answers(a):
    run = Path(a.run)
    raw = {r["id"]: r for r in load(run / "answer_logp.jsonl")}
    pm = pmfs(run, raw.values())
    rows, lines = (
        [],
        [
            "| attribute | n | Qwen acc | majority | Qwen answers | labels | confident (c >= 0.5, orders agree) |",
            "|---|---|---|---|---|---|---|",
        ],
    )
    for q in (q for q in load(BANK / "questions.jsonl") if q["id"] in pm):  # this run's passages
        p = pm[q["id"]]
        picks = [orders(len(q["options"]))[o][int(np.argmax(raw[f"{q['id']}:o{o}"]["logp"]))] for o in (0, 1)]
        agree = picks[0] == picks[1]
        rows.append(
            {k: q[k] for k in ("id", "passage", "author", "attr", "label")}
            | {"pmf": p.round(4).tolist(), "answer": int(p.argmax()), "order_agree": bool(agree)}
        )
    for attr, opts in ATTRS.items():
        rs = [r for r in rows if r["attr"] == attr]
        acc = np.mean([r["answer"] == r["label"] for r in rs])
        maj = max(sum(r["label"] == k for r in rs) for k in range(len(opts))) / len(rs)
        dist = [sum(r["answer"] == k for r in rs) for k in range(len(opts))]
        lab = [sum(r["label"] == k for r in rs) for k in range(len(opts))]
        nconf = sum(conf(r["pmf"]) >= 0.5 and r["order_agree"] for r in rs)
        lines.append(f"| {attr} | {len(rs)} | {acc:.3f} | {maj:.3f} | {dist} | {lab} | {nconf} |")
    (run / "qwen_answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "qwen.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def del_prep(a):
    """Leave-one-sentence-out answer prompts (both orders) for confident pairs of SRC (c >= 0.5, orders agree): all the
    Qwen-wrong ones and a seeded sample of Qwen-right ones."""
    from transformers import AutoTokenizer

    tok, run = AutoTokenizer.from_pretrained(MODEL), Path(a.run)
    passages = {p["id"]: p for p in load(BANK / "passages.jsonl")}
    confident = [r for r in load(Path(a.src) / "qwen_answers.jsonl") if conf(r["pmf"]) >= 0.5 and r["order_agree"]]
    rng = random.Random(0)  # every confident Qwen-wrong pair, plus --n-right Qwen-right pairs per attribute (the GPU budget)
    pairs = [r for r in confident if r["answer"] != r["label"]]
    for attr in ATTRS:
        right = [r for r in confident if r["attr"] == attr and r["answer"] == r["label"]]
        pairs += rng.sample(right, min(a.n_right, len(right)))
    with open(run / "inputs_del_answer.jsonl", "w") as f:
        for r in pairs:
            sents = passages[r["passage"]]["sentences"]
            for i in range(-1, len(sents)):
                text = join([s for j, s in enumerate(sents) if j != i])
                f.writelines(
                    json.dumps(_row(tok, f"{r['id']}~{i}:o{o}", text, r["attr"], order)) + "\n"
                    for o, order in enumerate(orders(len(ATTRS[r["attr"]])))
                )
    (run / "pairs.jsonl").write_text(
        "".join(
            json.dumps({"id": r["id"], "passage": r["passage"], "attr": r["attr"], "n": len(passages[r["passage"]]["sentences"])}) + "\n"
            for r in pairs
        )
    )
    print(f"{len(pairs)} confident pairs, {sum(len(passages[r['passage']]['sentences']) + 1 for r in pairs) * 2} prompts")


def del_analyze(a):
    run = Path(a.run)
    passages = {p["id"]: p for p in load(BANK / "passages.jsonl")}
    labels = {q["id"]: q["label"] for q in load(BANK / "questions.jsonl")}
    parts = sorted((REPO_ROOT / "data" / "probe_cache" / run.name).glob("*part-*"))
    logp = {r["id"]: r for part in parts for r in load(part / "logp.jsonl")}
    logp = list(logp.values())  # a chunk computed on two pods counts once
    pm = pmfs(run, logp)
    rows: list[dict] = []
    for pr in load(run / "pairs.jsonl"):
        full = pm[f"{pr['id']}~-1"]
        ans = int(full.argmax())
        dels: list[dict] = []
        for i, s in enumerate(passages[pr["passage"]]["sentences"]):
            p = pm[f"{pr['id']}~{i}"]
            dels.append({"i": i, "text": s, "tv": float(0.5 * np.abs(p - full).sum()), "qwen_shift": float(p[ans] - full[ans]),
                         "argmax_kept": int(p.argmax()) == ans, "pmf": p.round(4).tolist()})  # fmt: skip
        med = float(np.median([d["tv"] for d in dels]))
        drops = sorted(-d["qwen_shift"] for d in dels)
        thr = max(0.15, 5 * drops[len(drops) // 2])
        for d in dels:
            d["evidence"] = -d["qwen_shift"] > thr
        rows.append(pr | {"label": labels[pr["id"]], "answer": ans, "pmf": full.round(4).tolist(), "median_tv": med, "stable": med < 0.05,
                          "threshold": thr, "deletions": dels})  # fmt: skip
    (run / "pairs_scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    lines = [
        "| attribute | confident pairs | stable | stable with evidence | stable, Qwen wrong | stable, Qwen right |",
        "|---|---|---|---|---|---|",
    ]
    for attr in ATTRS:
        rs = [r for r in rows if r["attr"] == attr]
        st = [r for r in rs if r["stable"]]
        lines.append(f"| {attr} | {len(rs)} | {len(st)} | {sum(any(d['evidence'] for d in r['deletions']) for r in st)} | "
                     f"{sum(r['answer'] != r['label'] for r in st)} | {sum(r['answer'] == r['label'] for r in st)} |")  # fmt: skip
    (run / "ablate.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def view(a):
    """expview, one item per (passage, attribute): Qwen's prompts and pmf, the sentence-deletion table (evidence in bold)
    for ablated pairs, and the O-lens reads inline (tokens=) where --reads has them."""
    import sys

    sys.path.insert(0, "expview")
    from expview import Run, call, read

    run = Path(a.run)
    passages = {p["id"]: p for p in load(BANK / "passages.jsonl")}
    qwen = {r["id"]: r for r in load(run / "qwen_answers.jsonl")}
    abl = {r["id"]: r for r in load(Path(a.ablate) / "pairs_scored.jsonl")} if a.ablate else {}
    trans, by_id = {}, {}
    if a.reads:
        trans = {r["id"]: r for r in load(Path(a.reads) / "reads.transcripts.jsonl")}
        for r in load(Path(a.reads) / "reads.jsonl"):
            by_id.setdefault(r["id"], []).append(r)
    ex = Run(
        run / "expview" / "pairs",
        title="Blog Authorship Corpus author MCQs, question first",
        description="One item per (blog post, attribute): Qwen3.6-27B's question-first prompt in both option orders with its letter "
        "probabilities (not a generation), the sentence-deletion table for ablated pairs (**bold** = evidence sentence: drop in "
        "p(answer) > max(0.15, 5 x the pair's median)), and O-lens reads on the token view where read.",
        label_order=["attribute", "qwen_correct", "confident", "stable", "has_evidence", "has_reads"],
    )
    ex.reset()
    for key, r in qwen.items():
        attr, opts, p = r["attr"], ATTRS[r["attr"]], passages[r["passage"]]
        calls = []
        for o, order in enumerate(orders(len(opts))):
            shown = [opts[i] for i in order]
            pm = "\n".join(f"{LETTERS[j]}. {shown[j]}: {r['pmf'][i]:.3f}" for j, i in enumerate(order))
            calls.append(
                call(
                    f"order {o} (Qwen letter pmf, order-averaged)",
                    MODEL,
                    messages=[{"role": "user", "content": message(p["text"], attr, shown)}],
                    output={"role": "assistant", "content": pm},
                )
            )
        ab, n_ev = abl.get(key), 0
        if ab:
            rows = ["| # | drop in p(answer) | TV | sentence |", "|---|---|---|---|"]
            for d in ab["deletions"]:
                b = "**" if d["evidence"] else ""
                n_ev += d["evidence"]
                rows.append(f"| {d['i']} | {b}{-d['qwen_shift']:+.3f}{b} | {d['tv']:.3f} | {b}{d['text'].replace('|', '/')}{b} |")
            head = f"Qwen: {opts[ab['answer']]} (p {ab['pmf'][ab['answer']]:.2f}); "
            head += f"median TV {ab['median_tv']:.3f}; threshold {ab['threshold']:.2f}\n\n"
            calls.append(
                call(
                    "sentence deletions",
                    MODEL,
                    messages=[{"role": "user", "content": "(deletion table)"}],
                    output={"role": "assistant", "content": head + "\n".join(rows)},
                )
            )
        tokens = None
        if key in trans and key in by_id:
            rd: dict[str, list] = {}
            for x in sorted(by_id[key], key=lambda x: (x["token_index"], x["layer"])):
                lines = [[b["text"], b["fve"]] for b in sorted(x["bullets"], key=lambda b: -b["fve"])]
                rd.setdefault(str(x["token_index"]), []).append(
                    read("O-lens", lines, label=f"L{x['layer']}", score=x["joint_fve"], score_name="joint FVE")
                )
            tokens = {"tokens": trans[key]["tokens"], "sections": trans[key]["sections"], "reads": rd}
        ex.add(
            id=key,
            group=attr,
            tokens=tokens,
            calls=calls,
            labels={
                "attribute": attr,
                "label": opts[r["label"]],
                "qwen": opts[r["answer"]],
                "qwen_correct": ("right" if r["answer"] == r["label"] else "wrong"),
                "confident": conf(r["pmf"]) >= 0.5 and r["order_agree"],
                "stable": "n/a" if not ab else "yes" if ab["stable"] else "no",
                "has_evidence": n_ev > 0,
                "has_reads": tokens is not None,
                "gender": p["profile"]["gender"],
                "age": p["profile"]["age"],
            },
        )
    print("items in", ex.path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("csv")
    b.add_argument("--n", type=int, default=2000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--upload", action="store_true")
    b.add_argument("--append", action="store_true", help="add --n posts from bloggers not yet in the bank")
    for name in ("answers", "del-analyze"):
        sub.add_parser(name).add_argument("run")
    pr = sub.add_parser("prompts")
    pr.add_argument("run")
    pr.add_argument("--skip", nargs="*", default=[], help="runs whose passages are already answered")
    d = sub.add_parser("del-prep")
    d.add_argument("run")
    d.add_argument("--src", required=True)
    d.add_argument("--n-right", type=int, default=300)
    v = sub.add_parser("view")
    v.add_argument("run")
    v.add_argument("--ablate")
    v.add_argument("--reads")
    a = ap.parse_args()
    {"view": view, "build": build, "prompts": prompts, "answers": answers, "del-prep": del_prep, "del-analyze": del_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
