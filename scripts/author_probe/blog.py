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
    cells: dict[tuple, list] = {}
    for bid, g in ok.groupby("id"):
        r = g.iloc[rng.randrange(len(g))]
        cells.setdefault((r.gender, band(r.age)), []).append(r)
    per = a.n // len(cells)
    passages, questions = [], []
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
    with open(run / "inputs_answer.jsonl", "w") as f:
        for p in load(BANK / "passages.jsonl"):
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
    for q in load(BANK / "questions.jsonl"):
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
    logp = [r for part in sorted((REPO_ROOT / "data" / "probe_cache" / run.name).glob("part-*")) for r in load(part / "logp.jsonl")]
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


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("csv")
    b.add_argument("--n", type=int, default=2000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--upload", action="store_true")
    for name in ("prompts", "answers", "del-analyze"):
        sub.add_parser(name).add_argument("run")
    d = sub.add_parser("del-prep")
    d.add_argument("run")
    d.add_argument("--src", required=True)
    d.add_argument("--n-right", type=int, default=300)
    a = ap.parse_args()
    {"build": build, "prompts": prompts, "answers": answers, "del-prep": del_prep, "del-analyze": del_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
