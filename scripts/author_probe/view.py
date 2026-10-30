"""expview items for the author MCQ run: one item per passage, with Qwen's answer prompt and letter pmf for each
attribute (both option orders) and DSv4 Flash's replies, labels = dataset label / Qwen answer / DSv4 answer per attribute.

    uv run python -m scripts.author_probe.view RUN [--ids-every 7]   # RUN/expview/ (all passages)
"""

import argparse
import json
import sys
from pathlib import Path

from scripts.author_probe.bank import ATTRS, BANK, LETTERS, answer_message, qf_message
from scripts.author_probe.dsv4 import MODEL as DSV4

sys.path.insert(0, "expview")
from expview import Run, call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--qfirst", action="store_true", help="question-first prompts (bank.qf_message); stability from ablate-loo-qfirst")
    a = ap.parse_args()
    run = Path(a.run)
    load = lambda p: [json.loads(line) for line in Path(p).read_text().splitlines()]
    passages = load(BANK / "passages.jsonl")
    qs = {q["id"]: q for q in load(BANK / "questions.jsonl")}
    qwen = {r["id"]: r for r in load(run / "qwen_answers.jsonl")}
    dsv4 = {r["id"]: r for r in load(run / "dsv4_answers.jsonl")} if (run / "dsv4_answers.jsonl").exists() else {}
    abl = (
        run.parent / ("2026-10-30_ablate-loo-qfirst" if a.qfirst else "2026-10-30_ablate-loo") / "pairs_scored.jsonl"
    )  # stable = median single-comment-deletion TV < 0.05
    stable = {f"{r['passage']}:{r['attr']}": r["stable"] for r in load(abl)} if abl.exists() else {}
    ex = Run(
        run / "expview",
        title="SynthPAI author MCQs: Qwen3.6-27B letter pmf" + (" (question first)" if a.qfirst else ""),
        description=(
            "One item per passage (random comments of one SynthPAI author). For each attribute, Qwen3.6-27B (thinking off) "
            "is asked the question in two option orders; the 'reply' shown is its letter probabilities at the answer position "
            "(renormalised over the option letters), not a sampled generation. Labels: `<attr>` = dataset label, "
            "`<attr>_qwen` = argmax of the order-averaged pmf, `<attr>_p` = Qwen's probability on the label, "
            "`<attr>_dsv4` = DeepSeek V4 Flash's answer (order 0, effort medium). `<attr>_qwen_correct` right/wrong vs the label; "
            "`<attr>_confident` = chance-rescaled top probability >= 0.5 and both orders agree; `<attr>_stable` = median "
            "single-comment-deletion TV < 0.05 (yes / no; n/a = not in the ablation run, which covers one confident passage per author)."
        ),
    )
    ex.reset()
    for p in passages:
        calls, labels = [], {}
        for attr in ATTRS:
            q, r = qs[f"{p['id']}:{attr}"], qwen[f"{p['id']}:{attr}"]
            opts = q["options"]
            labels |= {attr: opts[q["label"]], f"{attr}_qwen": opts[r["answer"]], f"{attr}_p": round(r["pmf"][q["label"]], 3)}
            K = len(opts)
            conf = (max(r["pmf"]) - 1 / K) / (1 - 1 / K) >= 0.5 and r["order_agree"]
            st = stable.get(f"{p['id']}:{attr}")
            labels |= {
                f"{attr}_qwen_correct": "right" if r["answer"] == q["label"] else "wrong",
                f"{attr}_confident": conf,
                f"{attr}_stable": "n/a" if st is None else "yes" if st else "no",
            }
            for o, rec in enumerate(r["orders"]):
                order = list(range(len(opts))) if o == 0 else list(range(len(opts)))[::-1]
                shown = [opts[i] for i in order]
                pm = "\n".join(f"{LETTERS[j]}. {shown[j]}: {rec['pmf'][i]:.3f}" for j, i in enumerate(order))
                msg = qf_message(p["text"], attr, shown) if a.qfirst else answer_message(p["text"], q["question"], shown)
                calls.append(
                    call(
                        f"{attr} order {o} (Qwen letter pmf)",
                        "Qwen/Qwen3.6-27B",
                        messages=[{"role": "user", "content": msg}],
                        output={"role": "assistant", "content": pm},
                        parsed={o2: round(v, 3) for o2, v in zip(opts, rec["pmf"], strict=True)},
                    )
                )
            d = dsv4.get(f"{p['id']}:{attr}:o0")
            if d:
                labels[f"{attr}_dsv4"] = opts[d["option"]] if d["option"] is not None else "unparsed"
                msg = answer_message(p["text"], q["question"], opts)
                calls.append(
                    call(
                        f"{attr} order 0 (DSv4 Flash)",
                        DSV4,
                        messages=[{"role": "user", "content": msg}],
                        output={"role": "assistant", "content": d["content"]},
                        parsed={"answer": labels[f"{attr}_dsv4"]},
                    )
                )
        ex.add(id=p["id"], group=p["author"], labels=labels, calls=calls)
    print(f"{len(passages)} items in {ex.path}")


def pairs(a):
    """Question-first run, one item per (passage, attribute): Qwen's prompts and pmf (both orders), the comment-deletion
    table (evidence comments in bold) when the pair was ablated, and the O-lens reads of the prompt inline (tokens=, every
    bullet with its FVE, by layer) when READS has them."""
    from expview import read

    run, reads_dir = Path(a.run), Path(a.reads)
    load = lambda p: [json.loads(line) for line in Path(p).read_text().splitlines()]
    passages = {p["id"]: p for p in load(BANK / "passages.jsonl")}
    qwen = {r["id"]: r for r in load(run / "qwen_answers.jsonl")}
    abl = {
        f"{r['passage']}:{r['attr']}": r
        for d in ("2026-10-30_ablate-loo-qfirst", "2026-10-30_ablate-loo-qfirst-b")
        for r in load(run.parent / d / "pairs_scored.jsonl")
    }
    trans = {r["id"]: r for r in load(reads_dir / "reads.transcripts.jsonl")}
    by_id: dict[str, list[dict]] = {}
    for r in load(reads_dir / "reads.jsonl"):
        by_id.setdefault(r["id"], []).append(r)
    ex = Run(
        run / "expview" / "pairs",
        title="SynthPAI author MCQs, question first: per question, with O-lens reads and comment deletions",
        description=(
            "One item per (passage, attribute). Calls: Qwen3.6-27B's question-first prompt in both option orders with its letter "
            "probabilities (not a generation), and, for ablated pairs, every comment's deletion effect (TV, change in p(answer)); "
            "**bold** = evidence comment, drop in p(answer) > max(0.15, 5 x the pair's median). The token view shows the O-lens "
            "reads (space lens, layers 32/40/48, every punctuation and newline token of the question and comments) when the pair "
            "has them. Labels: confident (c >= 0.5, orders agree), stable (median deletion TV < 0.05; n/a = not ablated), "
            "qwen_correct, has_evidence, has_reads."
        ),
        label_order=["attribute", "qwen_correct", "confident", "stable", "has_evidence", "has_reads"],
    )
    ex.reset()
    for pid, p in passages.items():
        for attr, (_, opts) in ATTRS.items():
            key = f"{pid}:{attr}"
            r, ab = qwen[key], abl.get(key)
            K = len(opts)
            calls = []
            for o, rec in enumerate(r["orders"]):
                order = list(range(K)) if o == 0 else list(range(K))[::-1]
                shown = [opts[i] for i in order]
                pm = "\n".join(f"{LETTERS[j]}. {shown[j]}: {rec['pmf'][i]:.3f}" for j, i in enumerate(order))
                msgs = [{"role": "user", "content": qf_message(p["text"], attr, shown)}]
                calls.append(
                    call(f"order {o} (Qwen letter pmf)", "Qwen/Qwen3.6-27B", messages=msgs, output={"role": "assistant", "content": pm})
                )
            n_ev = 0
            if ab:
                drops = [-d["qwen_shift"] for d in ab["deletions"]]
                thr = max(0.15, 5 * sorted(drops)[len(drops) // 2])
                rows = ["| # | drop in p(answer) | TV | human guess | comment |", "|---|---|---|---|---|"]
                for d, x in zip(ab["deletions"], drops, strict=True):
                    b = "**" if x > thr else ""
                    n_ev += x > thr
                    text = d["text"].replace("\n", " ").replace("|", "/")
                    rows.append(f"| {d['i']} | {b}{x:+.3f}{b} | {d['tv']:.3f} | {d['human']['estimate'] or '-'} | {b}{text}{b} |")
                head = f"Qwen: {opts[ab['answer']]} (p {ab['pmf'][ab['answer']]:.2f}); median deletion TV {ab['median_tv']:.3f}; "
                head += f"evidence threshold {thr:.2f}\n\n"
                calls.append(call("comment deletions", "Qwen/Qwen3.6-27B", messages=[{"role": "user", "content": "(deletion table)"}],
                                  output={"role": "assistant", "content": head + "\n".join(rows)}))  # fmt: skip
            tokens = None
            if key in trans and key in by_id:
                rd: dict[str, list] = {}
                for x in sorted(by_id[key], key=lambda x: (x["token_index"], x["layer"])):
                    lines = [[b["text"], b["fve"]] for b in sorted(x["bullets"], key=lambda b: -b["fve"])]
                    rd.setdefault(str(x["token_index"]), []).append(
                        read("O-lens", lines, label=f"L{x['layer']}", score=x["joint_fve"], score_name="joint FVE")
                    )
                tokens = {"tokens": trans[key]["tokens"], "sections": trans[key]["sections"], "reads": rd}
            conf = (max(r["pmf"]) - 1 / K) / (1 - 1 / K) >= 0.5 and r["order_agree"]
            ex.add(
                id=key,
                group=attr,
                labels={
                    "attribute": attr,
                    "label": opts[r["label"]],
                    "qwen": opts[r["answer"]],
                    "qwen_correct": "right" if r["answer"] == r["label"] else "wrong",
                    "confident": conf,
                    "stable": "n/a" if not ab else "yes" if ab["stable"] else "no",
                    "has_evidence": n_ev > 0,
                    "has_reads": tokens is not None,
                },
                calls=calls,
                tokens=tokens,
            )
    print("items in", ex.path)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "pairs":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("run")
    ap.add_argument("--reads", required=True)
    pairs(ap.parse_args())
    sys.exit()


if __name__ == "__main__":
    main()
