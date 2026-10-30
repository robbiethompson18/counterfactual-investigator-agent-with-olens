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


if __name__ == "__main__":
    main()
