"""expview items for the author MCQ run: one item per passage, with Qwen's answer prompt and letter pmf for each
attribute (both option orders) and DSv4 Flash's replies, labels = dataset label / Qwen answer / DSv4 answer per attribute.

    uv run python -m scripts.author_probe.view RUN [--ids-every 7]   # RUN/expview/ (all passages)
"""

import argparse
import json
import sys
from pathlib import Path

from scripts.author_probe.bank import ATTRS, BANK, LETTERS, answer_message
from scripts.author_probe.dsv4 import MODEL as DSV4

sys.path.insert(0, "expview")
from expview import Run, call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    a = ap.parse_args()
    run = Path(a.run)
    load = lambda p: [json.loads(line) for line in Path(p).read_text().splitlines()]
    passages = load(BANK / "passages.jsonl")
    qs = {q["id"]: q for q in load(BANK / "questions.jsonl")}
    qwen = {r["id"]: r for r in load(run / "qwen_answers.jsonl")}
    dsv4 = {r["id"]: r for r in load(run / "dsv4_answers.jsonl")} if (run / "dsv4_answers.jsonl").exists() else {}
    ex = Run(
        run / "expview",
        title="SynthPAI author MCQs: Qwen3.6-27B letter pmf",
        description=(
            "One item per passage (random comments of one SynthPAI author). For each attribute, Qwen3.6-27B (thinking off) "
            "is asked the question in two option orders; the 'reply' shown is its letter probabilities at the answer position "
            "(renormalised over the option letters), not a sampled generation. Labels: `<attr>` = dataset label, "
            "`<attr>_qwen` = argmax of the order-averaged pmf, `<attr>_p` = Qwen's probability on the label, "
            "`<attr>_dsv4` = DeepSeek V4 Flash's answer (order 0, effort medium)."
        ),
    )
    ex.reset()
    for p in passages:
        calls, labels = [], {}
        for attr in ATTRS:
            q, r = qs[f"{p['id']}:{attr}"], qwen[f"{p['id']}:{attr}"]
            opts = q["options"]
            labels |= {attr: opts[q["label"]], f"{attr}_qwen": opts[r["answer"]], f"{attr}_p": round(r["pmf"][q["label"]], 3)}
            for o, rec in enumerate(r["orders"]):
                order = list(range(len(opts))) if o == 0 else list(range(len(opts)))[::-1]
                shown = [opts[i] for i in order]
                pm = "\n".join(f"{LETTERS[j]}. {shown[j]}: {rec['pmf'][i]:.3f}" for j, i in enumerate(order))
                msg = answer_message(p["text"], q["question"], shown)
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
