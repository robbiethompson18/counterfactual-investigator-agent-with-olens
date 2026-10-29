"""The MCQ answer probe as a baseline on cfbench's mcq_unselected eval (docs/mcq-unselected.md, docs/mcq-answer-probe.md).

How well does a linear probe on Qwen3.6-27B's residual stream predict the letter Qwen gives, on the cases the investigator
is scored on? Training set: the split's probe half (`split()["probe"]`, 2,000 questions) under the eval's own consistency
rule (census modal letter >= MIN_SHARE of all samples), not balanced, no eval-half question. Test set: every eval case of
both versions (`cases.jsonl` of the inputs run). The probe is Kylie's chosen one (probe.py): one-hot listwise, diagonal
standardisation, the `nl` site (the newline token after each option), hidden_states[52], l2 1. Qwen's own letter-logit
argmax at the answer position is the ceiling.

    # CPU: training questions + hs_at_positions inputs for both sets (no-hint prompt, sites = option newlines, cand = letters)
    uv run --with transformers --with jinja2 python -m environments.hint_mcq.probe_unselected prep INPUTS_RUN
    # GPU: python scripts/hs_at_positions.py RUN/hs_inputs.jsonl OUT --layers 52   -> data/probe_cache/<run name>/
    # CPU: fit on the train rows, predict every case; predictions.jsonl (conditions `probe`, `qwen_logits`), scores.md, probe.npz
    uv run python -m environments.hint_mcq.probe_unselected score RUN --pod "..." --cost 8 --finish

predictions.jsonl rows are cfbench answer-task rows (the case fields of `cases()`, draw 0, cost 0), so `cfbench.answer RUN`
scores them; the env viewer only indexes runs under data/runs/cfbench/.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from cfbench.evals.mcq_unselected import eval as ev
from counterfactual_investigator_agent_with_olens.runs import finish, new_run
from environments.hint_mcq.env import LETTERS
from environments.hint_mcq.probe import CACHE, MODEL, _fit_listwise, _softmax_by_q

HIDDEN, SITE, L2 = 52, "nl", 1.0
TRAIN = "train"  # the `set` of a training question in questions.jsonl; test questions carry their version
QWEN_LOGITS = "Qwen/Qwen3.6-27B letter-logit argmax at the answer position (ceiling)"


def consistent(item: str) -> tuple[str, int, int] | None:
    """(modal letter, k, n) under the eval's rule, or None: as `ev.prep` selects cases."""
    c = ev.census().get(item) or Counter()
    letters = Counter({a: k for a, k in c.items() if a})
    if not letters:
        return None
    modal, k = letters.most_common(1)[0]
    n = sum(c.values())
    return (modal, k, n) if k / n >= ev.MIN_SHARE else None


def prep(a):
    from transformers import AutoTokenizer

    inputs = Path(a.inputs)
    questions = []
    for item in ev.split()["probe"]:
        if c := consistent(item):
            questions.append({"set": TRAIN, "item": item, "modal": c[0], "k": c[1], "n": c[2], "gold": ev._mod()._table().loc[item].answer})
    n_train = len(questions)
    for _, row, _ in ev.cases(inputs, list(ev.VERSIONS)):
        questions.append({"set": row["version"], "modal": row["modal_cf"]} | row)
    run = new_run(
        "mcq_answer_probe",
        a.slug,
        description="MCQ answer probe trained on the mcq_unselected probe half, predicting Qwen's letter on every eval case",
        models=[{"name": MODEL, "served_by": "HF transformers bf16 (scripts/hs_at_positions.py)"}],
        params=vars(a) | {"hidden": HIDDEN, "site": SITE, "l2": L2, "n_train": n_train, "n_test": len(questions) - n_train},
    )
    tok = AutoTokenizer.from_pretrained(MODEL)
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    rows, odd = [], Counter()
    for q in questions:
        pr = ev.Prompt(str(q["item"]))
        text = tok.apply_chat_template(pr.messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        start = text.index(pr.text)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        offs = enc["offset_mapping"]
        nl = [next(i for i, (s, e) in enumerate(offs) if s <= start + c < e) for c, _ in ev.option_newlines(pr.text)]
        toks = [tok.decode([enc["input_ids"][p]]) for p in nl]
        assert all("\n" in t for t in toks), (q["item"], toks)
        odd.update(t for t in toks if t.strip(" ") not in ("\n", "\n\n"))
        assert q.setdefault("n_opts", len(nl)) == len(nl), (q["item"], q["n_opts"], len(nl))
        rows.append({"id": f"{q['set']}/{q['item']}", "ids": enc["input_ids"], "sites": {SITE: nl}, "cand": letter_ids[: len(nl)]})
    (run / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions))
    (run / "hs_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    right = sum(q["modal"] == q["gold"] for q in questions[:n_train])
    print(f"{run}: train {n_train} ({right} right), test {len(questions) - n_train}; odd newline tokens {dict(odd)}")
    print(f"GPU: python scripts/hs_at_positions.py {run}/hs_inputs.jsonl {CACHE / run.name} --layers {HIDDEN}")


def score(a):
    run = Path(a.run)
    cache = CACHE / run.name
    qs = [json.loads(line) for line in (run / "questions.jsonl").read_text().splitlines()]
    layers = json.loads((cache / "layers.json").read_text())
    X = np.asarray(np.load(cache / f"{SITE}.npy", mmap_mode="r")[:, layers.index(HIDDEN)], dtype=np.float64)
    logp = {r["id"]: r["logp"] for r in map(json.loads, (cache / "logp.jsonl").read_text().splitlines())}
    n_opts = np.array([q["n_opts"] for q in qs])
    assert len(X) == n_opts.sum(), (len(X), n_opts.sum())
    starts = np.r_[0, np.cumsum(n_opts)[:-1]]
    qi = np.repeat(np.arange(len(qs)), n_opts)
    train = np.array([q["set"] == TRAIN for q in qs])
    tr = train[qi]
    y = np.concatenate([[L == q["modal"] for L in LETTERS[: q["n_opts"]]] for q in qs]).astype(float)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = (X - mu) / sd
    w = _fit_listwise(Z[tr], y[tr], np.flatnonzero(np.r_[True, qi[tr][1:] != qi[tr][:-1]]), L2)
    np.savez(run / "probe.npz", mean=mu, scale=sd, w=w, hidden=HIDDEN, site=SITE, l2=L2)
    p = _softmax_by_q(Z @ w, starts)
    n_train = int(train.sum())
    predictor = f"MCQ answer probe (listwise, {SITE} hidden_states[{HIDDEN}], diag, l2 {L2:g}; {n_train} consistent probe-half questions)"
    rows, hits = [], Counter()
    for i, q in enumerate(qs):
        sl = slice(starts[i], starts[i] + q["n_opts"])
        probe = LETTERS[int(np.argmax(p[sl]))]
        qwen = LETTERS[int(np.argmax(logp[f"{q['set']}/{q['item']}"]))]
        for cond, ans in (("probe", probe), ("qwen_logits", qwen)):
            hits[q["set"], cond, "all"] += ans == q["modal"]
            hits[q["set"], cond, "right" if q["modal"] == q["gold"] else "wrong"] += ans == q["modal"]
        hits[q["set"], "n", "all"] += 1
        hits[q["set"], "n", "right" if q["modal"] == q["gold"] else "wrong"] += 1
        if train[i]:
            continue
        case = {k: v for k, v in q.items() if k not in ("set", "modal")} | {"eval": "mcq_unselected", "draw": 0, "valid": True, "cost": 0.0}
        extra = {"probe": {"predictor": predictor, "probe_p": [round(float(x), 4) for x in p[sl]]}}
        extra["qwen_logits"] = {"predictor": QWEN_LOGITS, "logp": [round(x, 4) for x in logp[f"{q['set']}/{q['item']}"]]}
        for cond, ans in (("probe", probe), ("qwen_logits", qwen)):
            rows.append(case | {"condition": cond, "answer": ans, "canon": ans, "correct": ans == q["modal"]} | extra[cond])
    (run / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    versions = [v for v in dict.fromkeys(q["set"] for q in qs) if v != TRAIN]
    lines = [
        (
            f"MCQ answer probe ({predictor}) and Qwen's own letter-logit argmax on the mcq_unselected cases. Accuracy = share of "
            "questions whose predicted letter is Qwen's modal letter; right / wrong = Qwen's modal letter is / isn't the gold."
        ),
        "",
        "| set | split | n | probe | qwen_logits |",
        "|---|---|---|---|---|",
    ]
    for s in [*versions, "pooled", f"{TRAIN} (in-sample)"]:
        sets = versions if s == "pooled" else [s.split(" ")[0]]
        for split in ("all", "right", "wrong"):
            n = sum(hits[v, "n", split] for v in sets)
            cells = [sum(hits[v, cond, split] for v in sets) / n for cond in ("probe", "qwen_logits")]
            lines.append(f"| {s} | {split} | {n} | " + " | ".join(f"{c:.3f}" for c in cells) + " |")
    (run / "scores.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    meta = json.loads((run / "meta.json").read_text())
    meta["params"] |= {"pod": a.pod, "cost_usd": a.cost, "predictors": {"probe": predictor, "qwen_logits": QWEN_LOGITS}}
    (run / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if a.finish:
        finish(run)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("inputs", help="the eval's inputs run (split.json, cases.jsonl)")
    p.add_argument("--slug", default="mcq-unselected-probe")
    s = sub.add_parser("score")
    s.add_argument("run")
    s.add_argument("--pod", help="which pod ran hs_at_positions.py, for meta.json")
    s.add_argument("--cost", type=float, help="its RunPod cost in dollars")
    s.add_argument("--finish", action="store_true", help="mark the run done and sync it to HF")
    a = ap.parse_args()
    {"prep": prep, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
