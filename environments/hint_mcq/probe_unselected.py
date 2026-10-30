"""The MCQ answer probe as a baseline on a "what letter will Qwen say?" eval: mcq_unselected (docs/mcq-unselected.md) or
supergpqa_inline (docs/supergpqa-inline.md); docs/mcq-answer-probe.md for the probe itself.

How well does a linear probe on Qwen3.6-27B's residual stream predict the letter Qwen gives, on the cases the investigator
is scored on? Training set: the eval's probe pool (`split()["probe"]`: mcq_unselected's 2,000-question probe half, or
supergpqa_inline's 3,328 consistent questions outside the eval set) under the eval's own consistency rule (census modal
letter >= MIN_SHARE of all samples), not balanced, no eval question. Test set: every eval case of every version
(`cases.jsonl` of the inputs run). The probe is Kylie's chosen one (probe.py): one-hot listwise, diagonal standardisation,
the `nl` site (the newline token after each option), hidden_states[52], l2 1. Conditions in predictions.jsonl:

- `probe`: Kylie's frozen probe, fit once on the 718 balanced MMLU-Pro questions of FROZEN (its cached activations
  in data/probe_cache/), applied unchanged, as probe.py's `oos` does.
- `probe_tuned`: the same recipe refit on the eval's own probe pool (Robbie's, PR #421).
- `probe_raw`: the 718-question fit without per-dimension scaling (raw h, l2 RAW_L2 from probe.py's
  `standardise` ablation; a global mean is a no-op under the per-question softmax).
- `qwen_logits`: Qwen's own letter-logit argmax at the answer position (ceiling).

    # CPU: training questions + hs_at_positions inputs for both sets (no-hint prompt, sites = option newlines, cand = letters)
    uv run --with transformers --with jinja2 python -m environments.hint_mcq.probe_unselected prep mcq_unselected INPUTS_RUN
    # GPU: python scripts/hs_at_positions.py RUN/hs_inputs.jsonl OUT --layers 52   -> data/probe_cache/<run name>/
    # CPU: fit on the train rows, predict every case; predictions.jsonl, scores.md, probe.npz
    uv run --with scipy python -m environments.hint_mcq.probe_unselected score RUN --pod "..." --cost 8 --finish

`prep mcq_unselected` also writes FROZEN's 718 questions (set `mmlu718`, rebuilt with today's prompt code) so `score` can
check the new activations against the cached ones. predictions.jsonl rows are cfbench answer-task rows (the case fields of
`cases()`, draw 0, cost 0), so `cfbench.answer RUN` scores them. The env viewer only indexes runs under data/runs/cfbench/,
so `mirror RUN --finish` copies them into one there; then `python -m environments evals NAME --upload`.
"""

import argparse
import importlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from counterfactual_investigator_agent_with_olens.runs import finish, new_run
from environments._lib.runfolder import REPO_ROOT
from environments.hint_mcq.probe import CACHE, MODEL, _fit_listwise, _softmax_by_q

HIDDEN, SITE, L2 = 52, "nl", 1.0
RAW_L2 = 1.0  # chosen by `probe.py whiten --variants demean` on the 718, every outer fold (docs/mcq-answer-probe.md)
LETTERS = "ABCDEFGHIJKLMNOP"
TRAIN, CHECK = "train", "mmlu718"  # `set` of a training question / of a FROZEN question; test questions carry their version
FROZEN = REPO_ROOT / "data/runs/mcq_answer_probe/2026-10-28_scale2k"  # Kylie's 718 (questions/options.jsonl; acts in CACHE)
QWEN_LOGITS = "Qwen/Qwen3.6-27B letter-logit argmax at the answer position (ceiling)"
CONDS = ("probe", "probe_tuned", "probe_raw", "qwen_logits")


def _ev(name: str):
    return importlib.import_module(f"cfbench.evals.{name}.eval")


def consistent(ev, item: str) -> tuple[str, int, int] | None:
    """(modal letter, k, n) under the eval's rule, or None: as the eval's `prep` selects cases."""
    c = ev.census().get(item) or Counter()
    letters = Counter({a: k for a, k in c.items() if a})
    if not letters:
        return None
    modal, k = letters.most_common(1)[0]
    n = sum(c.values())
    return (modal, k, n) if k / n >= ev.MIN_SHARE else None


def prep(a):
    from transformers import AutoTokenizer

    ev, inputs = _ev(a.eval), Path(a.inputs)
    table = ev._mod()._table()
    gold = "answer_letter" if "answer_letter" in table.columns else "answer"
    questions = []
    for item in ev.split()["probe"]:
        if c := consistent(ev, item):
            questions.append({"set": TRAIN, "item": item, "modal": c[0], "k": c[1], "n": c[2], "gold": table.loc[item][gold]})
    n_train = len(questions)
    for _, row, _ in ev.cases(inputs, list(ev.VERSIONS)):
        questions.append({"set": row["version"], "modal": row["modal_cf"]} | row)
    n_test = len(questions) - n_train
    if a.eval == "mcq_unselected":
        for q in map(json.loads, (FROZEN / "questions.jsonl").read_text().splitlines()):
            questions.append({"set": CHECK, "item": q["item"], "modal": q["own"], "gold": q["gold"], "n_opts": q["n_opts"]})
    run = new_run(
        "mcq_answer_probe",
        a.slug or f"{a.eval.replace('_', '-')}-probe",
        description=f"MCQ answer probes (own probe pool, Kylie's frozen 718) predicting Qwen's letter on every {a.eval} case",
        models=[{"name": MODEL, "served_by": "HF transformers bf16 (scripts/hs_at_positions.py)"}],
        params=vars(a) | {"hidden": HIDDEN, "site": SITE, "l2": L2, "n_train": n_train, "n_test": n_test},
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
    n_check = len(questions) - n_train - n_test
    print(f"{run}: train {n_train} ({right} right), test {n_test}, check {n_check}; odd newline tokens {dict(odd)}")
    print(f"GPU: python scripts/hs_at_positions.py {run}/hs_inputs.jsonl {CACHE / run.name} --layers {HIDDEN}")


def frozen() -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray]:
    """Kylie's 718 fit as probe.py's `oos` does ({name: (mean, scale, w)}), and the cached training activations."""
    layers = json.loads((FROZEN / "meta.json").read_text())["params"]["layers"]
    opts = [json.loads(line) for line in (FROZEN / "options.jsonl").read_text().splitlines()]
    q = np.array([o["q"] for o in opts])
    y = np.array([o["is_answer"] for o in opts], float)
    X = np.asarray(np.load(CACHE / FROZEN.name / f"{SITE}.npy", mmap_mode="r")[:, layers.index(HIDDEN)], dtype=np.float64)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    mu, sd = X.mean(0), X.std(0)
    one = np.ones_like(sd)
    return {
        "probe": (mu, sd, _fit_listwise((X - mu) / sd, y, starts, L2)),
        "probe_raw": (mu, one, _fit_listwise(X - mu, y, starts, RAW_L2)),
    }, X


def score(a):
    run = Path(a.run)
    cache = CACHE / run.name
    meta = json.loads((run / "meta.json").read_text())
    name = meta["params"]["eval"]
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
    w = _fit_listwise((X[tr] - mu) / sd, y[tr], np.flatnonzero(np.r_[True, qi[tr][1:] != qi[tr][:-1]]), L2)
    np.savez(run / "probe.npz", mean=mu, scale=sd, w=w, hidden=HIDDEN, site=SITE, l2=L2)
    probes, X718 = frozen()
    probes = {"probe_tuned": (mu, sd, w)} | probes
    p = {c: _softmax_by_q((X - m) / s @ v, starts) for c, (m, s, v) in probes.items()}
    check = np.array([q["set"] == CHECK for q in qs])[qi]
    if check.any():  # today's prompts and forward pass vs the cached activations the frozen probe was fit on
        cos = np.sum(X[check] * X718, 1) / np.linalg.norm(X[check], axis=1) / np.linalg.norm(X718, axis=1)
        print(f"{CHECK}: cosine new vs cached nl h[{HIDDEN}] min {cos.min():.4f} median {np.median(cos):.4f} over {len(cos)} options")
    n_train = int(train.sum())
    desc = f"listwise, {SITE} hidden_states[{HIDDEN}]"
    predictors = {
        "probe_tuned": f"MCQ answer probe ({desc}, diag, l2 {L2:g}; {n_train} consistent {name} probe-pool questions)",
        "probe": f"MCQ answer probe ({desc}, diag, l2 {L2:g}; Kylie's 718 balanced MMLU-Pro questions, frozen)",
        "probe_raw": f"MCQ answer probe ({desc}, no scaling, l2 {RAW_L2:g}; Kylie's 718 balanced MMLU-Pro questions, frozen)",
        "qwen_logits": QWEN_LOGITS,
    }
    rows, hits = [], Counter()
    for i, q in enumerate(qs):
        sl = slice(starts[i], starts[i] + q["n_opts"])
        lp = logp[f"{q['set']}/{q['item']}"]
        ans = {c: LETTERS[int(np.argmax(p[c][sl]))] for c in probes} | {"qwen_logits": LETTERS[int(np.argmax(lp))]}
        split = "right" if q["modal"] == q["gold"] else "wrong"
        for c in CONDS:
            hits[q["set"], c, "all"] += ans[c] == q["modal"]
            hits[q["set"], c, split] += ans[c] == q["modal"]
        hits[q["set"], "n", "all"] += 1
        hits[q["set"], "n", split] += 1
        if q["set"] in (TRAIN, CHECK):
            continue
        case = {k: v for k, v in q.items() if k not in ("set", "modal")} | {"eval": name, "draw": 0, "valid": True, "cost": 0.0}
        for c in CONDS:
            extra = {"logp": [round(x, 4) for x in lp]} if c == "qwen_logits" else {"probe_p": [round(float(x), 4) for x in p[c][sl]]}
            pred = {"condition": c, "answer": ans[c], "canon": ans[c], "correct": ans[c] == q["modal"], "predictor": predictors[c]}
            rows.append(case | pred | extra)
    (run / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    versions = [v for v in dict.fromkeys(q["set"] for q in qs) if v not in (TRAIN, CHECK)]
    sets = [*versions, "pooled", f"{TRAIN} (in-sample)"] + ([f"{CHECK} (frozen probes in-sample)"] if check.any() else [])
    lines = [
        (
            f"Probes and Qwen's own letter-logit argmax on the {name} cases. Accuracy = share of questions whose predicted letter is "
            "Qwen's modal letter; right / wrong = Qwen's modal letter is / isn't the gold."
        ),
        "",
        *[f"- `{c}`: {d}" for c, d in predictors.items()],
        "",
        "| set | split | n | " + " | ".join(CONDS) + " |",
        "|---|---|---|" + "---|" * len(CONDS),
    ]
    for s in sets:
        vs = versions if s == "pooled" else [s.split(" ")[0]]
        for split in ("all", "right", "wrong"):
            n = sum(hits[v, "n", split] for v in vs)
            cells = [sum(hits[v, c, split] for v in vs) / n for c in CONDS]
            lines.append(f"| {s} | {split} | {n} | " + " | ".join(f"{c:.3f}" for c in cells) + " |")
    (run / "scores.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    meta["params"] |= {"pod": a.pod, "cost_usd": a.cost, "predictors": predictors, "raw_l2": RAW_L2, "frozen": FROZEN.name}
    (run / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if a.finish:
        finish(run)


def mirror(a):
    """Copy a scored run's probe predictions into a cfbench run (meta params `eval`, `task` answer), so the env viewer's eval
    page shows the probe conditions beside the investigator's (environments/_lib/evals.py indexes data/runs/cfbench/ only).
    Only `probe` and `probe_tuned` (Kylie, 2026-10-30: no logits, no raw variant on the page); scores.md keeps all four."""
    src = Path(a.run)
    meta = json.loads((src / "meta.json").read_text())
    p = meta["params"]
    ev = _ev(p["eval"])
    conds = ["probe", "probe_tuned"]
    lines = (src / "predictions.jsonl").read_text().splitlines()
    rows = "".join(line + "\n" for line in lines if json.loads(line)["condition"] in conds)
    run = new_run(
        "cfbench",
        a.slug or src.name.split("_", 1)[1],
        description=f"{p['eval']}: linear probes (no investigator), mirrored from mcq_answer_probe/{src.name}",
        models=meta["models"],
        params={"env": f"eval:{p['eval']}", "eval": p["eval"], "envs": ev.ENVS, "inputs": str(Path(p["inputs"]).relative_to("data/runs")),
                "versions": list(ev.VERSIONS), "conditions": ",".join(conds), "predictors": "linear probes (no investigator)",
                "models": [ev.QWEN], "task": "answer", "draws": 1, "source": f"mcq_answer_probe/{src.name}"},
    )  # fmt: skip
    (run / "predictions.jsonl").write_text(rows)
    if a.finish:
        finish(run)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("eval", choices=["mcq_unselected", "supergpqa_inline"])
    p.add_argument("inputs", help="the eval's inputs run (split.json, cases.jsonl)")
    p.add_argument("--slug", help="default <eval>-probe")
    s = sub.add_parser("score")
    s.add_argument("run")
    s.add_argument("--pod", help="which pod ran hs_at_positions.py, for meta.json")
    s.add_argument("--cost", type=float, help="its GPU cost in dollars")
    s.add_argument("--finish", action="store_true", help="mark the run done and sync it to HF")
    m = sub.add_parser("mirror")
    m.add_argument("run", help="a scored probe run")
    m.add_argument("--slug", help="default: the probe run's slug")
    m.add_argument("--finish", action="store_true", help="mark the run done and sync it to HF")
    a = ap.parse_args()
    {"prep": prep, "score": score, "mirror": mirror}[a.cmd](a)


if __name__ == "__main__":
    main()
