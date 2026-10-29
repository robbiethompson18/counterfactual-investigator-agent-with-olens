"""The MCQ answer probe as a baseline on cfbench's hint_qwen_dsv4 evals (docs/mcq-answer-probe.md).

Both evals show runs of a prompt that hints the correct letter and ask about the counterfactual with no hint: did Qwen
know it (hint_qwen_dsv4, claim task) and which letter does it pick (hint_qwen_dsv4_letter). In both envs the hint comes
after the options, so with causal attention the residual at each option's newline in the hinted base-arm prompt equals
the no-hint prompt's: the probe reads Qwen's own answer from the base arm alone (zero counterfactuals).

    # CPU: the probe, refit on MMLU-Pro questions disjoint from every item in twomodel_qwen_dsv4_follow.csv, and the
    # inputs for scripts/hs_at_positions.py (each case's hinted base-arm prompt, sites = the option newlines)
    uv run --with scikit-learn --with transformers python -m environments.hint_mcq.probe_eval prep INPUTS_RUN
    # GPU: python scripts/hs_at_positions.py RUN/hs_inputs.jsonl data/probe_cache/<run name>
    # CPU: equality check against the cached no-hint activations, predictions in each eval's format, its own score.py
    uv run --with scikit-learn python -m environments.hint_mcq.probe_eval score RUN

INPUTS_RUN is the eval's inputs run (data/runs/cfbench_evals/<date>_hint-qwen-dsv4-inputs: cases.jsonl). Predictions go
to RUN/knew/predictions.jsonl and RUN/letter/predictions.jsonl, conditions `probe` (p = the probe's p of the hinted,
correct letter; letter = its argmax), `probe_argmax` (knew: p = 1 if the argmax is the hinted letter, else 0) and
`hinted_letter` (always the hinted letter: p = 1 / that letter).
"""

import argparse
import contextlib
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from counterfactual_investigator_agent_with_olens.runs import new_run
from environments._lib.core import load_env
from environments.hint_mcq.probe import CACHE, MODEL, _fit_listwise, _softmax_by_q

TRAIN = Path("data/runs/mcq_answer_probe/2026-10-28_scale2k")  # the probe's MMLU-Pro run (questions, options, nl.npy cache)
TQA = Path("data/runs/mcq_answer_probe/2026-10-28_tqa-oos")  # cached TruthfulQA activations (its edited prompt = the hinted prefix)
FOLLOW = Path("environments/hint_mcq/twomodel_qwen_dsv4_follow.csv")
VERSIONS = {
    "mmlu_pro": ("hint_mcq", "hint+d1"),
    "truthfulqa": ("hinted_truthfulqa", "hint/stanford"),
    "mmlu_pro_12k": ("hint_mcq", "hint+d1"),
}
# hint before the question (cfbench hint_qwen_dsv4 before_*): the option newlines come after the hint, so the probe reads
# hinted activations there; the equality check then measures how far the hint moved them
VERSIONS |= {
    "before_mmlu_pro": ("hint_mcq", "before/hint+d1"),
    "before_truthfulqa": ("hinted_truthfulqa", "hint_before/stanford"),
    "before_mmlu_pro_12k": ("hint_mcq", "before/hint+d1"),
}
HIDDEN = 52  # hidden_states index (the output of block 51): the frozen probe's
KNEW, DIDNT = "Qwen/Qwen3.6-27B (knew)", "Qwen/Qwen3.6-27B (didn't)"
LETTERS = "ABCDEFGHIJKLMNOP"


def _base_item(version: str, item: str, gold: str) -> str:
    return f"{item}.h{gold}" if version.endswith("truthfulqa") else item


def _sites(text: str, version: str) -> list[int]:
    """Char index of the newline after each option's text in the rendered hinted prompt (the last option ends at the
    "\\n\\n" before the hint)."""
    pat = (lambda L: f"\n{L}. ") if "mmlu_pro" in version else (lambda L: f"\n({L}) ")
    starts = [text.index(pat(L)) for L in LETTERS if pat(L) in text]
    return [*starts[1:], text.index("\n\n", starts[-1])]


def prep(a):
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoTokenizer

    # the complementary training set: consistent MMLU-Pro questions with cached activations, none in the eval's csv
    excl = {r["item"] for r in csv.DictReader(FOLLOW.open()) if r["source"] == "mmlu_pro"}
    qs = [json.loads(line) for line in (TRAIN / "questions.jsonl").read_text().splitlines()]
    keep = [r for r in qs if str(r["item"]) not in excl]
    right, wrong = [r for r in keep if r["right"]], [r for r in keep if not r["right"]]
    n = min(len(right), len(wrong))
    rng = random.Random(0)
    train = {str(r["item"]) for r in rng.sample(right, n) + rng.sample(wrong, n)}
    print(
        f"{len(qs)} cached questions, {len(qs) - len(keep)} in the eval csv removed, {len(right)} right / {len(wrong)} wrong left; "
        f"train {n} + {n}"
    )
    opts = [json.loads(line) for line in (TRAIN / "options.jsonl").read_text().splitlines()]
    layers = json.loads((TRAIN / "meta.json").read_text())["params"]["layers"]
    rows = np.array([str(o["item"]) in train for o in opts])
    X = np.asarray(np.load(CACHE / TRAIN.name / "nl.npy", mmap_mode="r")[:, layers.index(HIDDEN)], dtype=np.float64)[rows]
    y = np.array([o["is_answer"] for o in opts], float)[rows]
    q = np.array([o["q"] for o in opts])[rows]
    sc = StandardScaler().fit(X)
    w = _fit_listwise(sc.transform(X), y, np.flatnonzero(np.r_[True, q[1:] != q[:-1]]), a.l2)

    cases = [json.loads(line) for line in (Path(a.inputs) / "cases.jsonl").read_text().splitlines()]
    run = new_run(
        "mcq_answer_probe",
        a.slug,
        description="MCQ answer probe (refit off the eval csv items) as a baseline on the hint_qwen_dsv4 evals",
        models=[{"name": MODEL, "served_by": "HF transformers bf16 (scripts/hs_at_positions.py)"}],
        params=vars(a)
        | {"hidden": HIDDEN, "n_train_questions": 2 * n, "train_items": sorted(train, key=int), "n_excluded": len(qs) - len(keep)},
    )
    np.savez(run / "probe.npz", mean=sc.mean_, scale=sc.scale_, w=w)
    tok = AutoTokenizer.from_pretrained(MODEL)
    inputs, meta = [], []
    for c in cases:
        env, arm = VERSIONS[c["version"]]
        msgs = load_env(env).build(arm, _base_item(c["version"], c["item"], c["gold"])).messages
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ends = _sites(text, c["version"])
        n_opts = len(ends)
        offs = enc["offset_mapping"]
        nl = [next(i for i, (s, e) in enumerate(offs) if s <= ch < e) for ch in ends]
        toks = [tok.decode([enc["input_ids"][p]]) for p in nl]
        # a few MMLU-Pro options end in a stray "\n" (8326: "B. ...\n\n") or a space (9461: " \n\n"): the site token varies
        assert all(t.strip(" ") in ("\n", "\n\n") for t in toks) and toks[-1].strip(" ") == "\n\n", (c["item"], toks)
        inputs.append({"id": f"{c['version']}/{c['item']}", "ids": enc["input_ids"], "sites": {"nl": nl}, "cand": [enc["input_ids"][-1]]})
        meta.append({k: c[k] for k in ("version", "item", "class", "gold", "answers_cf")} | {"n_opts": n_opts})
    (run / "hs_inputs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in inputs))
    (run / "cases.jsonl").write_text("".join(json.dumps(x) + "\n" for x in meta))
    print(f"run {run}: {len(meta)} cases, inputs for hs_at_positions.py; then put its output in {CACHE / run.name}")


def _cached_no_hint(version: str, item: str) -> np.ndarray | None:
    """The cached no-hint nl activations of an item at HIDDEN, if it was in the probe's MMLU-Pro run or the TruthfulQA run."""
    run = TRAIN if version.removeprefix("before_") == "mmlu_pro" else TQA
    opts = [json.loads(line) for line in (run / "options.jsonl").read_text().splitlines()]
    idx = [i for i, o in enumerate(opts) if str(o["item"]) == item]
    if not idx:
        return None
    return np.asarray(np.load(CACHE / run.name / "nl.npy", mmap_mode="r")[idx, HIDDEN // 2], dtype=np.float64)  # every 2nd layer saved


def score(a):
    from sklearn.metrics import roc_auc_score

    run = Path(a.run)
    cases = [json.loads(line) for line in (run / "cases.jsonl").read_text().splitlines()]
    if a.cached_only:  # the cached no-hint activations stand in for the base arm's (same prefix), cases without them dropped
        cached = [(c, _cached_no_hint(c["version"], c["item"])) for c in cases]
        cases, X = [c for c, x in cached if x is not None], np.concatenate([x for _, x in cached if x is not None])
        out = run / "cached_only"
        print(f"cached-only: {len(cases)} cases")
    else:
        cache = CACHE / run.name
        layers = json.loads((cache / "layers.json").read_text())
        X = np.asarray(np.load(cache / "nl.npy", mmap_mode="r")[:, layers.index(HIDDEN)], dtype=np.float64)
        out = run
    pr = np.load(run / "probe.npz")
    s = (X - pr["mean"]) / pr["scale"] @ pr["w"]
    q = np.repeat(np.arange(len(cases)), [c["n_opts"] for c in cases])
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    p = _softmax_by_q(s, starts)
    # equality check: base-arm (hinted) activations vs the cached no-hint ones at the same option newlines
    cos, off = [], 0
    for c in cases:
        ref = None if a.cached_only else _cached_no_hint(c["version"], c["item"])
        x = X[off : off + c["n_opts"]]
        if ref is not None:
            cos += list((x * ref).sum(1) / np.linalg.norm(x, axis=1) / np.linalg.norm(ref, axis=1))
        off += c["n_opts"]
    check = {"n_options": len(cos), "min_cos": float(np.min(cos)), "median_cos": float(np.median(cos))} if cos else {}
    print(f"hinted vs cached no-hint activations at the option newlines: {check}")
    knew, letter = [], []
    for i, c in enumerate(cases):
        pi = p[starts[i] : starts[i] + c["n_opts"]]
        g = LETTERS.index(c["gold"])
        top = LETTERS[int(np.argmax(pi))]
        modal = max(c["answers_cf"], key=c["answers_cf"].get)
        k = int(c["class"] == "knew")
        for version in (c["version"], "pooled"):  # per source, and pooled as the evals score them since 5bf70f5
            base = {"version": version, "source": c["version"], "item": c["item"], "class": c["class"], "valid": True, "gold": c["gold"]}
            for cond, pk in (("probe", float(pi[g])), ("probe_argmax", float(top == c["gold"])), ("hinted_letter", 1.0)):
                knew.append(base | {"condition": cond, "p": pk, "model": KNEW if k else DIDNT, "k_cf": k, "n_cf": 1})
            for cond, ans in (("probe", top), ("hinted_letter", c["gold"])):
                letter.append(
                    base
                    | {
                        "condition": cond,
                        "answer": ans,
                        "modal_cf": modal,
                        "correct": ans == modal,
                        "probe_pmf": [round(float(x), 4) for x in pi],
                    }
                )
    for name, rows in (("knew", knew), ("letter", letter)):
        (out / name).mkdir(parents=True, exist_ok=True)
        (out / name / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = [
        f"Equality check (hinted base-arm vs cached no-hint activations at the option newlines, hidden_states[{HIDDEN}]): {check}",
        "",
    ]
    for name, mod in (("knew", "cfbench.evals.hint_qwen_dsv4.score"), ("letter", "cfbench.evals.hint_qwen_dsv4_letter.score")):
        cmd = [sys.executable, "-m", mod, str((out / name).resolve())]
        txt = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=a.evals_repo).stdout
        report += [f"## {name} ({mod})", "", "```", txt.rstrip(), "```", ""]
    # also: AUROC of the probe's p(gold), all versions pooled, for reference (the eval reports it per version)
    pk = [r["p"] for r in knew if r["condition"] == "probe" and r["version"] == "pooled"]
    yk = [r["k_cf"] for r in knew if r["condition"] == "probe" and r["version"] == "pooled"]
    report.append(f"Pooled AUROC of probe p(hinted letter), knew vs didn't, both versions: {roc_auc_score(yk, pk):.3f}")
    (out / "scores.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


PROBE = "probe"  # the condition name; its predictor
PREDICTOR = "MCQ answer probe (argmax p(option | activations), nl hidden_states[52])"
DROP = ("messages", "tool_calls", "tools", "turns", "rationale")


def cfbench_runs(a):
    """The probe's predictions as cfbench runs (condition `probe`), one per eval, on the same cases and inputs as an
    investigator run of that eval (its rows give the case fields), so the env viewer's eval pages show the probe beside the
    investigator conditions: the claim/answer tables, the charts and the paired comparisons."""
    from counterfactual_investigator_agent_with_olens.runs import finish

    base = Path(a.run)
    probe = {
        name: [json.loads(line) for line in (base / name / "predictions.jsonl").read_text().splitlines()] for name in ("knew", "letter")
    }
    for name, ref in (("knew", Path(a.knew)), ("letter", Path(a.letter))):
        pred = {(r["source"], r["item"]): r for r in probe[name] if r["condition"] == "probe" and r["version"] == "pooled"}
        meta = json.loads((ref / "meta.json").read_text())
        rows, seen = [], set()
        for line in (ref / "predictions.jsonl").read_text().splitlines():
            r = json.loads(line)
            if (key := (r["source"], r["item"])) in seen:
                continue
            seen.add(key)
            p = pred[key]
            out = {k: v for k, v in r.items() if k not in DROP} | {
                "condition": PROBE,
                "predictor": PREDICTOR,
                "draw": 0,
                "cost": 0.0,
                "valid": True,
            }
            if name == "knew":
                out["p"] = p["p"]
            else:
                out |= {"answer": p["answer"], "canon": p["answer"], "correct": p["answer"] == r["modal_cf"]}
            rows.append(out)
        params = meta["params"] | {"conditions": PROBE, "predictors": PREDICTOR, "probe_run": str(base), "template_run": ref.name}
        desc = f"{meta['params']['eval']}: the MCQ answer probe as a condition ({len(rows)} cases)"
        model = {"name": PREDICTOR, "served_by": f"deterministic, {base.name}"}
        run = new_run("cfbench", f"{meta['slug']}-probe", description=desc, models=[model], params=params)
        (run / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(run, len(rows), "rows")
        finish(run)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("inputs", help="the eval's inputs run (cases.jsonl)")
    p.add_argument("--slug", default="eval-baseline")
    p.add_argument("--l2", type=float, default=1.0)
    s = sub.add_parser("score")
    s.add_argument("run")
    s.add_argument("--evals-repo", default=".", help="checkout whose cfbench/evals/hint_qwen_dsv4*/score.py score the rows")
    s.add_argument("--cached-only", action="store_true", help="score the cases with cached no-hint activations, no GPU output")
    c = sub.add_parser("cfbench-runs")
    c.add_argument("run", help="this script's run (knew/ and letter/ predictions from `score`)")
    c.add_argument("knew", help="an investigator run of hint_qwen_dsv4 (data/runs/cfbench/...): its case fields and meta")
    c.add_argument("letter", help="an investigator run of hint_qwen_dsv4_letter")
    a = ap.parse_args()
    {"prep": prep, "score": score, "cfbench-runs": cfbench_runs}[a.cmd](a)


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        main()
