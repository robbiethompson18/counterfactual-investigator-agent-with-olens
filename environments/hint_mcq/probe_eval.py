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
    # CPU, hint-first (before_*) runs: tune the probe on the hinted activations themselves to read the no-hint answer
    uv run --with scikit-learn python -m environments.hint_mcq.probe_eval tune RUN

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
from concurrent.futures import ThreadPoolExecutor
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


L2_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def tune(a):
    """Hint-first cases: the hint is before the options, so the activations at the option newlines are hinted and the frozen
    probe is pulled toward the hinted letter. Fit listwise probes on those activations themselves, target = Qwen's no-hint
    modal answer, held out by question (GroupKFold, l2 by inner GroupKFold on the training folds only): from scratch,
    fine-tuned from the frozen probe (|w - w0|^2 penalty, the frozen standardiser kept so w0 means the same thing), pooled
    with the frozen probe's no-hint MMLU-Pro set (minus the eval's items), each also with an unpenalised "is the hinted
    option" feature. Baselines: the frozen probe and the hinted letter. Metrics on knew / didn't / balanced with a
    bootstrap over questions; writes RUN-independent tune.md + per-case held-out pmfs to a new run."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    src = Path(a.run)
    cases = [json.loads(line) for line in (src / "cases.jsonl").read_text().splitlines()]
    layers = json.loads((CACHE / src.name / "layers.json").read_text())
    X = np.asarray(np.load(CACHE / src.name / "nl.npy", mmap_mode="r")[:, layers.index(HIDDEN)], dtype=np.float64)
    n = np.array([c["n_opts"] for c in cases])
    starts = np.r_[0, np.cumsum(n)[:-1]]
    modal = [max(c["answers_cf"], key=c["answers_cf"].get) for c in cases]
    t = np.concatenate([np.arange(k) == LETTERS.index(m) for k, m in zip(n, modal)]).astype(float)
    # the hinted letter is data (a case's `hinted`, else its gold: the current evals hint the gold), and knew = Qwen's
    # no-hint answer is the hinted letter, so a design that also hints wrong letters re-runs unchanged
    hint = [LETTERS.index(c.get("hinted", c["gold"])) for c in cases]
    hinted = np.concatenate([np.arange(k) == h for k, h in zip(n, hint)]).astype(float)
    knew = np.array([LETTERS.index(m) == h for m, h in zip(modal, hint)])
    source = np.array([c["version"].removeprefix("before_") for c in cases])
    # one group per underlying question: TruthfulQA items carry a ".h<letter>" hint suffix in some runs; mmlu_pro and
    # mmlu_pro_12k share MMLU-Pro ids
    groups = np.array([f"{'tqa' if 'truthfulqa' in c['version'] else 'mmlu'}/{c['item'].split('.h')[0]}" for c in cases])
    assert len(set(groups)) == len(cases), "one case per question expected"
    pr = np.load(src / "probe.npz")
    frozen_items = set(json.loads((src / "meta.json").read_text())["params"]["train_items"])
    eval_items = {g.split("/")[1] for g in groups if g.startswith("mmlu/")}
    assert not frozen_items & eval_items, "frozen probe trained on an eval item"

    # the no-hint pool: the frozen probe's MMLU-Pro run, every cached question not in the eval (balanced as cached)
    opts = [json.loads(line) for line in (TRAIN / "options.jsonl").read_text().splitlines()]
    keep = np.array([str(o["item"]) not in eval_items for o in opts])
    tlayers = json.loads((TRAIN / "meta.json").read_text())["params"]["layers"]
    Xn = np.asarray(np.load(CACHE / TRAIN.name / "nl.npy", mmap_mode="r")[:, tlayers.index(HIDDEN)], dtype=np.float64)[keep]
    tn = np.array([o["is_answer"] for o in opts], float)[keep]
    qn = np.array([o["q"] for o in opts])[keep]
    stn = np.flatnonzero(np.r_[True, qn[1:] != qn[:-1]])
    n_drop = len({opts[i]["item"] for i in np.flatnonzero(~keep)})
    print(f"{len(cases)} hint-first cases, {X.shape[0]} options; no-hint pool {len(stn)} questions ({n_drop} eval items dropped)")

    def sub(idx):  # case indices -> option rows, question starts within them
        rows = np.concatenate([np.arange(starts[i], starts[i] + n[i]) for i in idx])
        return rows, np.r_[0, np.cumsum(n[idx])[:-1]]

    def fit(m, tr, l2):  # -> predict(case indices) -> pmf over their options
        rows, st = sub(tr)
        Xt, tt, ht = X[rows], t[rows], hinted[rows]
        if m["pool"]:
            Xt, tt, ht, st = np.vstack([Xt, Xn]), np.r_[tt, tn], np.r_[ht, np.zeros(len(tn))], np.r_[st, len(rows) + stn]
        if m["prior"]:
            mu, sd = pr["mean"], pr["scale"]
        else:
            sc = StandardScaler().fit(Xt)
            mu, sd = sc.mean_, sc.scale_

        def feat(Xs, hs):
            Z = (Xs - mu) / sd
            return np.column_stack([Z, hs]) if m["ind"] else Z

        w0 = np.r_[pr["w"], np.zeros(int(m["ind"]))] if m["prior"] else None
        w = _fit_listwise(feat(Xt, ht), tt, st, l2, n_free=int(m["ind"]), w0=w0)

        def predict(te):
            r, s = sub(te)
            return _softmax_by_q(feat(X[r], hinted[r]) @ w, s)

        return predict

    def nll(p, idx):
        r, _ = sub(idx)
        return -np.sum(t[r] * np.log(p + 1e-12)) / len(idx)

    methods = {
        "scratch": {"prior": False, "pool": False, "ind": False},
        "fine-tuned": {"prior": True, "pool": False, "ind": False},
        "pooled": {"prior": False, "pool": True, "ind": False},
        "scratch +hinted": {"prior": False, "pool": False, "ind": True},
        "fine-tuned +hinted": {"prior": True, "pool": False, "ind": True},
        "pooled +hinted": {"prior": False, "pool": True, "ind": True},
    }
    folds = list(GroupKFold(n_splits=a.folds).split(cases, groups=groups))

    def outer(name, k):
        m, (tr, te) = methods[name], folds[k]
        inner = list(GroupKFold(n_splits=a.inner).split(tr, groups=groups[tr]))
        cv = {l2: float(np.mean([nll(fit(m, tr[i], l2)(tr[j]), tr[j]) for i, j in inner])) for l2 in L2_GRID}
        l2 = min(cv, key=lambda x: cv[x])
        return name, k, l2, cv, fit(m, tr, l2)(te)

    with ThreadPoolExecutor(a.jobs) as ex:  # numpy releases the GIL in the matmuls
        jobs = list(ex.map(lambda nk: outer(*nk), [(name, k) for name in methods for k in range(len(folds))]))
    P = {"frozen": _softmax_by_q((X - pr["mean"]) / pr["scale"] @ pr["w"], starts)}
    # hinted letter: p(hinted) = the training folds' share of knew, the rest uniform
    P["hinted letter"] = np.zeros(len(X))
    for tr, te in folds:
        c = knew[tr].mean()
        r, _ = sub(te)
        P["hinted letter"][r] = np.where(hinted[r] == 1, c, (1 - c) / np.repeat(n[te] - 1, n[te]))
    # frozen recalibrated: two unpenalised weights (frozen score, is hinted) fit on the training folds, i.e. a temperature
    # and a hinted-option offset over the frozen probe: what's left once the hint's average pull is subtracted
    s0 = np.column_stack([(X - pr["mean"]) / pr["scale"] @ pr["w"], hinted])
    P["frozen recalibrated"] = np.zeros(len(X))
    for tr, te in folds:
        (rt, st), (r, se) = sub(tr), sub(te)
        P["frozen recalibrated"][r] = _softmax_by_q(s0[r] @ _fit_listwise(s0[rt], t[rt], st, 0.0), se)
    chosen = {}
    for name, k, l2, cv, p in jobs:
        r, _ = sub(folds[k][1])
        P.setdefault(name, np.zeros(len(X)))[r] = p
        chosen.setdefault(name, {})[k] = {"l2": l2, "inner_nll": {str(x): round(float(v), 4) for x, v in cv.items()}}

    # per-case quantities, then metrics over case subsets with a stratified bootstrap over questions
    def per_case(p):
        pm = np.array([p[s + LETTERS.index(m)] for s, m in zip(starts, modal)])
        ph = np.array([p[s + h] for s, h in zip(starts, hint)])
        top = np.array([int(np.argmax(p[s : s + k])) for s, k in zip(starts, n)])
        return {
            "top1": top == np.array([LETTERS.index(m) for m in modal]),
            "nll": -np.log(pm + 1e-12),
            "p_hint": ph,
            "top_hint": top == np.array(hint),
        }

    def metrics(v, idx):
        k, d = idx[knew[idx]], idx[~knew[idx]]
        out = {
            "top1 knew": v["top1"][k].mean(),
            "top1 didn't": v["top1"][d].mean(),
            "top1 bal": (v["top1"][k].mean() + v["top1"][d].mean()) / 2,
            "NLL knew": v["nll"][k].mean(),
            "NLL didn't": v["nll"][d].mean(),
            "NLL bal": (v["nll"][k].mean() + v["nll"][d].mean()) / 2,
            "didn't top1=hinted": v["top_hint"][d].mean(),
        }
        out["AUROC p(hinted)"] = roc_auc_score(knew[idx], v["p_hint"][idx]) if len(k) and len(d) else np.nan
        return out

    rng = np.random.default_rng(0)
    V = {name: per_case(p) for name, p in P.items()}

    def boot(idx):
        k, d = idx[knew[idx]], idx[~knew[idx]]
        return [np.r_[rng.choice(k, len(k)), rng.choice(d, len(d))] for _ in range(a.boot)]

    def table(idx, cols):
        bs = boot(idx)
        lines = ["| method | " + " | ".join(cols) + " |", "| --- " * (len(cols) + 1) + "|"]
        for name, v in V.items():
            pt = metrics(v, idx)
            draws = [metrics(v, b) for b in bs]
            cells = []
            for col in cols:
                x = np.array([dr[col] for dr in draws])
                lo, hi = np.nanpercentile(x, [2.5, 97.5])
                cells.append(f"{pt[col]:.3f} ±{(hi - lo) / 2:.3f}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        return lines

    allc = np.arange(len(cases))
    cols = ["top1 knew", "top1 didn't", "top1 bal", "NLL bal", "AUROC p(hinted)", "didn't top1=hinted"]
    report = [
        (
            f"Hint-first probe tuning on {src.name}: {len(cases)} cases ({knew.sum()} knew / {(~knew).sum()} didn't), target ="
            f" Qwen's no-hint modal answer, nl hidden_states[{HIDDEN}], {a.folds}-fold GroupKFold by question, l2 by inner"
            f" {a.inner}-fold over {L2_GRID}. ± = half-width of the 95% bootstrap interval over questions ({a.boot} draws,"
            " stratified by class)."
        ),
        "",
        "## All sources",
        "",
        *table(allc, cols),
        "",
        "NLL by class:",
        "",
        *table(allc, ["NLL knew", "NLL didn't"]),
        "",
    ]
    for s in sorted(set(source)):
        report += [f"## {s} ({(source == s).sum()} cases)", "", *table(np.flatnonzero(source == s), cols), ""]
    report += ["## Chosen l2 per outer fold", ""]
    report += [f"- {name}: {[chosen[name][k]['l2'] for k in range(len(folds))]}" for name in methods]
    run = new_run(
        "mcq_answer_probe",
        a.slug,
        description="MCQ answer probe tuned on hint-first activations to read Qwen's no-hint answer (scratch / fine-tuned / pooled)",
        models=[{"name": MODEL, "served_by": f"cached activations of {src.name} and {TRAIN.name}"}],
        params=vars(a) | {"hidden": HIDDEN, "l2_grid": L2_GRID, "chosen_l2": chosen, "n_pool_questions": len(stn)},
    )
    (run / "tune.md").write_text("\n".join(report) + "\n")
    fold_of = {int(i): k for k, (_, te) in enumerate(folds) for i in te}
    with (run / "tune_preds.jsonl").open("w") as f:
        for i, c in enumerate(cases):
            sl = slice(starts[i], starts[i] + n[i])
            pmfs = {name: [round(float(x), 4) for x in p[sl]] for name, p in P.items()}
            row = {k: c[k] for k in ("version", "item", "class", "gold")} | {"hinted": LETTERS[hint[i]], "modal_cf": modal[i]}
            row |= {"fold": fold_of[i], "pmf": pmfs}
            f.write(json.dumps(row) + "\n")
    print("\n".join(report))
    print(run)
    if a.finish:
        from counterfactual_investigator_agent_with_olens.runs import finish

        finish(run)


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
    t = sub.add_parser("tune")
    t.add_argument("run", help="a hint-first (before_*) run of `prep` + `score`, its nl activations in data/probe_cache/<run name>")
    t.add_argument("--slug", default="hint-first-tune")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--inner", type=int, default=4)
    t.add_argument("--boot", type=int, default=2000)
    t.add_argument("--jobs", type=int, default=8)
    t.add_argument("--finish", action="store_true")
    a = ap.parse_args()
    {"prep": prep, "score": score, "cfbench-runs": cfbench_runs, "tune": tune}[a.cmd](a)


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        main()
