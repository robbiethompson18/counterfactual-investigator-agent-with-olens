"""Qwen's answers and the linear probes for the author MCQ run.

    uv run python -m scripts.author_probe.fit answers RUN       # RUN/qwen_answers.jsonl + qwen.md (Qwen vs label, confidence)
    uv run --with torch --with scikit-learn python -m scripts.author_probe.fit probes RUN [--device cuda]

`answers` reads RUN/answer_logp.jsonl (scripts/hs_at_positions.py on inputs_answer.jsonl): letter log-probs per option
order, renormalised over the question's letters, mapped back to the canonical options and averaged over the two orders.
`probes` reads the passage-prompt activations in data/probe_cache/<run name>/{text_end,im_end}.npy ([passages, layers, d])
and fits a multinomial logistic regression per (site, layer, attribute, target) with standardised features, L2 at
sklearn-equivalent C, 5-fold GroupKFold by author; targets = Qwen's answer (argmax of its pmf) and the dataset label.
Writes probes.json and probes.md (per-layer AUROC/accuracy, best (site, layer) per attribute, all layers and every 4th).
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from environments._lib.runfolder import REPO_ROOT
from scripts.author_probe.bank import ATTRS, BANK

CACHE = REPO_ROOT / "data" / "probe_cache"
SITES = ("text_end", "im_end")  # passage-only prompt
Q_SITES = ("qmark", "last")  # order-0 answer prompt, one capture part per attribute (bank.py qprompts)
MIN_CLASS = 10  # a class enters macro AUROC only with >= this many positives


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def answers(a):
    run = Path(a.run)
    qs = load(BANK / "questions.jsonl")
    logp = {r["id"]: r["logp"] for r in load(run / "answer_logp.jsonl")}
    order_of = {r["id"]: r["order"] for r in load(run / "inputs_answer.jsonl")}
    rows, lines = [], []
    for q in qs:
        per = []
        for o in (0, 1):
            k = f"{q['id']}:o{o}"
            lp = np.array(logp[k])
            p = np.exp(lp - lp.max())
            p /= p.sum()
            canon = np.zeros(len(q["options"]))
            canon[order_of[k]] = p
            per.append({"pmf": canon.tolist(), "mass": float(np.exp(lp).sum())})
        pmf = np.mean([x["pmf"] for x in per], 0)
        rows.append(
            {k: q[k] for k in ("id", "passage", "author", "attr", "label")}
            | {"pmf": pmf.tolist(), "answer": int(pmf.argmax()), "orders": per}
            | {"order_agree": int(np.argmax(per[0]["pmf"])) == int(np.argmax(per[1]["pmf"])), "human_n": q["human"]["n"]}
        )
    (run / "qwen_answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    lines += [
        (
            "| attribute | n | Qwen acc | majority acc | Qwen answer dist | label dist | p(label) mean | top p median | orders agree"
            " | letter mass |"
        ),
        "|" + "---|" * 10,
    ]
    for attr, (_, opts) in ATTRS.items():
        rs = [r for r in rows if r["attr"] == attr]
        lab, ans = Counter(r["label"] for r in rs), Counter(r["answer"] for r in rs)
        acc = np.mean([r["answer"] == r["label"] for r in rs])
        lines.append(
            f"| {attr} | {len(rs)} | {acc:.3f} | {max(lab.values()) / len(rs):.3f} | {[ans[i] for i in range(len(opts))]} | "
            f"{[lab[i] for i in range(len(opts))]} | {np.mean([r['pmf'][r['label']] for r in rs]):.3f} | "
            f"{np.median([max(r['pmf']) for r in rs]):.3f} | {np.mean([r['order_agree'] for r in rs]):.3f} | "
            f"{np.median([x['mass'] for r in rs for x in r['orders']]):.3f} |"
        )
    lines += ["", "Qwen accuracy by whether the SynthPAI human reviewer guessed the attribute from any of the passage's comments:", ""]
    lines += ["| attribute | acc, human guessed | n | acc, not guessed | n |", "|---|---|---|---|---|"]
    for attr in ATTRS:
        g = [r["answer"] == r["label"] for r in rows if r["attr"] == attr and r["human_n"]]
        ng = [r["answer"] == r["label"] for r in rows if r["attr"] == attr and not r["human_n"]]
        lines.append(f"| {attr} | {np.mean(g):.3f} | {len(g)} | {np.mean(ng):.3f} | {len(ng)} |")
    lines += ["", "Top-option probability (Qwen's confidence), deciles:", ""]
    for attr in ATTRS:
        top = sorted(max(r["pmf"]) for r in rows if r["attr"] == attr)
        lines.append(f"- {attr}: " + " ".join(f"{top[int(i * (len(top) - 1) / 10)]:.2f}" for i in range(11)))
    (run / "qwen.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def fit_layers(X, y, train, test, n_cls: int, C: float, device: str):
    """Multinomial L2 logistic regression on every layer at once (independent problems, one joint L-BFGS). X [n, L, d]
    float16 numpy; returns held-out probabilities [L, n_test, n_cls]."""
    import torch

    Xtr = torch.tensor(X[train], device=device, dtype=torch.float32).transpose(0, 1)  # [L, n, d]
    mu, sd = Xtr.mean(1, keepdim=True), Xtr.std(1, keepdim=True) + 1e-4
    Xtr = (Xtr - mu) / sd
    Xte = (torch.tensor(X[test], device=device, dtype=torch.float32).transpose(0, 1) - mu) / sd
    ytr = torch.tensor(y[train], device=device)
    L, n, d = Xtr.shape
    W = torch.zeros(L, d, n_cls, device=device, requires_grad=True)
    b = torch.zeros(L, 1, n_cls, device=device, requires_grad=True)
    lam = 1.0 / (C * n)  # sklearn's C * sum(CE) + |w|^2 / 2, divided by C * n
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=500, tolerance_grad=1e-6, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = Xtr @ W + b
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, n_cls), ytr.repeat(L), reduction="sum") / n
        loss = loss + lam / 2 * (W**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return torch.softmax(Xte @ W + b, -1).cpu().numpy()


def scores(p: np.ndarray, y: np.ndarray, n_cls: int) -> dict:
    from sklearn.metrics import roc_auc_score

    cls = [k for k in range(n_cls) if MIN_CLASS <= (y == k).sum() <= len(y) - MIN_CLASS]
    auc = np.mean([roc_auc_score(y == k, p[:, k]) for k in cls]) if cls else float("nan")
    return {"auroc": float(auc), "acc": float((p.argmax(1) == y).mean()), "classes": cls}


def ridge_layers(X, y, tr, te, groups, alphas, device: str):
    """Ridge regression on every layer, standardised features, alpha chosen per layer by an inner 3-fold GroupKFold on the
    training fold (kernel form: n << d). Returns held-out predictions [L, n_test] and the chosen alphas."""
    import torch
    from sklearn.model_selection import GroupKFold

    def fit_predict(itr, ite):
        A = torch.tensor(X[itr], device=device, dtype=torch.float32).transpose(0, 1)
        mu, sd = A.mean(1, keepdim=True), A.std(1, keepdim=True) + 1e-4
        A = (A - mu) / sd
        B = (torch.tensor(X[ite], device=device, dtype=torch.float32).transpose(0, 1) - mu) / sd
        yt = torch.tensor(y[itr], device=device, dtype=torch.float32)
        ym = yt.mean()
        lam, U = torch.linalg.eigh(A @ A.transpose(1, 2))  # [L, n], [L, n, n]
        Uy = U.transpose(1, 2) @ (yt - ym)[None, :, None]  # [L, n, 1]
        Kte = B @ A.transpose(1, 2)  # [L, m, n]
        return torch.stack([(Kte @ (U @ (Uy / (lam[..., None] + al))))[..., 0] + ym for al in alphas])  # [alphas, L, m]

    inner = np.zeros((len(alphas), X.shape[1]))
    for itr, ite in GroupKFold(n_splits=3).split(tr, groups=groups[tr]):
        pred = fit_predict(tr[itr], tr[ite]).cpu().numpy()
        inner += ((pred - y[tr[ite]]) ** 2).sum(-1)
    pick = inner.argmin(0)  # per layer
    pred = fit_predict(tr, te).cpu().numpy()
    return pred[pick, np.arange(X.shape[1])], [alphas[i] for i in pick]


def ridge_scores(pred: np.ndarray, y: np.ndarray, cuts: list) -> dict:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    return {
        "spearman": float(spearmanr(pred, y)[0]),
        "mae": float(np.abs(pred - y).mean()),
        "cut_auroc": [float(roc_auc_score(y >= c, pred)) for c in cuts],
    }


RIDGE = {"age": ([30, 45, 60], "years"), "education": ([1, 2, 3], "bin rank 0-3")}


def probes(a):
    from sklearn.model_selection import GroupKFold

    run = Path(a.run)
    cache = CACHE / run.name
    layers = json.loads((cache / "layers.json").read_text())
    pids = [r["id"] for r in load(run / "passage_logp.jsonl")]  # hs_at_positions row order = activation row order
    ans = load(run / "qwen_answers.jsonl")
    age = {p["id"]: p["profile"]["age"] for p in load(BANK / "passages.jsonl")}
    authors = np.array([pid.rsplit("_", 1)[0] for pid in pids])
    folds = list(GroupKFold(n_splits=5).split(pids, groups=authors))
    old = json.loads((run / "probes.json").read_text()) if (run / "probes.json").exists() else None
    res: dict = old["results"] if old and old["C"] == a.C else {}
    for site in a.sites:
        for k, (attr, (_, opts)) in enumerate(ATTRS.items()):
            path = cache / f"{site}.npy" if site in SITES else cache / "qsites" / f"part-{k:04d}" / f"{site}.npy"
            X = np.load(path, mmap_mode="r")
            by = {r["passage"]: r for r in ans if r["attr"] == attr}
            targets = {"label": np.array([by[p]["label"] for p in pids]), "qwen": np.array([by[p]["answer"] for p in pids])}
            for tname, y in targets.items():
                if tname not in a.targets:
                    continue
                for C in a.C:
                    oof = np.zeros((len(layers), len(pids), len(opts)))
                    for tr, te in folds:
                        oof[:, te] = fit_layers(X, y, tr, te, len(opts), C, a.device)
                    rows = [scores(oof[li], y, len(opts)) for li in range(len(layers))]
                    if tname == "label":  # the primary probe: how does it relate to what Qwen answers?
                        for li, r in enumerate(rows):
                            vs = scores(oof[li], targets["qwen"], len(opts))
                            r |= {"agree_qwen": vs["acc"], "auroc_qwen": vs["auroc"], "classes_qwen": vs["classes"]}
                    res[f"{site}|{attr}|{tname}|{C}"] = rows
                    best = max(range(len(layers)), key=lambda li: rows[li]["auroc"])
                    print(f"{site} {attr} {tname} C={C}: best L{layers[best]} {rows[best]}", flush=True)
                    if C == a.C[0]:
                        np.save(cache / f"oof_{site}_{attr}_{tname}.npy", oof.astype(np.float16))
            # reference: Qwen's own pmf as a predictor of the label
            pmf = np.array([by[p]["pmf"] for p in pids])
            res[f"qwen_pmf|{attr}|label"] = scores(pmf, targets["label"], len(opts))
            if attr in RIDGE:
                cuts = RIDGE[attr][0]
                y = np.array([age[p] for p in pids], float) if attr == "age" else targets["label"].astype(float)
                oof = np.zeros((len(layers), len(pids)))
                alphas = []
                for tr, te in folds:
                    oof[:, te], al = ridge_layers(X, y, tr, te, authors, a.alphas, a.device)
                    alphas.append(al)
                res[f"{site}|{attr}|ridge"] = [
                    ridge_scores(oof[li], y, cuts) | {"alphas": [f[li] for f in alphas]} for li in range(len(layers))
                ]
                # reference: Qwen's expected bin (pmf-weighted option index) against the same numeric target
                res[f"qwen_pmf|{attr}|ridge"] = ridge_scores(pmf @ np.arange(len(opts)), y, cuts) | {"note": "Qwen expected bin index"}
                best = max(range(len(layers)), key=lambda li: res[f"{site}|{attr}|ridge"][li]["spearman"])
                print(f"{site} {attr} ridge: best L{layers[best]} {res[f'{site}|{attr}|ridge'][best]}", flush=True)
    (run / "probes.json").write_text(json.dumps({"layers": layers, "C": a.C, "results": res}) + "\n")
    report(run)


def report(run: Path):
    d = json.loads((run / "probes.json").read_text())
    layers, res, C = d["layers"], d["results"], d["C"][0]
    qa = load(run / "qwen_answers.jsonl")
    sites = [s for s in SITES + Q_SITES if f"{s}|sex|label|{C}" in res]
    cands = [(s, li) for s in sites for li in range(len(layers))]
    bow = json.loads((run / "bow.json").read_text()) if (run / "bow.json").exists() else {}

    def best(key_fn, metric, every4=False, qwen_only=False):
        cs = [c for c in cands if (not every4 or layers[c[1]] % 4 == 0) and (not qwen_only or f"{c[0]}|sex|qwen|{C}" in res)]
        return max(cs, key=lambda c: key_fn(c[0])[c[1]][metric])

    lines = [
        (
            "Out-of-sample: 5-fold GroupKFold by author. Multinomial logistic regression over the MCQ options, standardised features,"
            f" C = {C}. AUROC = macro one-vs-rest over classes with >= {MIN_CLASS} positives. Best = highest AUROC over 2 sites x 65"
            " layers (or the every-4th-layer subset), chosen on the same held-out predictions (mildly optimistic)."
        ),
        "",
        "## Label probe by site (best layer per site, AUROC vs label)",
        "",
        "| attribute | " + " | ".join(sites) + " | Qwen's own pmf | TF-IDF text baseline (C=10) |",
        "|" + "---|" * (len(sites) + 3),
    ]
    for attr in ATTRS:
        per = [max(r["auroc"] for r in res[f"{s}|{attr}|label|{C}"]) for s in sites]
        tf = bow.get(f"{attr}|label|10.0", {}).get("auroc", float("nan"))
        lines.append(f"| {attr} | " + " | ".join(f"{v:.3f}" for v in per) + f" | {res[f'qwen_pmf|{attr}|label']['auroc']:.3f} | {tf:.3f} |")
    lines += [
        "",
        "## Primary: probe trained on the dataset label",
        "",
        (
            "| attribute | best (site, layer) | AUROC vs label | acc vs label | argmax = Qwen's answer | AUROC vs Qwen's answer"
            " | best, every 4th layer | AUROC vs label | Qwen acc vs label | Qwen pmf AUROC vs label | majority acc |"
        ),
        "|" + "---|" * 11,
    ]
    for attr in ATTRS:
        f = lambda s, attr=attr: res[f"{s}|{attr}|label|{C}"]
        b, b4 = best(f, "auroc"), best(f, "auroc", True)
        r, r4, q = f(b[0])[b[1]], f(b4[0])[b4[1]], res[f"qwen_pmf|{attr}|label"]
        maj = max(Counter(x["label"] for x in qa if x["attr"] == attr).values()) / sum(x["attr"] == attr for x in qa)
        lines.append(
            f"| {attr} | {b[0]} L{layers[b[1]]} | {r['auroc']:.3f} | {r['acc']:.3f} | {r['agree_qwen']:.3f} | {r['auroc_qwen']:.3f} | "
            f"{b4[0]} L{layers[b4[1]]} | {r4['auroc']:.3f} | {q['acc']:.3f} | {q['auroc']:.3f} | {maj:.3f} |"
        )
    lines += ["", "## Single-axis ridge (age in years, education bin rank 0-3)", ""]
    lines += [
        (
            "Ridge per layer, standardised features, alpha by inner 3-fold GroupKFold. Best layer by Spearman. Cut AUROCs threshold the"
            " score at each bin boundary (age >= 30 / 45 / 60; education >= some college / bachelor's / graduate). Reference row:"
            " Qwen's expected option index under its pmf."
        ),
        "",
        (
            "| attribute | best (site, layer) | Spearman | MAE | cut AUROCs | best, every 4th | Spearman | Qwen E[bin] Spearman"
            " | Qwen cut AUROCs |"
        ),
        "|" + "---|" * 9,
    ]
    for attr, (_, unit) in RIDGE.items():
        f = lambda s, attr=attr: res[f"{s}|{attr}|ridge"]
        b, b4 = best(f, "spearman"), best(f, "spearman", True)
        r, r4, q = f(b[0])[b[1]], f(b4[0])[b4[1]], res[f"qwen_pmf|{attr}|ridge"]
        cut = lambda xs: " / ".join(f"{x:.3f}" for x in xs)
        lines.append(
            f"| {attr} ({unit}) | {b[0]} L{layers[b[1]]} | {r['spearman']:.3f} | {r['mae']:.2f} | {cut(r['cut_auroc'])} | "
            f"{b4[0]} L{layers[b4[1]]} | {r4['spearman']:.3f} | {q['spearman']:.3f} | {cut(q['cut_auroc'])} |"
        )
    lines += ["", "## Secondary: probe trained on Qwen's own answer", ""]
    lines += ["| attribute | best (site, layer) | AUROC vs Qwen answer | acc | best, every 4th layer | AUROC | acc |", "|" + "---|" * 7]
    for attr in ATTRS:
        f = lambda s, attr=attr: res[f"{s}|{attr}|qwen|{C}"]
        b, b4 = best(f, "auroc", qwen_only=True), best(f, "auroc", True, qwen_only=True)
        r, r4 = f(b[0])[b[1]], f(b4[0])[b4[1]]
        lines.append(
            f"| {attr} | {b[0]} L{layers[b[1]]} | {r['auroc']:.3f} | {r['acc']:.3f} | "
            f"{b4[0]} L{layers[b4[1]]} | {r4['auroc']:.3f} | {r4['acc']:.3f} |"
        )
    for t, name in (("label", "label probe, AUROC vs label"), ("qwen", "Qwen-answer probe, AUROC vs Qwen's answer")):
        lines += ["", f"## Per layer: {name} (every 4th layer)", ""]
        ss = [s for s in sites if f"{s}|sex|{t}|{C}" in res]
        cols = [f"{s}:{attr}" for attr in ATTRS for s in ss]
        lines += ["| layer | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
        for li, L in enumerate(layers):
            if L % 4 == 0:
                vals = [res[f"{s}|{attr}|{t}|{C}"][li]["auroc"] for attr in ATTRS for s in ss]
                lines.append(f"| {L} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    lines += ["", "## Per layer: ridge Spearman (every 4th layer)", ""]
    cols = [f"{s}:{attr}" for attr in RIDGE for s in sites]
    lines += ["| layer | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for li, L in enumerate(layers):
        if L % 4 == 0:
            lines.append(
                f"| {L} | " + " | ".join(f"{res[f'{s}|{attr}|ridge'][li]['spearman']:.3f}" for attr in RIDGE for s in sites) + " |"
            )
    if len(d["C"]) > 1:
        lines += ["", "## Regularisation check (best layer per C)", ""]
        for key in sorted({k.rsplit("|", 1)[0] for k in res if k.count("|") == 3}):
            bests = {c: max(r["auroc"] for r in res[f"{key}|{c}"]) for c in d["C"]}
            lines.append(f"- {key}: " + ", ".join(f"C={c}: {v:.3f}" for c, v in bests.items()))
    (run / "probes.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def qprobes(a):
    """The `last` site (final prompt token, after the options): both option orders pooled with labels in concept space,
    both orders of a passage in the same author fold; a cross-order test (fit on one order, test on the other, held-out
    authors); and a letter control (target = the letter of the gold option, pooled over orders). Writes qprobes.json/md."""
    from sklearn.model_selection import GroupKFold

    run = Path(a.run)
    cache = CACHE / run.name
    layers = json.loads((cache / "layers.json").read_text())
    pids = [r["id"] for r in load(run / "passage_logp.jsonl")]
    ans = load(run / "qwen_answers.jsonl")
    age = {p["id"]: p["profile"]["age"] for p in load(BANK / "passages.jsonl")}
    authors = np.array([pid.rsplit("_", 1)[0] for pid in pids])
    n = len(pids)
    folds = list(GroupKFold(n_splits=5).split(pids, groups=authors))
    res, lines = {}, []
    for k, (attr, (_, opts)) in enumerate(ATTRS.items()):
        K = len(opts)
        X = np.concatenate([np.load(cache / d / f"part-{k:04d}" / "last.npy") for d in ("qsites", "qsites_o1")])  # [2n, L, d]
        by = {r["passage"]: r for r in ans if r["attr"] == attr}
        y = np.array([by[p]["label"] for p in pids])
        y2, letter = np.concatenate([y, y]), np.concatenate([y, K - 1 - y])  # order 1 reverses the options
        pooled = [(np.concatenate([tr, tr + n]), np.concatenate([te, te + n])) for tr, te in folds]
        out = {}
        for name, target, splits in (
            ("pooled", y2, pooled),
            ("letter", letter, pooled),
            ("o0->o1", y2, [(tr, te + n) for tr, te in folds]),
            ("o1->o0", y2, [(tr + n, te) for tr, te in folds]),
        ):
            oof = np.zeros((len(layers), 2 * n, K))
            for tr, te in splits:
                oof[:, te] = fit_layers(X, target, tr, te, K, a.C, a.device)
            test = np.concatenate([te for _, te in splits])
            out[name] = [scores(oof[li][test], target[test], K) for li in range(len(layers))]
            b = max(range(len(layers)), key=lambda li: out[name][li]["auroc"])
            print(f"last {attr} {name}: best L{layers[b]} {out[name][b]}", flush=True)
        if attr in RIDGE:
            yr = np.array([age[p] for p in pids], float) if attr == "age" else y.astype(float)
            yr2, oof = np.concatenate([yr, yr]), np.zeros((len(layers), 2 * n))
            for tr, te in pooled:
                oof[:, te] = ridge_layers(X, yr2, tr, te, np.concatenate([authors, authors]), a.alphas, a.device)[0]
            out["ridge"] = [ridge_scores(oof[li], yr2, RIDGE[attr][0]) for li in range(len(layers))]
        res[attr] = out
    (run / "qprobes.json").write_text(json.dumps({"layers": layers, "C": a.C, "results": res}) + "\n")
    lines += [
        (
            f"`last` site (final prompt token), both option orders, C = {a.C}, 5-fold GroupKFold by author (both orders of a passage"
            " in one fold). AUROC = macro one-vs-rest; best layer per row chosen on the same held-out predictions."
        ),
        "",
        "| attribute | pooled, AUROC vs label (best L) | o0→o1 | o1→o0 | letter control, AUROC vs gold letter |",
        "|---|---|---|---|---|",
    ]
    for attr, (_, opts) in ATTRS.items():

        def cell(name: str, attr: str = attr) -> str:
            b = max(range(len(layers)), key=lambda li: res[attr][name][li]["auroc"])
            return f"{res[attr][name][b]['auroc']:.3f} (L{layers[b]})"

        lines.append(f"| {attr} | {cell('pooled')} | {cell('o0->o1')} | {cell('o1->o0')} | {cell('letter')} |")
    for attr in RIDGE:
        b = max(range(len(layers)), key=lambda li: res[attr]["ridge"][li]["spearman"])
        r = res[attr]["ridge"][b]
        lines.append(
            f"\n- ridge {attr}, pooled: Spearman {r['spearman']:.3f} (L{layers[b]}), MAE {r['mae']:.2f}, cut AUROCs {r['cut_auroc']}"
        )
    lines += ["", "Per layer (every 4th): pooled / o0→o1 / letter AUROC", ""]
    lines += ["| layer | " + " | ".join(ATTRS) + " |", "|" + "---|" * (len(ATTRS) + 1)]
    for li, L in enumerate(layers):
        if L % 4 == 0:
            vals = [
                f"{res[t]['pooled'][li]['auroc']:.2f} / {res[t]['o0->o1'][li]['auroc']:.2f} / {res[t]['letter'][li]['auroc']:.2f}"
                for t in ATTRS
            ]
            lines.append(f"| {L} | " + " | ".join(vals) + " |")
    (run / "qprobes.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def bow(a):
    """Text-only control: TF-IDF (word 1-2 grams) logistic regression on the passage, same author folds and targets."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    run = Path(a.run)
    text = {p["id"]: p["text"] for p in load(BANK / "passages.jsonl")}
    pids = [r["id"] for r in load(run / "passage_logp.jsonl")]
    ans = load(run / "qwen_answers.jsonl")
    authors = [pid.rsplit("_", 1)[0] for pid in pids]
    folds = list(GroupKFold(n_splits=5).split(pids, groups=authors))
    out, lines = {}, ["| attribute | target | C | AUROC | acc |", "|---|---|---|---|---|"]
    for attr, (_, opts) in ATTRS.items():
        by = {r["passage"]: r for r in ans if r["attr"] == attr}
        for tname, key in (("label", "label"), ("qwen", "answer")):
            y = np.array([by[p][key] for p in pids])
            for C in (1.0, 10.0):
                oof = np.zeros((len(pids), len(opts)))
                for tr, te in folds:
                    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
                    Xtr = vec.fit_transform([text[pids[i]] for i in tr])
                    m = LogisticRegression(C=C, max_iter=2000).fit(Xtr, y[tr])
                    oof[np.ix_(te, m.classes_)] = m.predict_proba(vec.transform([text[pids[i]] for i in te]))
                out[f"{attr}|{tname}|{C}"] = r = scores(oof, y, len(opts))
                lines.append(f"| {attr} | {tname} | {C} | {r['auroc']:.3f} | {r['acc']:.3f} |")
    (run / "bow.json").write_text(json.dumps(out) + "\n")
    (run / "bow.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("answers")
    s.add_argument("run")
    p = sub.add_parser("probes")
    p.add_argument("run")
    p.add_argument("--C", type=float, nargs="+", default=[0.01])
    p.add_argument("--device", default="cpu")
    p.add_argument("--targets", nargs="+", default=["label", "qwen"])
    p.add_argument("--sites", nargs="+", default=list(SITES), help=f"any of {SITES + Q_SITES}")
    p.add_argument("--alphas", type=float, nargs="+", default=[1e2, 1e3, 1e4, 1e5, 1e6])
    qp = sub.add_parser("qprobes")
    qp.add_argument("run")
    qp.add_argument("--C", type=float, default=0.01)
    qp.add_argument("--device", default="cpu")
    qp.add_argument("--alphas", type=float, nargs="+", default=[1e2, 1e3, 1e4, 1e5, 1e6])
    w = sub.add_parser("bow")
    w.add_argument("run")
    r = sub.add_parser("report")
    r.add_argument("run")
    a = ap.parse_args()
    {"answers": answers, "probes": probes, "bow": bow, "qprobes": qprobes, "report": lambda a: report(Path(a.run))}[a.cmd](a)


if __name__ == "__main__":
    main()
