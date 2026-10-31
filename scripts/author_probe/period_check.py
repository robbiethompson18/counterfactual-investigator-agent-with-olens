"""The sentence-end probe in the standard recipe, and whether its increments over Haiku are honest out of fold (Kylie,
2026-10-31). In period_probe.py the probe is PCA 128 + logistic, and the stacker's training rows carry probe scores from
probes whose training folds include the stacker's test fold. Here:

  probe      the standard probe (docs/standard-probe-protocol.md, probes/fit.py's fit_model/predict): training-only
             per-dimension standardization, unpenalized intercept, no PCA or whitening, fixed L2 (mean cross-entropy +
             lambda/2 ||w||^2, lambda = 1), as a two-option softmax; on the 5,120-dim h at layer 48 (fixed after
             period_probe's inner CV chose it in all 5 folds), and on h(k) - h(k-1). Score: P(significant).
  nested     per outer fold (5 folds over bloggers): inner out-of-fold probe scores for the outer training rows (5 folds), the
             combiner (logistic on [logit Haiku transcript p, probe score], and with the sentence length/position features)
             fit on them, the outer test rows scored with the probe refit on all outer training rows. Baseline: the same
             combiner without the probe, same folds.
  rank mean  no parameters: the mean of the ranks (or z-scores) of Haiku's p and the probe's out-of-fold score (and the
             length/position out-of-fold score from period_probe), against Haiku's p alone (and Haiku + length/position).
  seeds      the nested numbers for 5 shuffles of the bloggers into folds: mean ± SD.

Deviations from the protocol: 5-fold outer CV (not 2), a binary target ("deleting this sentence is significant", not
Qwen's letter), the eval's own site (the sentence's last token, not user_end / response_start), one fixed layer, no cap.

    uv run --with scikit-learn --with scipy python -m scripts.author_probe.period_check RUN   # appends to RUN/results.md
    uv run --with scikit-learn --with scipy python -m scripts.author_probe.period_check noq RUN ALT   # vs the post-alone capture
"""

import json
import sys
from pathlib import Path

import numpy as np

from probes.fit import fit_model, predict
from scripts.author_probe.blog import BANK, sentences
from scripts.author_probe.period_probe import INPUTS, activations, load, logit
from scripts.author_probe.pronoun_probe import PRED, auc, paired

LAYER = 48  # fixed after period_probe's inner CV chose it in all 5 folds
SEEDS = 5
FITS: list[dict] = []  # every probe fit's diagnostics (iterations, gradient, converged)


def folds(post: np.ndarray, k: int, seed: int) -> np.ndarray:
    us = np.random.default_rng(seed).permutation(np.unique(post))
    f = {u: i % k for i, u in enumerate(us)}
    return np.array([f[p] for p in post])


def lr():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def probe_score(X, y, tr, te) -> np.ndarray:
    """probes.fit's standard probe on X[tr] (two-option softmax, lambda 1), P(class 1) on X[te]."""
    rows = [{"label": int(v), "n_options": 2} for v in y[tr]]
    model, info = fit_model(X[tr], rows, essay=False)
    FITS.append(info)
    return predict(model, X[te], [{"n_options": 2}] * len(te), essay=False)[:, 1]


def nested(X, y, post, haiku, struct, seed) -> dict[str, np.ndarray]:
    """Out-of-fold scores: the probe alone, baseline (Haiku), + probe, Haiku + struct, + probe over Haiku + struct."""
    out = {k: np.zeros(len(y)) for k in ("probe_alone", "base", "probe", "base_struct", "probe_struct")}
    f = folds(post, 5, seed)
    for k in range(5):
        tr, te = np.flatnonzero(f != k), np.flatnonzero(f == k)
        inner = folds(post[tr], 5, seed + 1)
        s_tr = np.zeros(len(tr))
        for j in range(5):
            s_tr[inner == j] = probe_score(X, y, tr[inner != j], tr[inner == j])
        s_te = probe_score(X, y, tr, te)
        out["probe_alone"][te] = s_te
        for name, cols in (("base", haiku), ("base_struct", np.column_stack([haiku, struct]))):
            out[name][te] = lr().fit(cols[tr], y[tr]).predict_proba(cols[te])[:, 1]
            plus_tr, plus_te = np.column_stack([cols[tr], logit(s_tr)]), np.column_stack([cols[te], logit(s_te)])
            out[name.replace("base", "probe")][te] = lr().fit(plus_tr, y[tr]).predict_proba(plus_te)[:, 1]
        print(f"seed {seed} fold {k} done", flush=True)
    return out


def boot_diff(y, a, b, post, reps=2000) -> tuple[float, float, float, float]:
    """AUROC(a) - AUROC(b): estimate, 95% CI and one-sided p, resampling bloggers."""
    rng, us = np.random.default_rng(1), np.unique(post)
    idx = [np.flatnonzero(post == u) for u in us]
    d = []
    for _ in range(reps):
        i = np.concatenate([idx[j] for j in rng.integers(0, len(us), len(us))])
        d.append(auc(y[i], a[i]) - auc(y[i], b[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return auc(y, a) - auc(y, b), lo, hi, float(np.mean(np.array(d) <= 0))


def boot_ci(f, y, s, post, reps=2000) -> tuple[float, float]:
    rng, us = np.random.default_rng(2), np.unique(post)
    idx = [np.flatnonzero(post == u) for u in us]
    vals = []
    for _ in range(reps):
        pick = rng.integers(0, len(us), len(us))
        i = np.concatenate([idx[j] for j in pick])
        relabel = np.concatenate([[n] * len(idx[j]) for n, j in enumerate(pick)])
        vals.append(f(y[i], s[i], relabel) if f is paired else f(y[i], s[i]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def main(run: Path):
    from scipy.stats import rankdata

    A, index = activations(run)
    li = json.loads(next((run / "hs").glob("part-*/layers.json")).read_text()).index(LAYER)
    by = {r["id"]: r for r in index}
    preds: dict[str, dict] = {}
    for r in load(PRED):
        if r.get("valid", True) and r.get("p") is not None:
            preds.setdefault(r["item"], {})[r["condition"]] = r["p"]
    oof = {r["item"]: r for r in load(run / "scores.jsonl")}  # period_probe's out-of-fold scores (PCA probe, length/position)
    passages = {p["id"]: p for p in load(BANK / "passages.jsonl")}
    rows, items = [], []
    for c in load(INPUTS / "cases.jsonl"):
        if "transcript_inline" not in preds.get(c["item"], {}):
            continue
        ix = by[c["pair"]]
        k = int(c["item"].rsplit("#", 1)[1])  # the sentence index (a post can repeat a sentence's text)
        assert sentences(passages[c["pair"].split(":")[0]]["text"])[k] == c["comment"], c["item"]
        n = sum(kd.startswith("s") for kd in ix["kinds"])
        at = ix["kinds"].index(f"s{k}")
        length = len(c["comment"])
        rows.append((ix["rows"][at], ix["rows"][at - 1], int(c["positive"]), c["pair"].split(":")[0],
                     preds[c["item"]]["transcript_inline"], [np.log(length), length, k, n, k / n, k == n - 1, k == 0],
                     c["attr"], c["version"]))  # fmt: skip
        items.append(c["item"])
    H = A[[r[0] for r in rows], li].astype(np.float64)  # [cases, d] at L48
    D = H - A[[r[1] for r in rows], li].astype(np.float64)  # h(k) - h(k-1) (the question "?" before the first sentence)
    y = np.array([r[2] for r in rows])
    post = np.array([r[3] for r in rows])
    haiku = logit(np.array([r[4] for r in rows]))[:, None]
    struct = np.array([r[5] for r in rows], float)
    attr = np.array([r[6] for r in rows])
    version = np.array([r[7] for r in rows])

    res = {s: nested(H, y, post, haiku, struct, s) for s in range(SEEDS)}
    diff0 = nested(D, y, post, haiku, struct, 0)
    r0 = res[0]
    pca = np.array([oof[i]["probe_h"] for i in items])
    conv = sum(f["converged"] or f["gradient_max"] <= 2e-5 for f in FITS)
    out = [
        "\n## Standard L2 probe (no PCA) and increment checks (2026-10-31)\n",
        (
            f"{len(y)} cases with Haiku's p. Probe: probes/fit.py's standard fitter (training-only standardization, unpenalized "
            f"intercept, no PCA, mean cross-entropy + lambda/2 ||w||^2 with lambda = 1, two-option softmax) on the 5,120-dim h "
            f"at L{LAYER}, fixed after period_probe's inner CV chose it in all 5 folds: nested given the layer, but the layer (and "
            f"L2 rather than PCA 128) was chosen on this same eval set. Outer CV 5-fold over bloggers. "
            f"Deviations from docs/standard-probe-protocol.md: 5-fold not 2-fold, a binary target, the eval's own site (the "
            f"sentence's last token), one fixed layer, no training cap. {conv} of {len(FITS)} fits converged (L-BFGS, "
            f"max |grad| <= 2e-5).\n"
        ),
        "| probe (out-of-fold, seed 0) | AUROC [95% CI] | paired accuracy [95% CI] |\n|---|---|---|",
    ]
    for name, s in (("standard L2 on h", r0["probe_alone"]), ("standard L2 on h(k) - h(k-1)", diff0["probe_alone"]),
                    ("PCA 128 + logistic on h (period_probe, comparison)", pca)):  # fmt: skip
        (al, ah), (pl, ph) = boot_ci(auc, y, s, post), boot_ci(paired, y, s, post)
        out.append(f"| {name} | {auc(y, s):.3f} [{al:.3f}, {ah:.3f}] | {paired(y, s, post):.3f} [{pl:.3f}, {ph:.3f}] |")
    out.append("\nFully nested increments (seed 0): out-of-fold AUROC, 95% CI resampling bloggers, one-sided p (H1: + probe better).\n")
    out.append("| model | AUROC | baseline | difference [95% CI] | one-sided p |\n|---|---|---|---|---|")
    pairs = {"Haiku + probe vs Haiku": (r0, "probe", "base"),
             "Haiku + length/position + probe vs Haiku + length/position": (r0, "probe_struct", "base_struct"),
             "Haiku + probe on h(k) - h(k-1) vs Haiku": (diff0, "probe", "base")}  # fmt: skip
    for name, (r, a, b) in pairs.items():
        d, lo, hi, p = boot_diff(y, r[a], r[b], post)
        out.append(f"| {name} | {auc(y, r[a]):.3f} | {auc(y, r[b]):.3f} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | {p:.3f} |")
    out.append(f"\nAcross {SEEDS} fold seeds (nested, standard L2 probe on h): mean ± SD.\n")
    for name, (a, b) in {"probe alone (AUROC)": ("probe_alone", None), "increment over Haiku": ("probe", "base"),
                         "increment over Haiku + length/position": ("probe_struct", "base_struct")}.items():  # fmt: skip
        ds = [auc(y, res[s][a]) - (auc(y, res[s][b]) if b else 0) for s in res]
        out.append(f"- {name}: {np.mean(ds):+.3f} ± {np.std(ds, ddof=1):.3f} (seeds: {', '.join(f'{d:+.3f}' for d in ds)})")

    rk = lambda x: rankdata(x) / len(x)
    z = lambda x: (x - x.mean()) / x.std()
    s_probe = logit(r0["probe_alone"])
    s_struct = np.array([oof[i]["length_position"] for i in items])
    h = haiku[:, 0]
    out.append("\nZero-parameter combinations (out-of-fold scores; nothing fit here).\n")
    out.append("| combination | AUROC | vs | difference [95% CI] | one-sided p |\n|---|---|---|---|---|")
    for name, a, b in [
        ("mean rank(Haiku, probe)", (rk(h) + rk(s_probe)) / 2, rk(h)),
        ("mean z(Haiku, probe)", (z(h) + z(s_probe)) / 2, z(h)),
        ("mean rank(Haiku, length/position, probe)", (rk(h) + rk(s_struct) + rk(s_probe)) / 3, (rk(h) + rk(s_struct)) / 2),
        ("mean z(Haiku, length/position, probe)", (z(h) + z(s_struct) + z(s_probe)) / 3, (z(h) + z(s_struct)) / 2),
    ]:
        d, lo, hi, p = boot_diff(y, a, b, post)
        ref = "Haiku" if "length" not in name else "Haiku + length/position, same form"
        out.append(f"| {name} | {auc(y, a):.3f} | {ref} {auc(y, b):.3f} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | {p:.3f} |")
    out.append(
        "\nBy attribute and version (seed 0). Pooled fit: the probe and combiner as above (fit on both attributes), "
        "evaluated within the subset. Per-attribute fit: the probe and combiner fit on that attribute's cases only, nested the "
        "same way. Increments: out-of-fold AUROC difference, 95% CI resampling bloggers, one-sided p.\n"
    )
    out.append(
        "| subset | cases | Haiku | probe alone | + over Haiku [95% CI], p | + over Haiku + length/position [95% CI], p |\n"
        "|---|---|---|---|---|---|"
    )
    per = {a: nested(H[attr == a], y[attr == a], post[attr == a], haiku[attr == a], struct[attr == a], 0) for a in ("sex", "age")}
    for fit in ("pooled fit", "per-attribute fit"):
        for a in ("sex", "age"):
            for v in (None, "qwen_wrong", "qwen_right"):
                m = (attr == a) & ((version == v) if v else True)
                if fit == "pooled fit":
                    r, mm = r0, m
                else:
                    r, mm = per[a], m[attr == a]
                yy, pp = y[m], post[m]
                cells = []
                for hi_key, lo_key in (("probe", "base"), ("probe_struct", "base_struct")):
                    d, lo, hi, pv = boot_diff(yy, r[hi_key][mm], r[lo_key][mm], pp)
                    cells.append(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}], p {pv:.3f}")
                h_auc = auc(yy, haiku[m, 0])
                out.append(f"| {fit}: {a}, {v or 'both versions'} | {m.sum()} | {h_auc:.3f} | {auc(yy, r['probe_alone'][mm]):.3f} | "
                           f"{cells[0]} | {cells[1]} |")  # fmt: skip
    old = (run / "results.md").read_text()
    head = "\n## Standard L2 probe (no PCA) and increment checks"
    (run / "results.md").write_text((old.split(head)[0] if head in old else old) + "\n".join(out) + "\n")  # rerun-safe
    print("\n".join(out))
    s_h, s_d = r0["probe_alone"], diff0["probe_alone"]
    scores = [{"item": i, "probe_l2_h": float(s_h[n]), "probe_l2_diff": float(s_d[n])} for n, i in enumerate(items)]
    (run / "scores_l2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scores))  # seed 0 out-of-fold (slides)


def noq(run: Path, alt: Path):
    """Kylie, 2026-10-31: does the probe need the question asked first? The same cases, labels, sites (the sentence's last
    token), L48, standard probe and nested combiner, on `alt`'s activations (period_probe inputs-noq: the post alone as the
    user turn) next to `run`'s (question first). Same folds and seeds. Appends to alt/results.md."""
    A, index = activations(run)
    B, alt_index = activations(alt)
    li = json.loads(next((run / "hs").glob("part-*/layers.json")).read_text()).index(LAYER)
    lj = json.loads(next((alt / "hs").glob("part-*/layers.json")).read_text()).index(LAYER)
    by, alt_by = {r["id"]: r for r in index}, {r["id"]: r for r in alt_index}
    preds: dict[str, dict] = {}
    for r in load(PRED):
        if r.get("valid", True) and r.get("p") is not None:
            preds.setdefault(r["item"], {})[r["condition"]] = r["p"]
    rows = []
    for c in load(INPUTS / "cases.jsonl"):
        if "transcript_inline" not in preds.get(c["item"], {}):
            continue
        k = int(c["item"].rsplit("#", 1)[1])
        ix, jx = by[c["pair"]], alt_by[c["pair"]]
        n = sum(kd.startswith("s") for kd in ix["kinds"])
        length = len(c["comment"])
        rows.append((ix["rows"][ix["kinds"].index(f"s{k}")], jx["rows"][jx["kinds"].index(f"s{k}")], int(c["positive"]),
                     c["pair"].split(":")[0], preds[c["item"]]["transcript_inline"],
                     [np.log(length), length, k, n, k / n, k == n - 1, k == 0], c["attr"]))  # fmt: skip
    Hq = A[[r[0] for r in rows], li].astype(np.float64)
    Hn = B[[r[1] for r in rows], lj].astype(np.float64)
    y = np.array([r[2] for r in rows])
    post = np.array([r[3] for r in rows])
    haiku = logit(np.array([r[4] for r in rows]))[:, None]
    struct = np.array([r[5] for r in rows], float)
    attr = np.array([r[6] for r in rows])
    res = {
        (name, sd): nested(H, y, post, haiku, struct, sd) for name, H in (("question first", Hq), ("post alone", Hn)) for sd in range(SEEDS)
    }
    out = [
        "\n## Question first vs post alone (2026-10-31)\n",
        (
            f"{len(y)} cases; standard L2 probe at L{LAYER} on the sentence's last token, fully nested, the same folds. Seed 0 with "
            f"95% CIs resampling bloggers and one-sided p (H1: + probe better); mean ± SD over {SEEDS} fold seeds.\n"
        ),
        "| prompt | subset | probe alone | + over Haiku | + over Haiku + length/position | seeds: alone / + Haiku / + len/pos |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("question first", "post alone"):
        for sub in ("all", "sex", "age"):
            m = np.ones(len(y), bool) if sub == "all" else attr == sub
            r = res[name, 0]
            cells = []
            for a, b in (("probe", "base"), ("probe_struct", "base_struct")):
                d, lo, hi, pv = boot_diff(y[m], r[a][m], r[b][m], post[m])
                cells.append(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}], p {pv:.3f}")
            seeds = []
            for sd in range(SEEDS):
                q = res[name, sd]
                seeds.append([auc(y[m], q["probe_alone"][m]), auc(y[m], q["probe"][m]) - auc(y[m], q["base"][m]),
                              auc(y[m], q["probe_struct"][m]) - auc(y[m], q["base_struct"][m])])  # fmt: skip
            mu, sdv = np.mean(seeds, 0), np.std(seeds, 0, ddof=1)
            sd_cell = " / ".join(f"{a_:.3f} ± {b_:.3f}" for a_, b_ in zip(mu, sdv, strict=True))
            out.append(f"| {name} | {sub} | {auc(y[m], r['probe_alone'][m]):.3f} | {cells[0]} | {cells[1]} | {sd_cell} |")
    d, lo, hi, pv = boot_diff(y, res["question first", 0]["probe_alone"], res["post alone", 0]["probe_alone"], post)
    out.append(f"\nProbe alone, question first − post alone: {d:+.3f} [{lo:+.3f}, {hi:+.3f}], one-sided p {pv:.3f}.")
    with open(alt / "results.md", "a") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    if sys.argv[1] == "noq":  # period_check noq RUN ALT
        noq(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main(Path(sys.argv[1]))
