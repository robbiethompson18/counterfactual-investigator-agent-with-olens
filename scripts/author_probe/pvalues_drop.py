"""Pooled (qwen_wrong + qwen_right) paired comparisons of the O-lens with the transcript and with the placebo in drop-eval
runs, 95% CIs and one-sided p for H1 "O-lens better" from a bootstrap over authors (10k):

- raw Brier gain, plus a paired sign-flip permutation p at the author level (each author's summed gain flipped; 10k);
- AUROC difference (base-rate free), pooled and per version;
- Brier gain after Platt recalibration of each condition (logistic on logit p, 5-fold CV grouped by author), so a
  condition's miscalibration (Haiku's p's sit low) is removed.

Appended to RUN/scores.md.

    uv run python -m scripts.author_probe.pvalues_drop RUN [RUN ...]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def _platt(p: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    x = np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))[:, None]
    out = np.zeros(len(p))
    for tr, te in GroupKFold(n_splits=5).split(x, groups=groups):
        out[te] = LogisticRegression().fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    return out


def auroc_and_recal(keys, by, y, inv, n_auth, rng) -> list[str]:
    conds = ("transcript_inline", "olens_inline", "olens_placebo_inline")
    P = {c: np.array([by[k][c][0] for k in keys]) for c in conds}
    R = {c: _platt(P[c], y, inv) for c in conds}
    version = np.array([k.split("/")[0] for k in keys])
    groups = [np.flatnonzero(inv == a) for a in range(n_auth)]
    lines = ["", "AUROC difference (bootstrap over authors, 10k) and Brier gain after Platt recalibration (5-fold CV by author):", ""]
    for other in ("transcript_inline", "olens_placebo_inline"):
        name = "transcript" if other == "transcript_inline" else "placebo"
        for v in (None, *sorted(set(version))):
            m = np.ones(len(y), bool) if v is None else version == v
            idx_all = np.flatnonzero(m)
            gs = [g[m[g]] for g in groups if m[g].any()]
            obs = _auc(y[m], P["olens_inline"][m]) - _auc(y[m], P[other][m])
            gain = (R[other] - y) ** 2 - (R["olens_inline"] - y) ** 2
            boots_a, boots_g = [], []
            for _ in range(10000 if v is None else 2000):
                idx = np.concatenate([gs[i] for i in rng.integers(0, len(gs), len(gs))])
                if 0 < y[idx].sum() < len(idx):
                    boots_a.append(_auc(y[idx], P["olens_inline"][idx]) - _auc(y[idx], P[other][idx]))
                boots_g.append(gain[idx].mean())
            ba, bg = np.array(boots_a), np.array(boots_g)
            ci = lambda b: f"[{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}]"
            lines.append(
                f"- {v or 'pooled'}, O-lens − {name}: AUROC {obs:+.4f} {ci(ba)} one-sided p {np.mean(ba <= 0):.4f}; "
                f"recalibrated Brier gain {gain[idx_all].mean():+.4f} {ci(bg)} one-sided p {np.mean(bg <= 0):.4f}"
            )
    return lines


def main(run: Path) -> list[str]:
    by: dict[str, dict] = defaultdict(dict)
    for r in map(json.loads, (run / "predictions.jsonl").read_text().splitlines()):
        if r.get("valid", True) and r.get("p") is not None:  # the same filter as score.py
            by[f"{r['version']}/{r['item']}"][r["condition"]] = (r["p"], r["k_cf"], r["pair"].split(":")[0].split("_")[0])
    keys = [k for k, d in by.items() if len(d) == 3]
    y = np.array([by[k]["transcript_inline"][1] for k in keys])
    authors = np.array([by[k]["transcript_inline"][2] for k in keys])
    P = {c: np.array([by[k][c][0] for k in keys]) for c in ("transcript_inline", "olens_inline", "olens_placebo_inline")}
    uniq, inv = np.unique(authors, return_inverse=True)
    rng = np.random.default_rng(0)
    lines = [f"Pooled qwen_wrong + qwen_right: {len(keys)} cases, {len(uniq)} authors (bootstrap and sign-flip over authors, 10k)", ""]
    for other in ("transcript_inline", "olens_placebo_inline"):
        gain = (P[other] - y) ** 2 - (P["olens_inline"] - y) ** 2  # > 0: O-lens has the lower Brier
        per_author = np.bincount(inv, weights=gain, minlength=len(uniq))
        n_author = np.bincount(inv, minlength=len(uniq))
        boots = []
        for _ in range(10000):
            pick = rng.integers(0, len(uniq), len(uniq))
            boots.append(per_author[pick].sum() / n_author[pick].sum())
        boots = np.array(boots)
        obs = gain.mean()
        flips = rng.choice([-1.0, 1.0], size=(10000, len(uniq)))
        null = (flips * per_author).sum(1) / len(keys)
        p_perm = (np.sum(null >= obs) + 1) / (len(null) + 1)
        p_boot = np.mean(boots <= 0)
        name = "transcript" if other == "transcript_inline" else "placebo"
        lines.append(
            f"- O-lens − {name}: Brier gain {obs:+.4f}, 95% CI [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}], "
            f"one-sided p (bootstrap) {p_boot:.4f}, sign-flip p {p_perm:.4f}"
        )
    lines += auroc_and_recal(keys, by, y, inv, len(uniq), rng)
    with (run / "scores.md").open("a") as f:
        f.write("\n" + "\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    for x in sys.argv[1:]:
        print(Path(x).name, *main(Path(x)), sep="\n")
