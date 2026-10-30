"""Scores of a drop-eval run (author_mcq_drop_inline / author_blog_drop_inline): per version, condition and attribute,
AUROC, RMSE (root Brier) and accuracy at p >= 0.5 against the 0/1 truth; paired O-lens - transcript and O-lens - placebo
(Brier gain = Brier(other) - Brier(O-lens), AUROC difference) with 95% bootstrap CIs resampling AUTHORS (an author's
passages, pairs and comments move together).

    uv run --with scikit-learn python -m scripts.author_probe.score_drop RUN   # RUN/scores.md
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

CONDS = ("transcript_inline", "olens_inline", "olens_placebo_inline")
SHORT = {"transcript_inline": "transcript", "olens_inline": "O-lens", "olens_placebo_inline": "placebo"}


def auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if 0 < sum(y) < len(y) else float("nan")


def main(run: Path):
    rows = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
    by: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        if r.get("p") is not None:
            author = r["pair"].split(":")[0].split("_")[0]
            by[r["version"], r["item"]][r["condition"]] = (r["p"], r["k_cf"], r["attr"], author)
    cost = sum(r.get("cost") or 0 for r in rows)
    lines = [f"{run.name}: {len(rows)} predictions, {sum(r.get('p') is None for r in rows)} invalid, ${cost:.2f}", ""]
    lines += [
        "| version | attribute | n | " + " | ".join(f"{SHORT[c]} AUROC / RMSE / acc" for c in CONDS) + " |",
        "|---|---|---|---|---|---|",
    ]
    rng = np.random.default_rng(0)
    diffs = []
    for v in sorted({k[0] for k in by}):
        keys = [k for k in by if k[0] == v and all(c in by[k] for c in CONDS)]
        for attr in [None, *sorted({by[k][CONDS[0]][2] for k in keys})]:
            ks = [k for k in keys if attr is None or by[k][CONDS[0]][2] == attr]
            y = [by[k][CONDS[0]][1] for k in ks]
            cells = []
            for c in CONDS:
                p = np.array([by[k][c][0] for k in ks])
                cells.append(f"{auc(y, p):.2f} / {np.sqrt(np.mean((p - np.array(y)) ** 2)):.3f} / {np.mean((p >= 0.5) == np.array(y)):.2f}")
            lines.append(f"| {v} | {attr or 'all'} | {len(ks)} | " + " | ".join(cells) + " |")
        y = np.array([by[k][CONDS[0]][1] for k in keys])
        P = {c: np.array([by[k][c][0] for k in keys]) for c in CONDS}
        authors = np.array([by[k][CONDS[0]][3] for k in keys])
        groups = {a: np.flatnonzero(authors == a) for a in np.unique(authors)}
        names = list(groups)
        for a, b in (("olens_inline", "transcript_inline"), ("olens_inline", "olens_placebo_inline")):
            gain = (P[b] - y) ** 2 - (P[a] - y) ** 2
            bs_g, bs_a = [], []
            for _ in range(2000):
                idx = np.concatenate([groups[names[i]] for i in rng.integers(0, len(names), len(names))])
                bs_g.append(gain[idx].mean())
                if 0 < y[idx].sum() < len(idx):
                    bs_a.append(auc(y[idx], P[a][idx]) - auc(y[idx], P[b][idx]))
            q = lambda xs: f"[{np.percentile(xs, 2.5):+.3f}, {np.percentile(xs, 97.5):+.3f}]"
            diffs.append(
                f"- {v}, {SHORT[a]} − {SHORT[b]}: Brier gain {gain.mean():+.4f} {q(bs_g)}, AUROC {auc(y, P[a]) - auc(y, P[b]):+.3f} "
                f"{q(bs_a)} (n = {len(y)} cases, {len(names)} authors)"
            )
    lines += ["", "Paired, 95% bootstrap CIs over authors:", "", *diffs]
    (run / "scores.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
