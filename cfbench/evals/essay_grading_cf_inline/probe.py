"""A linear probe on Qwen's essay grade (Kylie, 2026-10-31): ridge from the residual stream at the end of a grading prompt
to Qwen's mean score on it, 5 folds grouped by essay, then the out-of-fold predictions ranked as an essay_grading_cf_inline
investigator, two ways, AUROC against "removing the opener raises the score":
- `cf_auroc`: probe(base prompt) - probe(arm prompt), which needs a forward pass on the counterfactual (base) prompt;
- `zero_cf_auroc`: a second ridge from the ARM prompt's activations to the BASE score of the same essay (trained on arm
  prompts only), minus the observed arm mean: the investigator's own setting, no counterfactual forward pass.
The null "minus the observed arm mean" (mean reversion) is printed for reference: AUROC 0.776 on these cases.

    python scripts/hs_at_positions.py hs_inputs.jsonl OUT --layers 32 40 48 52 56 60    # on a GPU (scripts/sky/hs_positions.yaml)
    uv run python -m cfbench.evals.essay_grading_cf_inline.probe OUT

Inputs: hs_at_positions' <site>.npy over the inputs run's layout.jsonl ids (every arm x intro, 2,004 prompts), sites
`last` (the "\\n\\n" after </think>, right before the reply), `im_end`, `q` (the token before <|im_end|>). Ridge is solved
in its dual (features standardised on the training fold), alpha by leave-one-out on the training fold. Folds group by
essay, so a case's arm and base prompts are always held out together.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from cfbench.evals.essay_grading_cf_inline.eval import mean
from cfbench.evals.essay_grading_cf_inline.score import auroc

INPUTS = Path("data/runs/cfbench_evals/2026-10-30_essay-grading-inline-inputs")
ALPHAS = np.logspace(-1, 5, 13)
K, SEED = 5, 0


def ridge_oof(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    uniq = rng.permutation(np.unique(groups))
    fold = {g: i % K for i, g in enumerate(uniq)}
    f = np.array([fold[g] for g in groups])
    out = np.zeros(len(y))
    for k in range(K):
        tr, te = f != k, f == k
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
        a, b = (x[tr] - mu) / sd, (x[te] - mu) / sd
        ym = y[tr].mean()
        w, v = np.linalg.eigh(a @ a.T)
        yt = v.T @ (y[tr] - ym)
        best = (np.inf, ALPHAS[0])
        for alpha in ALPHAS:  # leave-one-out error of kernel ridge: r_i / (1 - H_ii)
            h = (v**2) @ (w / (w + alpha))
            r = (y[tr] - ym) - v @ (w / (w + alpha) * yt)
            loo = np.mean((r / (1 - h)) ** 2)
            if loo < best[0]:
                best = (loo, alpha)
        coef = v @ (yt / (w + best[1]))  # dual coefficients
        out[te] = ym + (b @ a.T) @ coef
    return out


def main(out: Path):
    lay = [json.loads(line)["id"] for line in (INPUTS / "layout.jsonl").open()]
    cases = {(c["arm"], c["item"]): c for c in map(json.loads, (INPUTS / "cases.jsonl").open())}
    ids = [i.split("/", 1)[1].split("~") for i in lay]
    y = np.array([mean(cases[a, i]["answers"]) for a, i in ids])
    groups = np.array([i for _, i in ids])
    layers = json.loads((out / "layers.json").read_text())
    pos = {(a, i): n for n, (a, i) in enumerate(ids)}
    cf = [(a, i) for a, i in ids if a != "base"]
    truth = np.array([mean(cases["base", i]["answers"]) > mean(cases[a, i]["answers"]) for a, i in cf])
    shift = np.array([mean(cases["base", i]["answers"]) - mean(cases[a, i]["answers"]) for a, i in cf])
    print(f"{len(y)} prompts, mean score {y.mean():.2f} (sd {y.std():.2f}); {len(cf)} cf cases, {truth.mean():.2f} go up")
    arm_rows = np.array([pos[k] for k in cf])
    ya = np.array([mean(cases[k]["answers"]) for k in cf])
    yb = np.array([mean(cases["base", i]["answers"]) for _, i in cf])
    print(f"null (minus the observed arm mean): AUROC {auroc(-ya, truth):.3f}")
    res = []
    for site in ("last", "im_end", "q"):
        acts = np.load(out / f"{site}.npy").astype(np.float32)
        for li, L in enumerate(layers):
            p = ridge_oof(acts[:, li], y, groups)
            d = np.array([p[pos["base", i]] - p[pos[a, i]] for a, i in cf])
            z = ridge_oof(acts[arm_rows, li], yb, groups[arm_rows]) - ya
            res.append((site, L, pearsonr(p, y)[0], auroc(d, truth), pearsonr(d, shift)[0], auroc(z, truth)))
            print(f"  {site:6} L{L:<3} OOF r(score) {res[-1][2]:.3f}   cf AUROC {res[-1][3]:.3f}   r(shift) {res[-1][4]:.3f}"
                  f"   zero-cf AUROC {res[-1][5]:.3f}", flush=True)  # fmt: skip
    best = max(res, key=lambda r: r[2])
    print(f"best by OOF score r (picked without looking at any AUROC): {best[0]} L{best[1]}: r {best[2]:.3f}, cf AUROC {best[3]:.3f}, "
          f"zero-cf AUROC {best[5]:.3f}")  # fmt: skip
    keys = ("site", "layer", "r_score", "cf_auroc", "r_shift", "zero_cf_auroc")
    doc = {"null_auroc": auroc(-ya, truth), "best": dict(zip(keys, best)), "rows": [dict(zip(keys, r)) for r in res]}
    (out / "probe.json").write_text(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
