"""The scaled-up blog-author eval (Kylie, 2026-10-31): batches 1-3, flat 15% rule, every condition plus the sentence-end
probe and the nested probe + transcript combiner, pooled AUROC unweighted and weighted to the natural Qwen right/wrong rates.

Qwen-right pairs went to deletions at a fixed sample per attribute per batch (all confident Qwen-wrong pairs did), so the
case mix over-represents Qwen-wrong. Weighted scores give each Qwen-right case weight 1 / (its batch and attribute's sampling
rate), Qwen-wrong 1, at evaluation only (the probe and combiner fit unweighted); ESS = (sum w)^2 / sum w^2. CIs and one-sided
p resample bloggers (2,000 draws), with the weights carried along.

    uv run --with scikit-learn --with scipy --with transformers --with jinja2 python -m scripts.author_probe.scaleup OUT PRED_RUN...
    ... scaleup OUT PRED_RUN... --cases data/runs/author_mcq_blog/2026-10-31_uniform-sample/cases.jsonl   # natural sample, w = 1
"""

import json
import sys
from pathlib import Path

import numpy as np

from cfbench.evals.author_blog_drop_inline.eval import ABLATES, BATCH3, CONDITIONS, collect
from scripts.author_probe.blog import BANK, sentences
from scripts.author_probe.period_check import LAYER, nested
from scripts.author_probe.period_probe import activations, load, logit

CAPTURES = [Path(f"data/runs/author_mcq_blog/2026-10-31_{s}") for s in ("period-probe", "blog-c-probe", "uniform-probe")]
# Qwen-right sampling rate per (batch, attribute): 300 (batches 1-2) or 521 (batch 3) of the confident Qwen-right pairs
RATE = {(1, "sex"): 0.257, (1, "age"): 0.222, (2, "sex"): 0.254, (2, "age"): 0.229, (3, "sex"): 0.260, (3, "age"): 0.230}
DIFFS = [("olens_inline", "transcript_inline"), ("olens_inline", "olens_placebo_inline"), ("nla_inline", "transcript_inline"),
         ("nla_inline", "nla_placebo_inline"), ("transcript+probe", "transcript_inline")]  # fmt: skip


def wauc(y, s, w) -> float:
    """Weighted AUROC: sum over (positive, negative) pairs of w_i w_j [s_i > s_j] (ties 1/2), normalised."""
    o = np.argsort(s, kind="stable")
    s, y, w = s[o], y[o], w[o]
    wn = np.where(y == 0, w, 0.0)
    _, start, cnt = np.unique(s, return_index=True, return_counts=True)
    below = np.concatenate([[0], np.cumsum(wn)])  # negative weight strictly below each tie group
    tie_neg = np.add.reduceat(wn, start)
    grp = np.repeat(np.arange(len(start)), cnt)
    credit = below[start][grp] + 0.5 * tie_neg[grp]
    pos = y == 1
    return float((w[pos] * credit[pos]).sum() / (w[pos].sum() * wn.sum()))


def main(out: Path, pred_runs: list[str], cases: Path | None = None, allow_pooled: bool = False):
    cs = load(cases) if cases else collect()
    batch = {r["id"]: k + 1 for k, d in enumerate(ABLATES) for r in load(d / "pairs.jsonl")} | {
        r["id"]: 3 for r in load(BATCH3.parent / "2026-10-30_ablate-loo-c/pairs.jsonl")
    }
    ps: dict[str, dict[str, float]] = {}
    source: dict[tuple, set] = {}  # (condition, item) -> {(run, harness version)} its prediction comes from
    for run in pred_runs:
        for r in load(Path(run) / "predictions.jsonl"):
            if r.get("valid", True) and r.get("p") is not None:
                ps.setdefault(r["condition"], {})[r["item"]] = r["p"]
                source.setdefault((r["condition"], r["item"]), set()).add((Path(run).name, r.get("harness_version", 1)))
    sites: dict[str, tuple[np.ndarray, dict]] = {}
    for cap in (c for c in CAPTURES if c.exists()):
        A, index = activations(cap)
        li = json.loads(next((cap / "hs").glob("part-*/layers.json")).read_text()).index(LAYER)
        for r in (r for r in index if r["split"] == "eval"):  # period-probe also holds its batch-3 training pool: never used here
            sites.setdefault(r["id"], (A[r["rows"], li], r))
    passages = {p["id"]: p for p in load(BANK / "passages.jsonl")}
    conds = [k for k in CONDITIONS if k in ps]  # the conditions the given runs have (e.g. NLA pending)
    keep = [c for c in cs if all(c["item"] in ps[k] for k in conds) and c["pair"] in sites]
    print(f"{len(keep)} of {len(cs)} cases have every condition and a capture")
    # the 2026-10-31 audit: pooling runs within a condition aliases run with label (positives mostly in one run), so each
    # condition's predictions must come from one run; different conditions may come from different runs
    runs = max(({x for c in keep for x in source[k, c["item"]]} for k in conds), key=len)
    if len(runs) > 1 and not allow_pooled:
        raise SystemExit(f"predictions from {len(runs)} (run, harness) pairs {sorted(runs)}: score one run, or --allow-pooled")
    H, struct = [], []
    for c in keep:
        h, ix = sites[c["pair"]]
        k = int(c["item"].rsplit("#", 1)[1])
        assert sentences(passages[c["pair"].split(":")[0]]["text"])[k] == c["comment"], c["item"]
        n = sum(kd.startswith("s") for kd in ix["kinds"])
        H.append(h[ix["kinds"].index(f"s{k}")])
        length = len(c["comment"])
        struct.append([np.log(length), length, k, n, k / n, k == n - 1, k == 0])
    X = np.array(H, np.float64)
    y = np.array([int(c["positive"]) for c in keep])
    post = np.array([c["pair"].split(":")[0] for c in keep])
    xs = {k: np.array([ps[k][c["item"]] for c in keep]) for k in conds}
    nest = nested(X, y, post, logit(xs["transcript_inline"])[:, None], np.array(struct, float), 0)
    xs |= {"probe": nest["probe_alone"], "transcript+probe": nest["probe"]}
    w = np.array([1.0 if cases or c["version"] == "qwen_wrong" else 1 / RATE[batch[c["pair"]], c["attr"]] for c in keep])
    one = np.ones(len(y))
    us = np.unique(post)
    of = {u: np.flatnonzero(post == u) for u in us}
    rng = np.random.default_rng(0)
    idx = [np.concatenate([of[us[j]] for j in rng.integers(0, len(us), len(us))]) for _ in range(2000)]
    right = np.array([c["version"] == "qwen_right" for c in keep])
    head = (f"{len(y)} cases ({y.sum()} positive), {len(np.unique([c['pair'] for c in keep]))} post-question pairs, {len(us)} bloggers; "
            f"weighted ESS {w.sum() ** 2 / (w**2).sum():.0f}, weighted Qwen-right share {w[right].sum() / w.sum():.3f} (unweighted "
            f"{right.mean():.3f})")  # fmt: skip
    lines = [head, ""]
    for name, ww in (("unweighted", one), ("weighted to natural Qwen right/wrong rates", w)):
        boot = {k: np.array([wauc(y[i], x[i], ww[i]) for i in idx]) for k, x in xs.items()}
        lines += [f"## {name}", "", "| condition | AUROC [95% CI] |", "|---|---|"]
        for k, x in xs.items():
            lines.append(f"| {k} | {wauc(y, x, ww):.3f} [{np.quantile(boot[k], 0.025):.3f}, {np.quantile(boot[k], 0.975):.3f}] |")
        lines += ["", "| A − B | Δ AUROC [95% CI] | one-sided p (Δ ≤ 0) |", "|---|---|---|"]
        for a, b in (d for d in DIFFS if set(d) <= set(xs)):
            d = boot[a] - boot[b]
            lines.append(f"| {a} − {b} | {wauc(y, xs[a], ww) - wauc(y, xs[b], ww):+.3f} [{np.quantile(d, 0.025):+.3f}, "
                         f"{np.quantile(d, 0.975):+.3f}] | {(d <= 0).mean():.4f} |")  # fmt: skip
        lines.append("")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.md").write_text("\n".join(lines) + "\n")
    rows = [{"item": c["item"], "version": c["version"], "batch": batch.get(c["pair"], 0), "positive": int(c["positive"]),
             "weight": float(wi), "probe_l2_h": float(xs["probe"][i]), "haiku_plus_probe": float(xs["transcript+probe"][i])}
            for i, (c, wi) in enumerate(zip(keep, w, strict=True))]  # fmt: skip
    (out / "scores_l2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("\n".join(lines))


if __name__ == "__main__":
    args = sys.argv[2:]
    cases = Path(args.pop(args.index("--cases") + 1)) if "--cases" in args else None
    main(Path(sys.argv[1]), [a for a in args if a not in ("--cases", "--allow-pooled")], cases, "--allow-pooled" in args)
