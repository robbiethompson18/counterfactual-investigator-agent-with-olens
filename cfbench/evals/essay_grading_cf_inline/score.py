"""Score essay_grading_cf_inline runs: P(removing the opener raises Qwen's score).

    uv run python -m cfbench.evals.essay_grading_cf_inline.score data/runs/cfbench/<date>_<slug> [...]

Per condition: % right (guess = p >= 0.5, per case the mean over draws), AUROC of the mean p (up vs not), Brier, each with
a 95% CI resampling essays (truths are per essay); the same on the clear cases (|shift| >= 0.5 points); then every paired
difference against olens_inline over the same cases.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from cfbench.core import canon_condition

BOOT = 1000
PROBE = Path("data/runs/essay_probe/2026-10-31_essay-grading-acts/probe.json")  # probe.py's output
PAIRS = (("olens_inline", "olens_placebo_inline"), ("olens_inline", "transcript_inline"), ("olens_placebo_inline", "transcript_inline"))


def auroc(p: np.ndarray, y: np.ndarray) -> float:
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def metrics(ps: dict, truth: dict, keys: list) -> dict[str, float]:
    p, y = np.array([ps[k] for k in keys]), np.array([truth[k] for k in keys])
    return {"right": float(((p >= 0.5) == y).mean()), "brier": float(((p - y) ** 2).mean()),
            "auroc": auroc(p, y)}  # fmt: skip


def boot(f, keys: list, rng) -> list[dict]:
    """f(keys) over BOOT resamples of essays (all of an essay's cases together)."""
    by = defaultdict(list)
    for k in keys:
        by[k[1]].append(k)
    items = list(by)
    return [f([k for i in rng.integers(len(items), size=len(items)) for k in by[items[i]]]) for _ in range(BOOT)]


def main(paths: list[str]):
    rows = [json.loads(line) for p in paths for line in (Path(p) / "predictions.jsonl").open()]
    ps, truth, shift = defaultdict(lambda: defaultdict(list)), {}, {}
    for r in rows:
        if r.get("valid", True) and r.get("p") is not None:
            k = (r["base_arm"], r["item"])
            ps[canon_condition(r["condition"])][k].append(r["p"])
            truth[k], shift[k] = r["k_cf"], r["shift"]
    mp = {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in ps.items()}
    rng = np.random.default_rng(0)
    for name, keep in (("all", lambda k: True), ("clear |shift|>=0.5", lambda k: abs(shift[k]) >= 0.5)):
        print(f"== {name}")
        for c, d in sorted(mp.items()):
            keys = [k for k in d if keep(k)]
            m = metrics(d, truth, keys)
            bs = boot(lambda ks, d=d: metrics(d, truth, ks), keys, rng)
            ci = {s: np.nanpercentile([b[s] for b in bs], [2.5, 97.5]) for s in m}
            up = np.mean([truth[k] for k in keys])
            print(f"  {c:22} n={len(keys)} up={up:.2f}  " + "  ".join(f"{s} {m[s]:.3f} [{ci[s][0]:.3f}, {ci[s][1]:.3f}]" for s in m))
        for a, b in PAIRS:
            if a in mp and b in mp:
                keys = [k for k in mp[a].keys() & mp[b].keys() if keep(k)]
                diff = lambda ks, a=a, b=b: {
                    s: metrics(mp[a], truth, ks)[s] - metrics(mp[b], truth, ks)[s] for s in ("right", "auroc", "brier")
                }
                m, bs = diff(keys), boot(diff, keys, rng)
                print(f"  {a} - {b}: " + "  ".join(f"{s} {m[s]:+.3f} [{np.nanpercentile([x[s] for x in bs], 2.5):+.3f}, "
                                                   f"{np.nanpercentile([x[s] for x in bs], 97.5):+.3f}]" for s in m))  # fmt: skip


if __name__ == "__main__":
    main(sys.argv[1:])


def stats(rows: list[dict]) -> list[dict]:
    """The env viewer's claim table (environments/_lib/evals.py), hint_qwen_dsv4's % right schema: one row per condition,
    a case = (arm, intro), its score the mean of its draws; CI resampling essays."""
    invalid: dict[str, int] = defaultdict(int)
    by: dict = defaultdict(lambda: defaultdict(list))
    truth = {}
    for r in rows:
        cond = canon_condition(r["condition"])
        if not r["valid"]:
            invalid[cond] += 1
            continue
        k = (r["base_arm"], r["item"])
        by[cond][k].append(r["p"])
        truth[k] = r["k_cf"]
    rng, out = np.random.default_rng(0), []
    for cond, d in sorted(by.items()):
        keys = list(d)
        right = lambda ks, d=d: float(np.mean([np.mean([(p >= 0.5) == truth[k] for p in d[k]]) for k in ks]))
        bs = np.array(boot(right, keys, rng), dtype=float)
        mp = {k: float(np.mean(v)) for k, v in d.items()}
        m = metrics(mp, truth, keys)
        out.append({"condition": cond, "version": "cf", "invalid": invalid[cond], "n": sum(map(len, d.values())), "n_items": len(keys),
                    "accuracy": right(keys), "accuracy_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    "auroc": m["auroc"], "mean_brier": float(np.mean([(p - truth[k]) ** 2 for k in keys for p in d[k]]))})  # fmt: skip
    return out


def PAGE(rows: list[dict]) -> str:
    """The eval's page section, AUROC first (Kylie): per condition AUROC with a 95% CI (essays resampled), on all cases and
    on the clear ones; paired AUROC differences with CI and one-sided p; % right and Brier for reference; AUROC by opener
    group (environments/essay_grading/wide.py)."""
    from cfbench.evals.page import bars, swatch, table
    from environments.essay_grading.wide import GROUPS

    group = {f"w-{s}": g for g, d in GROUPS.items() for s in d}
    ps, truth, shift = defaultdict(lambda: defaultdict(list)), {}, {}
    for r in rows:
        if r.get("valid", True) and r.get("p") is not None:
            k = (r["base_arm"], r["item"])
            ps[canon_condition(r["condition"])][k].append(r["p"])
            truth[k], shift[k] = r["k_cf"], r["shift"]
    mp = {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in ps.items()}
    order = [c for c in ("transcript_inline", "olens_inline", "olens_placebo_inline") if c in mp]
    rng = np.random.default_rng(0)
    f3 = lambda x: f"{x:.3f}"
    pfmt = lambda p: "< 0.001" if p < 0.001 else f"= {p:.3f}"
    lede = (
        "<p class=\"lede\">Per case (an opener x an intro) the mean of its draws' p. Truth = Qwen's mean score with the "
        "plain opener is higher than with this one. 95% CIs resample essays (an essay's truth is shared by its ~9 arms).</p>"
    )
    out = ["<h2>AUROC: does the investigator rank the cases that go up above the ones that don't?</h2>", lede]
    for name, keep in (("all cases", lambda k: True),):  # the clear-cases cut is in main() only (Kylie: drop it from the page)
        res, pair_rows = [], []
        for c in order:
            keys = [k for k in mp[c] if keep(k)]
            a = metrics(mp[c], truth, keys)["auroc"]
            bs = [b["auroc"] for b in boot(lambda ks, c=c: metrics(mp[c], truth, ks), keys, rng)]
            res.append((c, a, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), len(keys), metrics(mp[c], truth, keys)))
        out.append(bars([(c, a, lo, hi) for c, a, lo, hi, *_ in res], 0.5, 1.0, f"AUROC, {name} (n = {res[0][4]}); axis from 0.5 (chance)"))
        for a, b in PAIRS:
            if a in mp and b in mp:
                keys = [k for k in mp[a].keys() & mp[b].keys() if keep(k)]
                d = lambda ks, a=a, b=b: {
                    s: metrics(mp[a], truth, ks)[s] - metrics(mp[b], truth, ks)[s] for s in ("auroc", "right", "brier")
                }
                m, bs = d(keys), boot(d, keys, rng)
                cell = lambda s, better, m=m, bs=bs: (f"<b>{m[s]:+.3f}</b> [{np.nanpercentile([x[s] for x in bs], 2.5):+.3f}, "
                                          f"{np.nanpercentile([x[s] for x in bs], 97.5):+.3f}] "
                                          f"<span class=\"mute\">p {pfmt(np.mean([better(x[s]) for x in bs]))}</span>")  # fmt: skip
                pair_rows.append([f"{swatch(a)} − {swatch(b)}", cell("auroc", lambda x: x <= 0), cell("right", lambda x: x <= 0),
                                  cell("brier", lambda x: x >= 0)])  # fmt: skip
        out.append(table(["condition", "cases", "<b>AUROC</b> [95% CI]", "% right (p ≥ 0.5)", "Brier"],
                         [[swatch(c), str(n), f"<b>{f3(a)}</b> [{f3(lo)}, {f3(hi)}]", f"{100 * m['right']:.1f}", f3(m["brier"])]
                          for c, a, lo, hi, n, m in res]))  # fmt: skip
        # baselines that need no investigator: mean reversion (minus the observed arm mean), and the linear probe if it was run
        obs = {(r["base_arm"], r["item"]): sum(float(x) * n for x, n in r["answers_base"].items()) / sum(r["answers_base"].values())
               for r in rows}  # fmt: skip
        keys = [k for k in truth if keep(k)]
        nul = {k: -obs[k] for k in keys}
        bs = [b["auroc"] for b in boot(lambda ks, nul=nul: metrics(nul, truth, ks), keys, rng)]
        ci = f"[{f3(np.nanpercentile(bs, 2.5))}, {f3(np.nanpercentile(bs, 97.5))}]"
        base_rows = [["null: minus the observed arm mean (mean reversion)", f"<b>{f3(metrics(nul, truth, keys)['auroc'])}</b> {ci}"]]
        if PROBE.exists():
            b = json.loads(PROBE.read_text())["best"]
            base_rows.append(
                [f"linear probe, zero-counterfactual ({b['site']} token, L{b['layer']}; probe.py)", f"<b>{f3(b['zero_cf_auroc'])}</b>"]
            )
            base_rows.append(
                [f"linear probe with a forward pass on the edited prompt ({b['site']} token, L{b['layer']})", f3(b["cf_auroc"])]
            )
        out.append(table(["baseline (no investigator)", "AUROC"], base_rows))
        out.append(
            '<p class="lede">Paired differences (the same essay resamples for both conditions); '
            "p is one-sided: the first condition is better.</p>"
        )
        out.append(table(["difference", "AUROC", "% right", "Brier (lower is better)"], pair_rows))
    out.append("<h3>AUROC by opener group</h3>")
    body = []
    for g in sorted({group.get(a, "old arms") for a, _ in truth}):
        keys = [k for k in truth if group.get(k[0], "old arms") == g]
        a = {c: metrics(mp[c], truth, [k for k in keys if k in mp[c]])["auroc"] for c in order}
        body.append(
            [g, str(len(keys)), *(f3(a[c]) for c in order), f"{a.get('olens_inline', np.nan) - a.get('olens_placebo_inline', np.nan):+.3f}"]
        )
    out.append(table(["group", "cases", *(swatch(c) for c in order), "O-lens − placebo"], body))
    return "\n".join(out)
