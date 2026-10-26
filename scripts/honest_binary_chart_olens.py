"""Chart for the honest-grader binary prediction runs with the O-lens conditions (docs/honest-grader-envs.md): 100 − RMSE of
the predicted rate for Env diff only / Transcript-only / With O-lens / O-lens placebo / Edit does nothing, over the cases that
have every condition, with a side panel of paired differences and 95% case bootstrap CIs, plus a per-scenario table on
stdout. scripts/honest_binary_chart.py is the same chart without the lens (all 242 cases); this one is restricted to the
cases the lens runs cover. Same metric and look as cfbench.report.charts (docs/cfbench-scoring.md).

    uv run --with matplotlib python scripts/honest_binary_chart_olens.py [--out docs/img/honest-grader/binary_rmse_olens.png]

Runs: data/runs/cfbench/2026-10-26_honest-ts*, honest-nt2-allpairs-* (env diff), honest-ol-* (olens, olens_placebo).
"""

import argparse
import collections
import glob
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID, NULL_FILL, BLUE, GRAY = "#0b0b0b", "#52514e", "#e1e0d9", "#ffffff", "#2a78d6", "#898781"
ORANGE, BLACK = "#e8831a", "#0b0b0b"  # CLAUDE.md chart colors: O-lens orange, placebo black
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True}
)
RUNS = ["2026-10-26_honest-ts-*", "2026-10-26_honest-ts2-*", "2026-10-26_honest-nt2-allpairs-*", "2026-10-26_honest-ol-*"]
CONDS = {"mse": "agentic", "mse_nt": "no_transcript", "mse_ol": "olens", "mse_pl": "olens_placebo"}


def load_cases() -> dict[tuple, dict]:
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for pat in RUNS:
        for d in glob.glob(f"data/runs/cfbench/{pat}"):
            if json.loads(Path(d, "meta.json").read_text())["status"] != "done":
                continue
            for line in Path(d, "predictions.jsonl").read_text().splitlines():
                r = json.loads(line)
                if r["valid"]:
                    scenario = r["env"].split(":")[-1].split("/")[0].replace("honest_", "")
                    rows[(scenario, r["env"].split("@")[1], r["base_arm"], r["cf_arm"])][r["condition"]].append(r)
    cases = {}
    for k, g in rows.items():
        if not all(g[c] for c in CONDS.values()):
            continue
        r = g["agentic"][0]
        q, qb = r["k_cf"] / r["n_cf"], r["k_base"] / r["n_base"]
        cases[k] = {fld: np.mean([(x["p"] - q) ** 2 for x in g[c]]) for fld, c in CONDS.items()} | {
            "null": (qb - q) ** 2,
            "noise": q * (1 - q) / r["n_cf"],
        }
    return cases


def rmse(cases, keys, fld) -> float:
    return float(np.sqrt(np.mean([cases[k][fld] for k in keys])))


def acc(v: float) -> float:
    return 100 - v * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/img/honest-grader/binary_rmse.png")
    args = ap.parse_args()
    cases = load_cases()
    keys = list(cases)
    rng = np.random.default_rng(0)
    flds = ("mse", "mse_nt", "mse_ol", "mse_pl", "null")
    pt = {fld: rmse(cases, keys, fld) for fld in (*flds, "noise")}
    boots = collections.defaultdict(list)
    for _ in range(5000):
        idx = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        b = {fld: rmse(cases, idx, fld) for fld in flds}
        for fld, v in b.items():
            boots[fld].append(v)
        boots["ts_minus_nt"].append(b["mse_nt"] - b["mse"])
        boots["ol_minus_ts"].append(b["mse"] - b["mse_ol"])
        boots["ol_minus_pl"].append(b["mse_pl"] - b["mse_ol"])
        boots["nt_minus_null"].append(b["null"] - b["mse_nt"])
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.25, 1]})
    series = [
        ("Env diff only", pt["mse_nt"], ci(boots["mse_nt"]), {"color": GRAY}),
        ("Transcript-only", pt["mse"], ci(boots["mse"]), {"color": BLUE}),
        ("With O-lens", pt["mse_ol"], ci(boots["mse_ol"]), {"color": ORANGE}),
        ("O-lens placebo", pt["mse_pl"], ci(boots["mse_pl"]), {"color": BLACK}),
        ("Edit does nothing", pt["null"], ci(boots["null"]), {"color": NULL_FILL, "edgecolor": INK, "hatch": "////", "lw": 0.8}),
    ]
    for j, (lab, v, c, kw) in enumerate(series):
        ax.bar(j, acc(v), 0.7, label=lab, zorder=2, **kw)
        ax.plot([j, j], [acc(c[1]), acc(c[0])], color=INK, lw=1, zorder=3)
        ax.text(j, acc(c[0]) + 0.8, f"{acc(v):.0f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(acc(pt["noise"]), ls=":", color=MUTED, lw=1.2, label="ceiling (sampling noise)", zorder=3)
    ax.set_xticks([])
    ax.set_xlim(-0.7, 4.7)
    ax.set_ylim(0, 112)
    ax.set_ylabel("100 − RMSE of predicted rate (pp)")
    title = (
        f"Honest-grader scenarios: predicted rate of Score ≤ cutoff (Gemini 3.8 Flash)\n"
        f"the two honest_peer pairs, cases with every condition, n={len(keys)}"
    )
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(loc="upper left", ncol=3, fontsize=8, frameon=False)
    diffs = [
        ("With O-lens − Transcript-only", (pt["mse"] - pt["mse_ol"]) * 100, ci(boots["ol_minus_ts"]), ORANGE),
        ("With O-lens − O-lens placebo", (pt["mse_pl"] - pt["mse_ol"]) * 100, ci(boots["ol_minus_pl"]), ORANGE),
        ("Transcript-only − Env diff only", (pt["mse_nt"] - pt["mse"]) * 100, ci(boots["ts_minus_nt"]), BLUE),
        ("Env diff only − Edit does nothing", (pt["null"] - pt["mse_nt"]) * 100, ci(boots["nt_minus_null"]), GRAY),
    ]
    for y, (lab, d, c, col) in enumerate(reversed(diffs)):
        lo, hi = c[0] * 100, c[1] * 100
        bx.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
        bx.plot(d, y, "o", ms=7, color=col, mec="white", mew=1.5, zorder=3)
        bx.text(hi + 0.6, y, f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]", va="center", fontsize=8, color=INK)
        bx.text(lo - 0.6, y, lab, va="center", ha="right", fontsize=8, color=INK)
    bx.axvline(0, color=MUTED, lw=1, zorder=1)
    bx.set_yticks([])
    bx.set_ylim(-0.8, 3.8)
    lo_all, hi_all = min(c[0] for _, _, c, _ in diffs) * 100, max(c[1] for _, _, c, _ in diffs) * 100
    bx.set_xlim(lo_all - 28, hi_all + 16)
    bx.set_xlabel("RMSE reduction (pp, higher is better)")
    bx.set_title("Differences, 95% case bootstrap (paired, same cases)", loc="left", fontsize=10)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    names = [("envdiff", "mse_nt"), ("TS", "mse"), ("OL", "mse_ol"), ("placebo", "mse_pl"), ("null", "null"), ("ceiling", "noise")]
    bars = " ".join(f"{lab} {acc(pt[fld]):.1f}" for lab, fld in names)
    print(f"wrote {args.out}: n={len(keys)} {bars}")
    for lab, d, c, _ in diffs:
        print(f"  {lab}: {d:+.1f} [{c[0] * 100:+.1f}, {c[1] * 100:+.1f}]")

    print("\nper scenario (weak item only; 100 − RMSE in pp, and the OL − TS difference with its 95% case bootstrap)")
    print(f"{'scenario':13} {'n':>3} {'envdiff':>8} {'TS':>6} {'OL':>6} {'placebo':>8} {'null':>6} {'ceiling':>8}   OL − TS")
    by = collections.defaultdict(list)
    for k in keys:
        if "weak" in k[1]:
            by[k[0]].append(k)
    rows = []
    for sc, ks in by.items():
        d = []
        for _ in range(2000):
            idx = [ks[i] for i in rng.integers(0, len(ks), len(ks))]
            d.append(rmse(cases, idx, "mse") - rmse(cases, idx, "mse_ol"))
        accs = [acc(rmse(cases, ks, fld)) for fld in ("mse_nt", "mse", "mse_ol", "mse_pl", "null", "noise")]
        rows.append((sc, len(ks), *accs, (rmse(cases, ks, "mse") - rmse(cases, ks, "mse_ol")) * 100, ci(d)))
    for sc, n, nt, ts, ol, pl, null, ceil, d, c in sorted(rows, key=lambda r: -r[8]):
        lo, hi = c[0] * 100, c[1] * 100
        print(f"{sc:13} {n:>3} {nt:8.1f} {ts:6.1f} {ol:6.1f} {pl:8.1f} {null:6.1f} {ceil:8.1f}   {d:+5.1f} [{lo:+.1f}, {hi:+.1f}]")


if __name__ == "__main__":
    main()
