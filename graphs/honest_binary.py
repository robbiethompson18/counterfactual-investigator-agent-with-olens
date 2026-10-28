"""Chart for the honest-grader binary prediction runs (docs/honest-grader-envs.md): 100 − RMSE of the predicted rate for
Env diff only / Transcript-only / Edit does nothing over every case, with a side panel of paired differences and 95% case
bootstrap CIs, plus a per-scenario table on stdout. Same metric and look as graphs.cfbench_report (docs/cfbench-scoring.md).

    uv run python -m graphs.honest_binary [--out docs/img/honest-grader/binary_rmse.png]

Cases are (thing judged, base framing, counterfactual framing); a case needs an `agentic` and a `no_transcript` prediction.
Runs: data/runs/cfbench/2026-10-26_honest-ts*, honest-ts-allpairs-*, honest-nt2-allpairs-* (the killed honest-nt-allpairs-*
are abandoned and skipped by status).
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
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True}
)
RUNS = ["2026-10-26_honest-ts-*", "2026-10-26_honest-ts2-*", "2026-10-26_honest-nt2-allpairs-*"]


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
        if not (g["agentic"] and g["no_transcript"]):
            continue
        r = g["agentic"][0]
        q, qb = r["k_cf"] / r["n_cf"], r["k_base"] / r["n_base"]
        cases[k] = {
            "mse": np.mean([(x["p"] - q) ** 2 for x in g["agentic"]]),
            "mse_nt": np.mean([(x["p"] - q) ** 2 for x in g["no_transcript"]]),
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
    pt = {fld: rmse(cases, keys, fld) for fld in ("mse", "mse_nt", "null", "noise")}
    boots = collections.defaultdict(list)
    for _ in range(5000):
        idx = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        b = {fld: rmse(cases, idx, fld) for fld in ("mse", "mse_nt", "null")}
        for fld, v in b.items():
            boots[fld].append(v)
        boots["ts_minus_nt"].append(b["mse_nt"] - b["mse"])
        boots["nt_minus_null"].append(b["null"] - b["mse_nt"])
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 5), gridspec_kw={"width_ratios": [1.15, 1]})
    series = [
        ("Env diff only", pt["mse_nt"], ci(boots["mse_nt"]), {"color": GRAY}),
        ("Transcript-only", pt["mse"], ci(boots["mse"]), {"color": BLUE}),
        ("Edit does nothing", pt["null"], ci(boots["null"]), {"color": NULL_FILL, "edgecolor": INK, "hatch": "////", "lw": 0.8}),
    ]
    for j, (lab, v, c, kw) in enumerate(series):
        ax.bar(j, acc(v), 0.7, label=lab, zorder=2, **kw)
        ax.plot([j, j], [acc(c[1]), acc(c[0])], color=INK, lw=1, zorder=3)
        ax.text(j, acc(c[0]) + 0.8, f"{acc(v):.0f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(acc(pt["noise"]), ls=":", color=MUTED, lw=1.2, label="ceiling (sampling noise)", zorder=3)
    ax.set_xticks([])
    ax.set_xlim(-0.7, 2.7)
    ax.set_ylim(0, 112)
    ax.set_ylabel("100 − RMSE of predicted rate (pp)")
    title = f"Honest-grader scenarios: predicted rate of Score ≤ cutoff (Gemini 3.8 Flash)\nall cases, n={len(keys)}"
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)
    diffs = [
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
    bx.set_ylim(-0.8, 1.8)
    lo_all, hi_all = min(c[0] for _, _, c, _ in diffs) * 100, max(c[1] for _, _, c, _ in diffs) * 100
    bx.set_xlim(lo_all - 28, hi_all + 16)
    bx.set_xlabel("RMSE reduction (pp, higher is better)")
    bx.set_title("Differences, 95% case bootstrap (paired, same cases)", loc="left", fontsize=10)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    names = [("envdiff", "mse_nt"), ("TS", "mse"), ("null", "null"), ("ceiling", "noise")]
    bars = " ".join(f"{lab} {acc(pt[fld]):.1f}" for lab, fld in names)
    print(f"wrote {args.out}: n={len(keys)} {bars}")
    for lab, d, c, _ in diffs:
        print(f"  {lab}: {d:+.1f} [{c[0] * 100:+.1f}, {c[1] * 100:+.1f}]")

    print("\nper scenario (weak item only; 100 − RMSE in pp, and the TS − env diff difference with its 95% case bootstrap)")
    print(f"{'scenario':13} {'n':>3} {'envdiff':>8} {'TS':>6} {'null':>6} {'ceiling':>8}   TS − envdiff")
    by = collections.defaultdict(list)
    for k in keys:
        if "weak" in k[1]:
            by[k[0]].append(k)
    rows: list[tuple] = []
    for sc, ks in by.items():
        d = []
        for _ in range(2000):
            idx = [ks[i] for i in rng.integers(0, len(ks), len(ks))]
            d.append(rmse(cases, idx, "mse_nt") - rmse(cases, idx, "mse"))
        accs = [acc(rmse(cases, ks, fld)) for fld in ("mse_nt", "mse", "null", "noise")]
        rows.append((sc, len(ks), *accs, (rmse(cases, ks, "mse_nt") - rmse(cases, ks, "mse")) * 100, ci(d)))
    for sc, n, nt, ts, null, ceil, d, c in sorted(rows, key=lambda r: -r[6]):
        print(f"{sc:13} {n:>3} {nt:8.1f} {ts:6.1f} {null:6.1f} {ceil:8.1f}   {d:+5.1f} [{c[0] * 100:+.1f}, {c[1] * 100:+.1f}]")


if __name__ == "__main__":
    main()
