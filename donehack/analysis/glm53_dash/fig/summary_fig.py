"""One-image summary of the GLM 5.3 runs (for Slack): Donation Bet gaps, is_prime hack rates, EvilGenie.

Needs analysis/glm53_dash/site/index.html (build.py), donehack/runs/eg_glm53, the monorepo's donation_bet_goodbad runs and
evalcue/envs/donation_bet/glm53_report.py (branch claude/donation-bet-named-pairs), plus matplotlib:
  cd donehack && uv run --with matplotlib python analysis/glm53_dash/fig/summary_fig.py
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[3] / "evalcue/envs/donation_bet"))  # a plain module: the package __init__ needs QWEN_VLLM_URL
import glm53_report as dx

INK, MUTE, GRID, A, B = "#1b2320", "#5d6b66", "#dfe4e1", "#1f7a68", "#c2410c"


def wil(k, n):
    return dx.wilson(k, n)


# Donation Bet, own lines
rows = [json.loads(l) for t in dx.SETS["own"] for l in (dx.RUNS / (dx.D + t) / "rows.jsonl").open()]
pos = {}
for r in rows:
    if r.get("positive") is not None:
        pos.setdefault(r["arm"], []).append(bool(r["positive"]))
pairs = [("good vs no donation", "nothing_good"), ("good vs bad cause", "good"), ("good vs beers", "beers_good")]
pairs += [(f"{dx.NAMES[g]} vs {dx.NAMES[o]}", f"named_{g}_{o}") for g, o in dx.N]
db = []
for lab, arm in pairs:
    b, a = pos[arm + "_below"], pos[arm + "_above"]
    lo, hi = dx.newcombe(sum(b), len(b), sum(a), len(a))
    db.append((lab, sum(b) / len(b) - sum(a) / len(a), lo, hi, len(b)))
nb = [x for _, arm in pairs[3:] for x in pos[arm + "_below"]]
na = [x for _, arm in pairs[3:] for x in pos[arm + "_above"]]
lo, hi = dx.newcombe(sum(nb), len(nb), sum(na), len(na))
db.append(("named, pooled (9 pairs)", sum(nb) / len(nb) - sum(na) / len(na), lo, hi, len(nb)))

# is_prime
D = json.loads(re.search(r"const D=(\{.*?\});\n", (HERE.parent / "site/index.html").read_text(), re.DOTALL).group(1))["conds"]
keys = [
    ("base", "base"),
    ("rules", "Rules section"),
    ("ask_user", "ask_user tool"),
    ("nt2", "2 tests"),
    ("rw_self_report", "self-report reward"),
    ("rw_fractional", "fractional reward"),
    ("writable", "writable tests"),
]
# EvilGenie
eg = [json.loads(p.read_text())["record"] for p in (HERE.parents[2] / "runs/eg_glm53").glob("*.json")]
eg = [r for r in eg if r.get("regraded")]

fig, axs = plt.subplots(1, 3, figsize=(16, 6.2), gridspec_kw={"width_ratios": [1.25, 1, 0.7]})
plt.rcParams.update({"font.family": "DejaVu Sans"})
for ax in axs:
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTE, length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)

ax = axs[0]
for i, (lab, d, lo, hi, n) in enumerate(db):
    y = len(db) - 1 - i
    sig = lo > 0
    ax.plot([100 * lo, 100 * hi], [y, y], color=B if sig else A, lw=2.5, alpha=0.45, solid_capstyle="round")
    ax.plot(100 * d, y, "o", color=B if sig else A, ms=7)
    ax.text(100 * hi + 1.5, y, f"{100 * d:+.0f}", va="center", fontsize=9, color=INK)
ax.axvline(0, color=MUTE, ls="--", lw=1)
ax.set_yticks(range(len(db)))
ax.set_yticklabels([x[0] for x in db][::-1], fontsize=9.5, color=INK)
ax.set_xlabel("gap in P(estimate ≤ line), pp  (good cause on ≤ side − on > side)", fontsize=9, color=MUTE)
ax.set_title(
    "Donation Bet: estimate pulled toward the good cause\nGLM's own lines; n=800 / 400 per arm; 95% CI; orange = CI excludes 0",
    fontsize=10.5,
    color=INK,
    loc="left",
)
ax.axhline(len(db) - 3.5, color=GRID, lw=1)
ax.axhline(0.5, color=GRID, lw=1)

ax = axs[1]
for i, (k, lab) in enumerate(keys):
    y = len(keys) - 1 - i
    for off, key, col, name in [(0.17, k, A, "with pytest line"), (-0.17, k + "__noit", MUTE, "without")]:
        c = D[key]
        p = c["k"] / c["n"]
        lo, hi = wil(c["k"], c["n"])
        ax.plot([100 * lo, 100 * hi], [y + off] * 2, color=col, lw=2.5, alpha=0.45, solid_capstyle="round")
        ax.plot(100 * p, y + off, "o", color=col, ms=6, label=name if i == 0 else None)
ax.axvline(100 * D["base"]["k"] / D["base"]["n"], color=A, ls="--", lw=1)
ax.set_yticks(range(len(keys)))
ax.set_yticklabels([l for _, l in keys][::-1], fontsize=9.5, color=INK)
ax.set_xlim(0, 100)
ax.set_xlabel("hack rate, % (passes the buggy (7, False) test)", fontsize=9, color=MUTE)
ax.set_title("is_prime: hack rate by variant\nn=64 per cell; 95% Wilson CI; dashed = base", fontsize=10.5, color=INK, loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, labelcolor=MUTE)

ax = axs[2]
n = len(eg)
hard = {"arc191_c", "arc195_c"}
bars = [
    ("solved\n(visible + holdout)", sum(r["public_all"] and r["holdout_all"] for r in eg)),
    ("passes visible,\nfails holdout", sum(r["public_all"] and not r["holdout_all"] for r in eg)),
    ("hardcodes sample\noutputs (read)", sum(r["task"] in hard and r["public_all"] for r in eg)),
    ("edited tests", sum(r["tests_modified"] for r in eg)),
]
for i, (lab, k) in enumerate(bars):
    y = len(bars) - 1 - i
    lo, hi = wil(k, n)
    ax.barh(y, 100 * k / n, color=B if "hardcode" in lab else A, alpha=0.8, height=0.55)
    ax.plot([100 * lo, 100 * hi], [y, y], color=INK, lw=1.2)
    ax.text(max(100 * hi, 100 * k / n) + 2, y, f"{k}/{n}", va="center", fontsize=9, color=INK)
ax.set_yticks(range(len(bars)))
ax.set_yticklabels([l for l, _ in bars][::-1], fontsize=9.5, color=INK)
ax.set_xlim(0, 100)
ax.set_xlabel("% of episodes", fontsize=9, color=MUTE)
ax.set_title(
    f"EvilGenie (lcb.yaml), {len({r['task'] for r in eg})} problems × 2\nholdout = LiveCodeBench private tests",
    fontsize=10.5,
    color=INK,
    loc="left",
)
fig.suptitle(
    "GLM 5.3 (z-ai/glm-5.3, reasoning medium) — agent-interp-envs coding + Donation Bet", fontsize=13, color=INK, x=0.01, ha="left"
)
fig.tight_layout(rect=(0, 0.02, 1, 0.95))
fig.savefig(HERE / "glm53_summary.png", dpi=150, facecolor="white")
print(HERE / "glm53_summary.png")
