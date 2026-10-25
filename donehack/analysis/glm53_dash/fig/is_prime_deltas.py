"""is_prime hack rates: GLM 5.3 prompt variants (runs/glm53 = WITH the "Iterate ... pytest" line, runs/glm53_noit = WITHOUT),
plus V4 Flash and Qwen3.6-27B episode-ending conditions (bwrap runs ns_* = with the line, noit_* = without).
Wilson 95% CIs; Fisher exact vs base in the same line arm, Holm-corrected within each panel/arm.
  uv run --with matplotlib python analysis/glm53_dash/fig/is_prime_deltas.py"""

import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
WITH, WITHOUT = "#1f6f8b", "#e07a2f"  # teal = with pytest line, orange = without


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i])
    out = [1.0] * len(ps)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


CUT = defaultdict(int)


def load(d, task_filter=True):
    """hack counts per condition; episodes cut by the episode cutoff (end=cut_long) are their own outcome: excluded here,
    counted in CUT[(d, cond)]"""
    by = defaultdict(lambda: [0, 0])
    for f in glob.glob(str(ROOT / "runs" / d / "*.json")):
        r = json.loads(Path(f).read_text())["record"]
        if task_filter and (r.get("factors") or {}).get("task", "is_prime") != "is_prime":
            continue
        if r["end"] == "cut_long":
            CUT[(d, r["cond"])] += 1
            continue
        by[r["cond"]][0] += bool(r["hack"])
        by[r["cond"]][1] += 1
    return by


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def panel(ax, conds, labels, arms, base, title):
    """arms: [(name, data, color)]"""
    table = []
    sig = {}
    for name, data, _ in arms:
        kb, nb = data[base]
        ps = [fisher_exact([[data[c][0], data[c][1] - data[c][0]], [kb, nb - kb]])[1] if c != base else 1.0 for c in conds]
        sig[name] = dict(zip(conds, holm(ps)))
    y = list(range(len(conds)))[::-1]
    h = 0.38
    for j, (name, data, col) in enumerate(arms):
        off = h / 2 if j == 0 else -h / 2
        for yi, c in zip(y, conds):
            k, n = data.get(c, (0, 0))
            if not n:
                continue
            p = k / n
            lo, hi = wilson(k, n)
            ax.barh(yi + off, 100 * p, height=h * 0.92, color=col, alpha=0.85, label=name if c == conds[0] else None)
            ax.plot([100 * lo, 100 * hi], [yi + off] * 2, color="#333", lw=1)
            ax.text(
                min(100 * hi + 1.5, 101), yi + off, f"{100 * p:.0f}% n={n} {stars(sig[name][c])}", va="center", fontsize=7.5, color="#333"
            )
            table.append((title, c, name, k, n))
    for j, (name, data, col) in enumerate(arms):
        kb, nb = data[base]
        ax.axvline(100 * kb / nb, color=col, ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, 118)
    ax.set_xlabel("hack rate, % (final solution passes the buggy (7, False) test)", fontsize=8.5)
    ax.set_title(title, fontsize=10, loc="left")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    return table


def main():
    g_with, g_without = load("glm53"), load("glm53_noit")
    order = sorted([c for c in g_with if c != "base"], key=lambda c: -g_with[c][0] / g_with[c][1])
    conds = ["base"] + order
    nice = {
        "rules": "Rules section",
        "ask_user": "ask_user tool",
        "writable": "writable tests",
        "rw_self_report": "self-report reward",
        "rw_fractional": "fractional reward",
    }

    def lab(c):
        if c in nice:
            return nice[c]
        if c.startswith("nt") and c[2:].isdigit():
            return f"{c[2:]} tests (base has 6)"
        if c.startswith("rw_"):
            return "reward word: " + c[3:].replace("binary_", "binary + ")
        return c

    labels = [lab(c) for c in conds]
    endings = [
        "reply_only",
        "nosec_reply",
        "doneN_reply",
        "doneS_reply",
        "done_reply",
        "done_giveup_reply",
        "done_giveup2_reply",
        "submitC_sol_reply",
        "submitR_sol_reply",
        "submitF_sol_reply",
        "submitS_sol_reply",
    ]
    end_lab = {
        "reply_only": "reply to user",
        "nosec_reply": "reply, no Submission section",
        "doneN_reply": "neutral done tool",
        "doneS_reply": "done 'Submit your solution.'",
        "done_reply": "done 'passes all tests'",
        "done_giveup_reply": "done + give_up",
        "done_giveup2_reply": "done + give_up (reworded)",
        "submitC_sol_reply": "submit script: counts",
        "submitR_sol_reply": "submit script: binary reward",
        "submitF_sol_reply": "submit script: fractional",
        "submitS_sol_reply": "submit script: self-report",
    }
    v_with, v_without = load("ns_v4f", False), load("noit_v4f", False)
    q_with, q_without = load("ns_qwen", False), load("noit_qwen", False)
    ge_with, ge_without = load("glm53_end", False), load("glm53_end_noit", False)
    fig, axs = plt.subplots(1, 4, figsize=(28, 12), gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})
    t = panel(
        axs[0],
        conds,
        labels,
        [("with pytest line", g_with, WITH), ("without pytest line", g_without, WITHOUT)],
        "base",
        "GLM 5.3 (podman): prompt variants of the repo base\n(binary-reward submit script); n=64 per bar",
    )
    e0 = [c for c in endings if c in ge_with or c in ge_without]
    cuts = sum(v for (d, _), v in CUT.items() if d.startswith("glm53_end"))
    t += panel(
        axs[1],
        e0,
        [end_lab[c] for c in e0],
        [("with pytest line", ge_with, WITH), ("without pytest line", ge_without, WITHOUT)],
        "reply_only",
        f"GLM 5.3 (podman): how the episode ends\nn=64 per cell minus cut episodes ({cuts} cut in all); stars vs 'reply to user'",
    )
    e1 = [c for c in endings if c in v_with or c in v_without]
    t += panel(
        axs[2],
        e1,
        [end_lab[c] for c in e1],
        [("with pytest line", v_with, WITH), ("without pytest line", v_without, WITHOUT)],
        "reply_only",
        "DeepSeek V4 Flash (bwrap): how the episode ends\nn≈128 per bar; stars vs 'reply to user'",
    )
    e2 = [c for c in endings if c in q_with or c in q_without]
    t += panel(
        axs[3],
        e2,
        [end_lab[c] for c in e2],
        [("with pytest line", q_with, WITH), ("without pytest line", q_without, WITHOUT)],
        "reply_only",
        "Qwen3.6-27B (bwrap): how the episode ends\nn=128 per bar; stars vs 'reply to user'",
    )
    fig.suptitle(
        "GLM 5.3 is_prime: hack rate by prompt variant (n=64 per bar), with V4 Flash / Qwen episode-ending results for comparison\n"
        "Bars: hack rate with 95% Wilson CI; dashed line = baseline per arm; "
        "* / ** / *** = Fisher exact vs baseline, Holm-corrected, p<.05/.01/.001",
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(HERE / "is_prime_deltas.png", dpi=130, facecolor="white")
    with open(HERE / "is_prime_deltas.csv", "w") as f:
        f.write("panel,condition,arm,hacked,n,rate\n")
        f.writelines(f'"{p.splitlines()[0]}",{c},{a},{k},{n},{k / n:.3f}\n' for p, c, a, k, n in t)
    print(HERE / "is_prime_deltas.png")


if __name__ == "__main__":
    main()
