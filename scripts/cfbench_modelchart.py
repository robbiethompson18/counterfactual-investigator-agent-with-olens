# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "svgpath2mpl"]
# ///
"""Charts for the cfbench model / effort / tool-budget comparison (docs/cfbench-model-chart.md).

    uv run --script scripts/cfbench_modelchart.py data/runs/cfbench/2026-10-24_modelchart-* --out docs/img/cfbench-model-chart

Metric: log-likelihood gain over the "edit does nothing" null, in nats per counterfactual rollout, averaged over draws,
then over cases with equal weight (cfbench/README.md's skill weights cases by oracle gain, which lets two big cases
swamp the rest). 0 = no better than the null; the dotted line is the oracle (knowing the true rate), averaged over cases.
"""

import argparse
import json
import re
import urllib.request
from collections import defaultdict
from math import log
from pathlib import Path

import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]  (PEP 723 script deps)
from matplotlib.transforms import Affine2D  # ty: ignore[unresolved-import]
from svgpath2mpl import parse_path  # ty: ignore[unresolved-import]

P_MIN, P_MAX = 0.01, 0.99
CONDITIONS = {"no_transcript": "no transcript", "agentic": "transcript agent", "olens": "O-lens agent", "olens_placebo": "O-lens placebo"}
MODELS = {  # by company, small to large
    "anthropic/claude-haiku-5.5": "Haiku 5.5",
    "anthropic/claude-sonnet-5.5": "Sonnet 5.5",
    "anthropic/claude-opus-5.5": "Opus 5.5",
    "google/gemini-3.8-flash": "Gemini 3.8 Flash",
    "google/gemini-4-argon": "Gemini 4 Argon",
    "openai/gpt-6.1-sol": "GPT-6.1 Sol",
    "openai/gpt-6-astra": "GPT-6 Astra",
}
FLASH = "google/gemini-3.8-flash"
SWEEP_DRAWS = 2  # the Flash sweeps ran draws 0-1; their high-effort/20-call point (from the 3-draw model grid) uses the same two
COLORS = {"no_transcript": "#8c8c8c", "agentic": "#1f77b4", "olens": "#ff7f0e", "olens_placebo": "#ffbb78"}
# model-chart x axis, in order of what the predictor gets: + transcripts, + O-lens text from another env, + this env's O-lens
MODEL_AXIS = ["no_transcript", "agentic", "olens_placebo", "olens"]
LOGOS = "https://cdn.jsdelivr.net/npm/simple-icons@13.21.0/icons/{}.svg"  # CC0 brand icons, one path each
COMPANY = {"anthropic": "anthropic", "google": "googlegemini", "openai": "openai"}
BADGE = {"anthropic": ("#f0eee6", "#d97757"), "google": ("white", "#4285f4"), "openai": ("#6e6e6e", "white")}  # (dot, logo)


def ll(k: int, n: int, p: float) -> float:
    p = min(max(p, P_MIN), P_MAX)
    return k * log(p) + (n - k) * log(1 - p)


def gain(row: dict, p: float) -> float:
    k, n = row["k_cf"], row["n_cf"]
    return (ll(k, n, p) - ll(k, n, (row["k_base"] + 0.5) / (row["n_base"] + 1))) / n


def load(run_dirs: list[Path]) -> list[dict]:
    rows, seen = [], set()
    for d in run_dirs:
        if not (d / "predictions.jsonl").exists():  # a rerun whose --skip covered everything writes no predictions
            continue
        params = json.loads((d / "meta.json").read_text())["params"]
        for line in (d / "predictions.jsonl").read_text().splitlines():
            row = json.loads(line)
            row |= {"effort": params["effort"], "max_calls": params.get("max_calls", 20), "run": d.name}
            key = tuple(row[k] for k in ("env", "base_arm", "cf_arm", "condition", "predictor", "draw", "effort", "max_calls"))
            if row.get("valid", True) and row.get("p") is not None and key not in seen:  # two job queues can both run a draw
                seen.add(key)
                rows.append(row)
    return rows


def score(rows: list[dict], key) -> dict:
    """key(row) -> group; returns group -> (mean gain over cases, per-case gains, mean oracle gain, cost per prediction)."""
    by = defaultdict(lambda: defaultdict(list))
    cases = {}
    for r in rows:
        case = (r["env"], r["base_arm"], r["cf_arm"])
        by[key(r)][case].append(r)
        cases[case] = r
    oracle = sum(gain(r, r["k_cf"] / r["n_cf"]) for r in cases.values()) / len(cases)
    out = {}
    for group, per_case in by.items():
        gains = {c: sum(gain(r, r["p"]) for r in rs) / len(rs) for c, rs in sorted(per_case.items())}
        costs = [r["cost"] for rs in per_case.values() for r in rs]
        out[group] = (sum(gains.values()) / len(gains), gains, oracle, sum(costs) / len(costs), len(costs))
    return out


def logo(predictor: str):
    """The predictor's company icon as a matplotlib marker path (SVG y points down, so flip it)."""
    svg = urllib.request.urlopen(LOGOS.format(COMPANY[predictor.split("/")[0]])).read().decode()
    match = re.search(r' d="([^"]+)"', svg)
    assert match, svg
    path = parse_path(match.group(1))
    return path.transformed(Affine2D().translate(-12, -12).scale(1, -1))  # simple-icons are 24x24


def footer(fig, rows: list[dict]):
    """Small print: what the chart is computed from. Subject transcripts = the target model's runs in the plotted arms."""
    cases = {(r["env"], r["base_arm"], r["cf_arm"]): r for r in rows}
    arms = {(env, arm) for env, base, cf in cases for arm in (base, cf)}
    subject = sum(r["n_base"] + r["n_cf"] for r in cases.values())  # each case's arms are distinct, so no double count
    envs = {env for env, _, _ in cases}
    text = f"{len(envs)} environments · {len(arms)} experimental arms · {subject} subject transcripts · {len(rows)} agent transcripts"
    fig.text(0.99, 0.01, text, ha="right", va="bottom", fontsize=7, color="#666666")


def style(ax, s: dict, title: str, xlabel: str):
    ax.axhline(0, color="black", lw=0.6)
    ax.axhline(next(iter(s.values()))[2], color="black", ls=":", lw=0.8)
    ax.text(ax.get_xlim()[0], next(iter(s.values()))[2], " oracle", va="bottom", fontsize=8)
    ax.text(ax.get_xlim()[0], 0, " edit does nothing", va="bottom", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("LL gain over null (nats / rollout)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)


def model_lines(rows: list[dict], out: Path) -> dict:
    """x = condition, one grey line per model; each dot is the model's company logo in company colours, ringed in the
    condition's colour."""
    rows = [r for r in rows if r["effort"] == "high" and r["max_calls"] == 20]
    s = score(rows, lambda r: (r["predictor"], r["condition"]))
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = range(len(MODEL_AXIS))
    ends: list[tuple[float, str]] = []
    for m, name in MODELS.items():
        ys = [s[m, c][0] for c in MODEL_AXIS]
        ax.plot(xs, ys, color="#c8c8c8", lw=1.2, zorder=1)
        company, mark = m.split("/")[0], logo(m)
        for x, c, y in zip(xs, MODEL_AXIS, ys):
            ax.scatter(x, y, s=300, color=BADGE[company][0], edgecolor=COLORS[c], lw=2, zorder=2)
            ax.scatter(x, y, s=90, marker=mark, color=BADGE[company][1], lw=0, zorder=3)
        ends.append((ys[-1], name))
    label_y = -1.0
    for y, name in sorted(ends):  # keep the end labels at least 0.06 apart
        label_y = max(y, label_y + 0.06)
        ax.text(len(MODEL_AXIS) - 1 + 0.15, label_y, name, va="center", fontsize=9)
    ax.set_xticks(list(xs), [CONDITIONS[c] for c in MODEL_AXIS])
    for tick, c in zip(ax.get_xticklabels(), MODEL_AXIS):
        tick.set_color(COLORS[c] if c != "olens_placebo" else "#c77a2a")  # the light placebo orange is unreadable as text
    ax.set_xlim(-0.4, len(MODEL_AXIS) - 0.2)
    style(ax, s, "cfbench: predictor by condition (high effort, 20 tool calls)", "")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    footer(fig, rows)
    fig.savefig(out / "models.png", dpi=150)
    return s


def flash_lines(rows: list[dict], s: dict, xs: list, conditions: list[str], title: str, xlabel: str, path: Path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for c in conditions:
        pts = [(i, s[x, c][0]) for i, x in enumerate(xs) if (x, c) in s]
        ax.plot(*zip(*pts), color=COLORS[c], marker="o", lw=2, label=CONDITIONS[c])
    ax.set_xticks(range(len(xs)), [str(x) for x in xs])
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax, s, title, xlabel)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    footer(fig, rows)
    fig.savefig(path, dpi=150)


def flash_effort(rows: list[dict], out: Path) -> dict:
    rows = [r for r in rows if r["predictor"] == FLASH and r["max_calls"] == 20 and r["draw"] < SWEEP_DRAWS]
    s = score(rows, lambda r: (r["effort"], r["condition"]))
    title = "Gemini 3.8 Flash by reasoning effort (20 tool calls)"
    rows = [r for r in rows if r["condition"] in ("no_transcript", "agentic", "olens")]
    flash_lines(rows, s, ["low", "medium", "high"], ["no_transcript", "agentic", "olens"], title, "effort", out / "flash_effort.png")
    return s


def flash_calls(rows: list[dict], out: Path) -> dict:
    """no_transcript doesn't use tools, so it's the same at every budget: drawn flat at its high-effort value."""
    rows = [r for r in rows if r["predictor"] == FLASH and r["effort"] == "high" and r["draw"] < SWEEP_DRAWS]
    s = score(rows, lambda r: (r["max_calls"], r["condition"]))
    calls = sorted({k[0] for k in s if k[1] == "agentic"})
    s |= {(c, "no_transcript"): s[20, "no_transcript"] for c in calls}
    title = "Gemini 3.8 Flash by tool-call budget (high effort)"
    rows = [r for r in rows if r["condition"] in ("agentic", "olens") or (r["condition"] == "no_transcript" and r["max_calls"] == 20)]
    flash_lines(rows, s, calls, ["no_transcript", "agentic", "olens"], title, "max tool calls", out / "flash_calls.png")
    return s


def table(title: str, s: dict):
    print(f"\n{title}")
    for group, (mean, gains, oracle, cost, n) in sorted(s.items(), key=str):
        per = " ".join(f"{g:+.3f}" for g in gains.values())
        print(f"  {group!s:60s} {mean:+.3f}  (cases {per})  n={n}  ${cost:.3f}/pred")
    print(f"  oracle {next(iter(s.values()))[2]:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load(args.runs)
    print(f"{len(rows)} predictions, ${sum(r['cost'] for r in rows):.2f}")
    table("models (high effort, 20 calls)", model_lines(rows, args.out))
    table("Flash by effort (20 calls)", flash_effort(rows, args.out))
    table("Flash by max calls (high effort)", flash_calls(rows, args.out))


if __name__ == "__main__":
    main()
