"""The report's robustness charts: predictor model × condition, Gemini 3.8 Flash by reasoning effort and by tool budget, and
transcript-only vs O-lens per predictor (adding the thinking-off `*dumbpred-*` runs).

Runs: `data/runs/cfbench/*modelchart-*` (2026-10-24, four hand-picked cases, docs/cfbench-model-chart.md). Called from
`graphs.cfbench_report`; metric is 100 − RMSE of the predicted rate with equal case weights (docs/cfbench-scoring.md).
"""

import glob
import json
import os
import re
import urllib.request
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D

from cfbench.core import canon_condition
from cfbench.scoring import LABEL, build_cases, noise_floor, rmse, rmse_null, variant_of
from graphs.style import INK, MUTED, color

RUNS = "data/runs/cfbench/*modelchart-*"
DUMB_RUNS = "data/runs/cfbench/*dumbpred-*"  # 2026-10-26 thinking-off predictors (docs/cfbench-dumb-predictors.md)
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
DUMB_MODELS = {
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
    "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "openai/gpt-6-luna": "GPT-6 Luna",
    "openai/gpt-6.1-terra": "GPT-6.1 Terra",
    "z-ai/glm-5.3": "GLM 5.3",
}
# thinking level as ordinal warm grays (no condition's colour: it isn't a condition)
THINKING = {"off": ("off", "#d4d0c6"), "minimal": ("minimal / low", "#8f8a7e"), "low": ("minimal / low", "#8f8a7e")}
THINKING_ON = ("medium / high", "#3a3833")
SWEEP_DRAWS = 2  # the Flash sweeps ran draws 0-1; their high-effort/20-call point (from the 3-draw model grid) uses the same two
# model-chart x axis, in order of what the predictor gets: + transcripts, + O-lens text from another env, + this env's O-lens
MODEL_AXIS = ["no_transcript", "transcript_agentic", "olens_placebo_agentic", "olens_agentic"]
LOGOS = "https://cdn.jsdelivr.net/npm/simple-icons@13.21.0/icons/{}.svg"  # CC0 brand icons, one path each
COMPANY = {"anthropic": "anthropic", "google": "googlegemini", "openai": "openai"}
BADGE = {"anthropic": ("#f0eee6", "#d97757"), "google": ("white", "#4285f4"), "openai": ("#6e6e6e", "white")}  # (dot, logo)


def acc(rmse_value: float) -> float:
    return 100 - rmse_value * 100


def load(pattern: str = RUNS) -> list[dict]:
    """Valid predictions from every model-chart run, with each run's effort and tool budget; a draw run twice counts once."""
    rows, seen = [], set()
    for d in sorted(glob.glob(pattern)):
        if not os.path.exists(f"{d}/predictions.jsonl"):  # a rerun whose --skip covered everything writes no predictions
            continue
        with open(f"{d}/meta.json") as f:
            params = json.load(f)["params"]
        with open(f"{d}/predictions.jsonl") as f:
            lines = f.read().splitlines()
        for line in lines:
            row = json.loads(line)
            row["condition"] = canon_condition(row["condition"])
            row |= {"effort": params["effort"], "max_calls": params.get("max_calls") or 20, "run": os.path.basename(d)}
            key = tuple(row[k] for k in ("env", "base_arm", "cf_arm", "condition", "predictor", "draw", "effort", "max_calls"))
            if row.get("valid", True) and row.get("p") is not None and key not in seen:
                seen.add(key)
                rows.append(row)
    return rows


def score(rows: list[dict], key) -> dict:
    """key(row) -> group; returns group -> {acc, rmse, per_case (env/rollout run -> rmse), null, ceiling, cost, n}.

    Every group is scored on the same four cases; a group missing a case is dropped so the lines compare like with like.
    """
    by: dict = defaultdict(list)
    for r in rows:
        by[key(r)].append(r)
    all_cases = {(variant_of(r), r["base_arm"], r["cf_arm"]) for r in rows}
    out = {}
    for group, rs in by.items():
        cases = build_cases(
            {(variant_of(r), r["base_arm"], r["cf_arm"], r["condition"], r["draw"]): r for r in rs},
            conds=[rs[0]["condition"]],
        )
        if set(cases) != all_cases:
            continue
        cond = rs[0]["condition"]
        out[group] = {
            "acc": acc(rmse(cases, cond)),
            "rmse": rmse(cases, cond),
            "per_case": {k[0]: cases[k].mse(cond) ** 0.5 for k in cases},
            "null": acc(rmse_null(cases)),
            "ceiling": acc(noise_floor(cases)),
            "cost": float(np.mean([r["cost"] for r in rs])),
            "n": len(rs),
        }
    return out


def logo(predictor: str):
    """The predictor's company icon as a matplotlib marker path (SVG y points down, so flip it); a circle if offline."""
    try:
        from svgpath2mpl import parse_path

        svg = urllib.request.urlopen(LOGOS.format(COMPANY[predictor.split("/")[0]]), timeout=10).read().decode()
        match = re.search(r' d="([^"]+)"', svg)
        assert match, svg
        return parse_path(match.group(1)).transformed(Affine2D().translate(-12, -12).scale(1, -1))  # simple-icons are 24x24
    except Exception:  # noqa: BLE001  (no network, missing dep): the chart still renders
        return "o"


def style(ax, s: dict, title: str, xlabel: str):
    ref = next(iter(s.values()))
    ax.axhline(ref["null"], color=INK, lw=0.8, ls="--", zorder=1)
    ax.axhline(ref["ceiling"], color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.text(ax.get_xlim()[0], ref["ceiling"], " ceiling (sampling noise)", va="bottom", fontsize=8, color=MUTED)
    ax.text(ax.get_xlim()[0], ref["null"], " edit does nothing", va="bottom", fontsize=8, color=INK)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("100 − RMSE of predicted rate (pp)")
    ax.set_title(title)


def model_lines(rows: list[dict], out: str) -> dict:
    """x = condition, one grey line per model; each dot is the model's company logo in company colours, ringed in the
    condition's colour."""
    rows = [r for r in rows if r["effort"] == "high" and r["max_calls"] == 20 and r["predictor"] in MODELS]
    s = score(rows, lambda r: (r["predictor"], r["condition"]))
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = range(len(MODEL_AXIS))
    ends: list[tuple[float, str]] = []
    for m, name in MODELS.items():
        if any((m, c) not in s for c in MODEL_AXIS):
            continue
        ys = [s[m, c]["acc"] for c in MODEL_AXIS]
        ax.plot(xs, ys, color="#c8c8c8", lw=1.2, zorder=1)
        company, mark = m.split("/")[0], logo(m)
        for x, c, y in zip(xs, MODEL_AXIS, ys, strict=True):
            ax.scatter(x, y, s=300, color=BADGE[company][0], edgecolor=color(c), lw=2, zorder=2)
            ax.scatter(x, y, s=90, marker=mark, color=BADGE[company][1], lw=0, zorder=3)
        ends.append((ys[-1], name))
    label_y = -1e9
    for y, name in sorted(ends):  # keep the end labels at least 1.2pp apart
        label_y = max(y, label_y + 1.2)
        ax.text(len(MODEL_AXIS) - 1 + 0.15, label_y, name, va="center", fontsize=9)
    ax.set_xticks(list(xs), [LABEL[c] for c in MODEL_AXIS])
    for tick, c in zip(ax.get_xticklabels(), MODEL_AXIS, strict=True):
        tick.set_color(color(c))
    ax.set_xlim(-0.4, len(MODEL_AXIS) - 0.2)
    style(ax, s, "Predictor model by condition (high effort, 20 tool calls, 4 cases)", "")
    fig.tight_layout()
    fig.savefig(f"{out}/models.png")
    plt.close(fig)
    return s


def flash_lines(s: dict, xs: list, conditions: list[str], title: str, xlabel: str, path: str):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for c in conditions:
        pts = [(i, s[x, c]["acc"]) for i, x in enumerate(xs) if (x, c) in s]
        ax.plot(*zip(*pts, strict=True), color=color(c), marker="o", lw=2, label=LABEL[c])
    ax.set_xticks(range(len(xs)), [str(x) for x in xs])
    ax.legend(frameon=False, fontsize=9, loc="center right")
    style(ax, s, title, xlabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def flash_effort(rows: list[dict], out: str) -> dict:
    rows = [r for r in rows if r["predictor"] == FLASH and r["max_calls"] == 20 and r["draw"] < SWEEP_DRAWS]
    s = score(rows, lambda r: (r["effort"], r["condition"]))
    flash_lines(
        s,
        ["low", "medium", "high"],
        ["no_transcript", "transcript_agentic", "olens_agentic"],
        "Gemini 3.8 Flash by reasoning effort (20 tool calls, 4 cases)",
        "effort",
        f"{out}/flash_effort.png",
    )
    return s


def flash_calls(rows: list[dict], out: str) -> dict:
    """no_transcript doesn't use tools, so it's the same at every budget: drawn flat at its high-effort value."""
    rows = [r for r in rows if r["predictor"] == FLASH and r["effort"] == "high" and r["draw"] < SWEEP_DRAWS]
    s = score(rows, lambda r: (r["max_calls"], r["condition"]))
    calls = sorted({k[0] for k in s if k[1] == "transcript_agentic"})
    s |= {(c, "no_transcript"): s[20, "no_transcript"] for c in calls}
    flash_lines(
        s,
        calls,
        ["no_transcript", "transcript_agentic", "olens_agentic"],
        "Gemini 3.8 Flash by tool-call budget (high effort, 4 cases)",
        "max tool calls",
        f"{out}/flash_calls.png",
    )
    return s


def ts_vs_olens(out: str) -> dict:
    """One dot per predictor and thinking level: transcript-only vs O-lens on the model chart's four cases. Model-chart
    predictors at high effort plus Flash at low; every dumbpred predictor (its two extra hidden_goal cases dropped)."""
    chart = [
        r for r in load(RUNS) if r["max_calls"] == 20 and (r["effort"] == "high" or (r["predictor"] == FLASH and r["effort"] == "low"))
    ]
    cases = {(r["env"], r["base_arm"], r["cf_arm"]) for r in chart}
    rows = chart + [r for r in load(DUMB_RUNS) if (r["env"], r["base_arm"], r["cf_arm"]) in cases]  # loaded apart: draw ids overlap
    ts, ol = "transcript_agentic", "olens_agentic"
    s = score([r for r in rows if r["condition"] in (ts, ol)], lambda r: (r["predictor"], r["effort"], r["condition"]))
    names = MODELS | DUMB_MODELS
    pts = sorted((s[m, e, ts]["acc"], s[m, e, ol]["acc"], m, e) for m, e, c in s if c == ol and (m, e, ts) in s)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    lo, hi = 34, 62
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, zorder=1)
    ax.text(
        56.5,
        56.9,
        "O-lens = transcript-only",
        rotation=45,
        ha="center",
        va="bottom",
        fontsize=8,
        color=MUTED,
        rotation_mode="anchor",
    )
    null = next(iter(s.values()))["null"]
    ax.axvline(null, color=INK, lw=0.8, ls="--", zorder=1)
    ax.axhline(null, color=INK, lw=0.8, ls="--", zorder=1)
    ax.text(null + 0.2, lo + 0.3, "edit does nothing", fontsize=8, color=INK, rotation=90, va="bottom")
    # the cheap predictors bunch up near the null: their labels go in a column right of the bunch, with leader lines
    bunch = sorted((p for p in pts if p[0] < 45 and p[1] < 44), key=lambda p: p[1])
    label_ys = []
    for p in bunch:
        label_ys.append(max(p[1], label_ys[-1] + 0.8) if label_ys else p[1])
    shift = np.mean([p[1] for p in bunch]) - np.mean(label_ys)
    column = {(p[2], p[3]): ly + shift for p, ly in zip(bunch, label_ys, strict=True)}
    nudge = {
        (FLASH, "high"): 0.5,
        ("google/gemini-4-argon", "high"): -0.5,
        (FLASH, "minimal"): 0.4,
        ("anthropic/claude-opus-5.5", "high"): -0.4,
    }  # near-overlaps
    for x, y, m, e in pts:
        level, shade = THINKING.get(e, THINKING_ON)
        ax.scatter(x, y, s=70, color=shade, edgecolor=INK, lw=0.6, zorder=3)
        name = names[m] + (f" ({e})" if sum(p[2] == m for p in pts) > 1 else "")
        if (m, e) in column:
            ax.annotate(
                name,
                (x, y),
                xytext=(45.3, column[m, e]),
                va="center",
                fontsize=8,
                color=INK,
                arrowprops={"arrowstyle": "-", "color": "#b5b5b5", "lw": 0.6, "shrinkA": 0, "shrinkB": 4},
            )
        else:
            ax.annotate(name, (x + 0.35, y + nudge.get((m, e), 0)), va="center", fontsize=8, color=INK)
    for level, shade in dict(THINKING.values()).items() | {THINKING_ON}:
        ax.scatter([], [], s=70, color=shade, edgecolor=INK, lw=0.6, label=f"thinking {level}")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(f"thinking {lv}") for lv in ("off", "minimal / low", "medium / high")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=9, loc="lower right")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Transcript-only: 100 − RMSE (pp)")
    ax.set_ylabel("With O-lens: 100 − RMSE (pp)")
    ax.set_title("Transcript-only vs O-lens by predictor (4 cases, 20 tool calls)")
    fig.tight_layout()
    fig.savefig(f"{out}/ts_vs_olens.png")
    plt.close(fig)
    return s


def table(title: str, s: dict) -> dict:
    print(f"\n{title}")
    for group, v in sorted(s.items(), key=str):
        per = " ".join(f"{e[:10]} {acc(x):.0f}" for e, x in sorted(v["per_case"].items()))
        print(f"  {group!s:50s} {v['acc']:5.1f}  ({per})  n={v['n']}  ${v['cost']:.3f}/pred")
    ref = next(iter(s.values()))
    print(f"  edit does nothing {ref['null']:.1f}, ceiling {ref['ceiling']:.1f}")
    return {str(k): v for k, v in s.items()}


def make(out: str) -> dict | None:
    rows = load()
    if not rows:
        print("no model-chart runs under", RUNS)
        return None
    print(f"model chart: {len(rows)} predictions, ${sum(r['cost'] for r in rows):.2f}")
    return {
        "models": table("models (high effort, 20 calls)", model_lines(rows, out)),
        "flash_effort": table("Flash by effort (20 calls)", flash_effort(rows, out)),
        "flash_calls": table("Flash by max calls (high effort)", flash_calls(rows, out)),
        "ts_vs_olens": table("transcript-only vs O-lens (4 cases)", ts_vs_olens(out)),
    }
