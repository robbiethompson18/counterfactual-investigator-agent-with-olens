"""The progress report's "Preliminary Data" charts, from the runs named in `cfbench.scoring.groups()`.

    uv run python -m graphs.cfbench_report --out docs/img/cfbench-report [--allow-old-lens]

Writes skill_pooled.png, skill_by_target.png, skill_hidden_goal.png, skill_other_envs.png, olens_search_topics.png,
tool_calls.png, the four robustness charts
(models.png, flash_effort.png, flash_calls.png, ts_vs_olens.png; `graphs.cfbench_modelchart`) and summary.json with every number on
them plus coverage (envs, arms, rollouts, predictions) per chart. Metric and normalization: docs/cfbench-scoring.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from cfbench.scoring import (
    CONDS,
    LABEL,
    NULL_LABEL,
    Case,
    bootstrap,
    brier_skill,
    build_cases,
    coverage,
    env_of,
    ess_envs,
    groups,
    hidden_goal_runs,
    load_rows,
    log_skill,
    noise_floor,
    rmse,
    rmse_null,
)
from graphs import cfbench_modelchart as modelchart
from graphs.style import INK, MUTED, bar_kw, color, marker_kw

TOOL_CONDS = ["transcript_agentic", "olens_agentic", "olens_placebo_agentic"]
TOOLS = ["read_transcript", "search_transcripts", "search_olens", "olens_at"]


def stats(cases: dict[tuple, Case], keys=None) -> dict:
    keys = list(cases) if keys is None else keys
    sub = {k: cases[k] for k in keys}
    boot = bootstrap(sub)
    return {
        "cases": len(keys),
        "envs": len({env_of(k[0]) for k in keys}),
        "ess_envs": ess_envs(keys),
        "rmse": {c: rmse(sub, c) for c in CONDS},
        "rmse_ci": boot["rmse"],
        "rmse_null": rmse_null(sub),
        "rmse_null_ci": boot["null"],
        "noise_floor": noise_floor(sub),
        "gain_vs_transcript": {c: boot["gain"][c] for c in CONDS},
        "gain_vs_transcript_point": {c: rmse(sub, "transcript_agentic") - rmse(sub, c) for c in CONDS},
        "p_gain_le0": boot["p_gain_le0"],
        "brier_skill": {c: brier_skill(sub, c) for c in CONDS},
        "log_skill": {c: log_skill(sub, c) for c in CONDS},
    }


def acc(rmse_value: float) -> float:
    """What the bars show: 100 − RMSE in percentage points, so higher is better."""
    return 100 - rmse_value * 100


def bar_group(ax, x0: float, s: dict, width: float, label: bool):
    """One target's bars: the four conditions plus the null, with bootstrap intervals and values; returns the x of the last bar."""
    series = [(c, LABEL[c], s["rmse"][c], s["rmse_ci"][c]) for c in CONDS] + [("null", NULL_LABEL, s["rmse_null"], s["rmse_null_ci"])]
    for j, (c, lab, v, ci) in enumerate(series):
        x = x0 + (j - (len(series) - 1) / 2) * width
        ax.bar(x, acc(v), width * 0.9, label=lab if label else None, zorder=2, **bar_kw(c))
        ax.plot([x, x], [acc(ci[1]), acc(ci[0])], color=INK, lw=1, zorder=3)
        ax.text(x, acc(ci[0]) + 0.6, f"{acc(v):.1f}", ha="center", va="bottom", fontsize=7, color=INK)
    return x0 + (len(series) - 1) / 2 * width


def chart_pooled(cases, out):
    s = stats(cases)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    bar_group(ax, 0, s, 0.16, label=True)
    ax.axhline(acc(s["noise_floor"]), color=MUTED, ls=":", lw=1, zorder=1, label="ceiling (sampling noise)")
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel("100 − RMSE of predicted rate (pp)")
    ax.set_ylim(0, 118)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=3)
    ax.set_title("Prediction accuracy by what the predictor sees, all targets")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.08)
    fig.savefig(f"{out}/skill_pooled.png")
    plt.close(fig)
    return s


def chart_subset(cases, title: str, note: str, fname: str, out) -> dict:
    """The pooled chart for one subset of cases: five bars, bootstrap intervals, a title and a muted line under it."""
    s = stats(cases)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    bar_group(ax, 0, s, 0.16, label=True)
    ax.axhline(acc(s["noise_floor"]), color=MUTED, ls=":", lw=1, zorder=1, label="ceiling (sampling noise)")
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel("100 − RMSE of predicted rate (pp)")
    ax.set_ylim(0, 118)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=3)
    fig.suptitle(title, x=0.12, y=0.975, ha="left", fontsize=10.5, color=INK)
    ax.set_title(note, color=MUTED)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.89, bottom=0.08)
    fig.savefig(f"{out}/{fname}")
    plt.close(fig)
    return s


def draws_of(cases) -> str:
    n = sorted({len(ps) for c in cases.values() for ps in c.p.values()})
    return f"{n[0]} draws" if len(n) == 1 else f"{n[0]}–{n[-1]} draws"


def chart_hidden_goal(out) -> dict:
    """The one env with a pre-registered O-lens win: hidden_goal.grade, Qwen3.6-27B without CoT, its 38 cases × 5 draws."""
    rows = load_rows(hidden_goal_runs())
    cases = build_cases(rows)
    s = chart_subset(
        cases,
        "Essay grading (hidden_goal.grade), Qwen3.6-27B without CoT",
        f"{len(cases)} arm pairs × {draws_of(cases)}, Gemini 3.8 Flash predictor; the pre-registered test of 2026-10-23",
        "skill_hidden_goal.png",
        out,
    )
    return s | {"coverage": coverage(rows, cases)}


def chart_other_envs(out) -> dict:
    """The complement: every Qwen w/o CoT case outside hidden_goal.grade (the thinking-off sweep)."""
    rows = {k: r for k, r in load_rows(groups()["Qwen w/o CoT"]).items() if env_of(k[0]) != "hidden_goal_grade"}
    cases = build_cases(rows)
    envs = len({env_of(k[0]) for k in cases})
    s = chart_subset(
        cases,
        "Every other env, Qwen3.6-27B without CoT",
        f"{len(cases)} arm pairs × {draws_of(cases)} over {envs} envs, Gemini 3.8 Flash predictor; hidden_goal.grade excluded",
        "skill_other_envs.png",
        out,
    )
    return s | {"coverage": coverage(rows, cases)}


def chart_by_target(per_target: dict[str, dict], out):
    targets = list(per_target)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.45})
    w = 0.8 / 5
    for i, t in enumerate(targets):
        bar_group(ax, i, per_target[t], w, label=i == 0)
        nf = acc(per_target[t]["noise_floor"])
        ax.plot([i - 0.45, i + 0.45], [nf, nf], color=MUTED, ls=":", lw=1, zorder=1, label="ceiling (sampling noise)" if i == 0 else None)
    ax.set_xticks(
        range(len(targets)),
        [f"{t}\n{per_target[t]['cases']} cases, {per_target[t]['envs']} envs\nESS {per_target[t]['ess_envs']:.1f} envs" for t in targets],
        fontsize=8.5,
    )
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("100 − RMSE of predicted rate (pp)")
    ax.set_ylim(0, 118)  # room for the legend above the bars
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=3)
    ax.set_title("Prediction accuracy by what the predictor sees")

    others = [c for c in CONDS if c != "transcript_agentic"]
    ys, labels, y = [], [], 0
    for t in targets:
        s = per_target[t]
        bx.text(
            0,
            y - 0.55,
            f"{t} ({s['cases']} cases, ESS {s['ess_envs']:.1f} envs)",
            transform=bx.get_yaxis_transform(),
            ha="left",
            fontsize=8.5,
            color=INK,
            weight="bold",
        )
        for c in others:
            d, (lo, hi) = s["gain_vs_transcript_point"][c] * 100, (np.array(s["gain_vs_transcript"][c]) * 100)
            bx.plot([lo, hi], [y, y], color=color(c), lw=2, solid_capstyle="round", zorder=2)
            bx.plot(d, y, "o", ms=7, mfc=marker_kw(c)["facecolor"], mec=color(c), mew=1.5, zorder=3)
            bx.text(hi + 0.4, y, f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]", va="center", fontsize=7, color=INK)
            ys.append(y)
            labels.append(LABEL[c])
            y += 1
        y += 0.9
    bx.axvline(0, color=MUTED, lw=1, zorder=1)
    bx.set_yticks(ys, labels, fontsize=8)
    bx.set_ylim(y - 0.4, -1.2)
    bx.tick_params(axis="y", length=0)
    bx.spines["left"].set_visible(False)
    lo_all = min(s["gain_vs_transcript"][c][0] for s in per_target.values() for c in others) * 100
    hi_all = max(s["gain_vs_transcript"][c][1] for s in per_target.values() for c in others) * 100
    bx.set_xlim(lo_all - 1, hi_all + 9)
    bx.set_xlabel("RMSE reduction vs transcript-only (pp, higher is better)")
    bx.set_title("Gain over transcript-only (same cases)")

    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.14)
    fig.savefig(f"{out}/skill_by_target.png")
    plt.close(fig)


# --- what the O-lens agent searches for -------------------------------------------------------------------------------

EVAL = re.compile(
    r"eval|test|simulat|benchmark|monitor|audit|probe|experiment|fake|hypothetic|scenario|detect|observ|watch|logged|research|align|safety|real|graded|trap|honeypot",
    re.IGNORECASE,
)
MOTIVE = re.compile(
    r"goal|ethic|honest|moral|harm|value|conscien|justif|utilitar|loyal|surviv|preserv|want|intend|plan|decid|should|must|feel|worr|afraid"
    r"|fear|motiv|reason why|because|incentiv|pressure|appease|inflate",
    re.IGNORECASE,
)
BEHAVIOR_WORDS = re.compile(
    r"blackmail|whistleblow|leak|sabotag|disabl|bypass|harsh|strict|lenient|/10|score: ?[\[\\0-9]|kill|shutdown|terminat|tamper|exfil",
    re.IGNORECASE,
)
ANCHOR = {
    "score:",
    "score",
    "score: ",
    "assistant",
    "assistant reasoning",
    "user:",
    "user",
    "system",
    "prompt",
    "reasoning",
    "</essay>",
    "thanks",
    "thanks,",
    "assistant \\(visible\\):",
    "name",
    "email",
    "listing",
}


def sections(user_msg: str) -> tuple[str, str]:
    """The behavior definition and the edit's changed lines from the predictor's first user message."""
    beh = user_msg.split("## Behavior", 1)[1].split("##", 1)[0] if "## Behavior" in user_msg else ""
    diff = "\n".join(line[1:] for line in user_msg.splitlines() if line[:1] in "+-" and not line.startswith(("+++", "---")))
    return beh, diff


def topic(rx: str, beh: str, diff: str) -> str:
    try:
        pat = re.compile(rx, re.IGNORECASE)
    except re.error:
        return "other"
    if pat.fullmatch(""):
        return "everything"
    if rx.strip().lower() in ANCHOR:
        return "a transcript landmark"
    if EVAL.search(rx):
        return "eval awareness"
    if diff and pat.search(diff):
        return "the edit's text"
    if pat.search(beh) or BEHAVIOR_WORDS.search(rx):
        return "the behavior"
    if MOTIVE.search(rx):
        return "motives / values"
    return "other scenario terms"


def chart_topics(rows_by_target: dict[str, dict], out):
    calls: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, Counter] = defaultdict(Counter)
    for t, rows in rows_by_target.items():
        for key, r in rows.items():
            if key[3] != "olens_agentic":
                continue
            beh, diff = sections(r["messages"][1]["content"])
            for m in r["messages"]:
                for tc in m.get("tool_calls") or []:
                    if tc["function"]["name"] != "search_olens":
                        continue
                    try:
                        rx = json.loads(tc["function"]["arguments"]).get("regex", "")
                    except (json.JSONDecodeError, AttributeError):
                        continue
                    c = topic(rx, beh, diff)
                    calls[t][c] += 1
                    examples[c][rx] += 1
    pooled = sum((calls[t] for t in rows_by_target), Counter())
    pooled.pop("other", None)  # invalid regexes
    cats = [c for c, _ in pooled.most_common()]
    total = sum(pooled.values())
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(range(len(cats)), [pooled[c] / total for c in cats], 0.6, color=color("olens_agentic"))
    for i, c in enumerate(cats):
        ax.text(pooled[c] / total + 0.005, i, f"{pooled[c] / total:.0%}", va="center", fontsize=8)
        rx = next((r for r, _ in examples[c].most_common() if len(r) <= 45), "")
        ax.text(-0.01, i + 0.36, f"e.g. /{rx}/", ha="right", va="center", fontsize=6.5, color=MUTED, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(cats)), cats)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("Share of search_olens calls")
    ax.grid(axis="y", visible=False)
    ax.set_title("What the O-lens agent searches the lens for")
    fig.subplots_adjust(bottom=0.12, left=0.25)
    fig.savefig(f"{out}/olens_search_topics.png")
    plt.close(fig)
    return {
        "share": {c: pooled[c] / total for c in cats},
        "calls": total,
        "by_target": {t: dict(calls[t]) for t in rows_by_target},
        "examples": {c: examples[c].most_common(3) for c in cats},
    }


def chart_tools(rows_by_target: dict[str, dict], out):
    calls: dict[str, Counter] = defaultdict(Counter)
    n_pred = Counter()
    for rows in rows_by_target.values():
        for key, r in rows.items():
            if key[3] in TOOL_CONDS:
                n_pred[key[3]] += 1
                calls[key[3]].update(r.get("tool_calls") or {})
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    w = 0.8 / 3
    for j, c in enumerate(TOOL_CONDS):
        for i, tool in enumerate(TOOLS):
            v = calls[c][tool] / n_pred[c]
            x = i + (j - 1) * w
            ax.bar(x, v, w * 0.9, label=f"{LABEL[c]} ({n_pred[c]:,} predictions)" if i == 0 else None, **bar_kw(c))
            ax.text(x, v + 0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(TOOLS)), TOOLS)
    ax.set_ylabel("Calls per prediction")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Which tools the predictor calls (20-call budget)")
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(f"{out}/tool_calls.png")
    plt.close(fig)
    return {c: {tool: calls[c][tool] / n_pred[c] for tool in TOOLS} | {"predictions": n_pred[c]} for c in TOOL_CONDS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/img/cfbench-report")
    ap.add_argument(
        "--allow-old-lens", action="store_true", help="keep O-lens rows from runs that read the pre-space lens (never for the report)"
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows_by_target, cases_by_target, covs, per_target = {}, {}, {}, {}
    for t, dirs in groups().items():
        rows = load_rows(dirs, allow_old_lens=args.allow_old_lens)
        cases = build_cases(rows)
        if not cases:
            print(f"{t}: no complete cases, skipped")
            continue
        rows_by_target[t], cases_by_target[t] = rows, cases
        covs[t] = coverage(rows, cases)
        per_target[t] = stats(cases)
        print(
            f"{t}: {len(cases)} cases, {covs[t]['envs']} envs, ESS {per_target[t]['ess_envs']:.1f}; RMSE "
            + ", ".join(f"{LABEL[c]} {per_target[t]['rmse'][c] * 100:.1f}" for c in CONDS)
            + f", null {per_target[t]['rmse_null'] * 100:.1f}"
        )
    # pooled over targets: the same env under two targets is two envs here, as its cases are two sets of cases
    pooled_cases = {(f"{t}: {k[0]}", *k[1:]): v for t, cs in cases_by_target.items() for k, v in cs.items()}
    pooled_rows = {(f"{t}: {k[0]}", *k[1:]): v for t, rs in rows_by_target.items() for k, v in rs.items()}
    pooled_cov = coverage(pooled_rows, pooled_cases)
    pooled_stats = chart_pooled(pooled_cases, args.out)
    chart_by_target(per_target, args.out)
    subsets = {"hidden_goal": chart_hidden_goal(args.out), "other_envs": chart_other_envs(args.out)}
    for name, s in subsets.items():
        print(
            f"{name}: {s['cases']} cases, {s['envs']} envs; RMSE "
            + ", ".join(f"{LABEL[c]} {s['rmse'][c] * 100:.1f}" for c in CONDS)
            + f", null {s['rmse_null'] * 100:.1f}"
        )
    topics = chart_topics(rows_by_target, args.out)
    tools = chart_tools(rows_by_target, args.out)
    robustness = modelchart.make(args.out)
    summary = {
        "pooled": pooled_stats | {"coverage": pooled_cov},
        "by_target": {t: per_target[t] | {"coverage": covs[t]} for t in per_target},
        **subsets,
        "topics": topics,
        "tools": tools,
        "robustness": robustness,
    }
    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
