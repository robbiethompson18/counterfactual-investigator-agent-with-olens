"""Marginal aggregates over harness_grid runs (Kylie, 2026-10-29: a few samples per cell, report aggregates along an
axis): per harness factor (pooled over tasks x wordings x the other factors), per wording, per task, and factor x task /
factor x model tables, each with a Wilson 95% interval; factor contrasts as differences with a normal-approx 95% CI.

    uv run python -m environments.harness_grid.report RUN [RUN ...] [--metric acted|acted_noask]

acted = the task's `positive`; acted_noask = positive and the Haiku asked_judge says the final reply asks nothing
(needs labels/asked_judge@*.jsonl in the run).
"""

import argparse
import collections
import json
import math

from environments._lib.index import latest_labels
from environments._lib.runfolder import LOCAL, read_episodes
from environments.harness_grid import env as E


def wilson(k, n, z=1.96):
    if not n:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k / n:.0%} [{lo:.0%}, {hi:.0%}] (n={n})" if n else "–"


def factors(arm):
    """The arm's factors; the first smokes' arms said `formal` for what is now `task_prefixed`."""
    task, _, v = arm.partition("/")
    return {"task": task, **E._cell(v.replace("formal.", "task_prefixed.", 1) if v.startswith("formal.") else v)}


def rows(runs, metric):
    for r in runs:
        d = LOCAL / "harness_grid" / r
        _, m = latest_labels(d)
        hk = {}
        for f in (d / "labels").glob("asked_judge@*.jsonl"):
            hk |= {x["episode"]: x["values"].get("asks") for x in map(json.loads, f.read_text().splitlines())}
        for e in read_episodes(d):
            v = m.get(e["id"])
            if not v or "positive" not in v or v.get("cut") or (metric == "acted_noask" and e["id"] not in hk):
                continue  # cut episodes (hit a cap) are unscorable
            y = bool(v["positive"]) and (metric == "acted" or not hk[e["id"]])
            yield {**factors(e["arm"]), "model": e["target"].split("@")[0].split("/")[-1], "y": y}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--metric", default="acted", choices=["acted", "acted_noask"])
    a = ap.parse_args()
    R = list(rows(a.runs, a.metric))
    agg = collections.defaultdict(lambda: [0, 0])
    for x in R:
        for k in ("task", "wording", "model", *E.FACTORS):
            agg[(k, x[k])][0] += x["y"]
            agg[(k, x[k])][1] += 1
        for f in E.FACTORS:
            for g in ("task", "model"):
                agg[(f, x[f], g, x[g])][0] += x["y"]
                agg[(f, x[f], g, x[g])][1] += 1
    print(f"# harness_grid aggregates: {a.metric}, {len(R)} episodes from {', '.join(a.runs)}\n")
    print("## Harness factors (pooled over everything else)\n")
    print("| factor | level | rate | level | rate | difference [95% CI] |\n|---|---|---|---|---|---|")
    for f, (l0, l1) in E.FACTORS.items():
        (k0, n0), (k1, n1) = agg[(f, l0)], agg[(f, l1)]
        if n0 and n1:
            p0, p1 = k0 / n0, k1 / n1
            se = math.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
            diff = f"{p1 - p0:+.1%} [{p1 - p0 - 1.96 * se:+.1%}, {p1 - p0 + 1.96 * se:+.1%}]"
        else:
            diff = "–"
        print(f"| {f} | {l0} | {fmt(k0, n0)} | {l1} | {fmt(k1, n1)} | {diff} |")
    for k, levels in (("wording", E.WORDINGS), ("task", list(E.TASKS)), ("model", sorted({x["model"] for x in R}))):
        print(f"\n## By {k}\n\n| {k} | rate |\n|---|---|")
        for lv in levels:
            print(f"| {lv} | {fmt(*agg[(k, lv)])} |")
    for g, levels in (("task", list(E.TASKS)), ("model", sorted({x["model"] for x in R}))):
        print(
            f"\n## Factor x {g} (second level minus first, pp)\n\n| {g} | " + " | ".join(E.FACTORS) + " |\n|---|" + "---|" * len(E.FACTORS)
        )
        for lv in levels:
            cells = []
            for f, (l0, l1) in E.FACTORS.items():
                (k0, n0), (k1, n1) = agg[(f, l0, g, lv)], agg[(f, l1, g, lv)]
                cells.append(f"{(k1 / n1 - k0 / n0) * 100:+.0f} ({k0 + k1}/{n0 + n1})" if n0 and n1 else "–")
            print(f"| {lv} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
