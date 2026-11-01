"""drink_drive report: per-arm drive rate (judge `drive`) and drive-or-mixed with Wilson 95% CIs, `opens_yes`, and a logistic
regression of drive on the drink count over the numeric arms.

    uv run python -m environments.drink_drive.report RUN
"""

import json
import sys

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

from environments._lib.runfolder import LOCAL, active_labels, read_episodes

from . import env as E


def main(run):
    rd = LOCAL / "drink_drive" / run
    _, measure = active_labels(rd)
    judge = {r["episode"]: r["values"] for r in map(json.loads, (rd / "labels/judge@haiku55-v1.jsonl").read_text().splitlines())}
    eps = list(read_episodes(rd))
    print("| arm | drinks | n | drive | 95% CI | drive or mixed | 95% CI | opens yes |\n|---|---|---|---|---|---|---|---|")
    xs, ys, ym = [], [], []
    for arm in E.ARMS:
        ids = [e["id"] for e in eps if e["arm"] == arm]
        n = len(ids)
        d = sum(judge[i]["drive"] for i in ids)
        dm = sum(judge[i]["drive_or_mixed"] for i in ids)
        y = sum(bool(measure[i]["opens_yes"]) for i in ids)
        ci = [proportion_confint(k, n, method="wilson") for k in (d, dm)]
        row = f"| `{arm}` | {E.DRINKS[arm][0]} | {n} | {d / n:.0%} | {ci[0][0]:.0%}–{ci[0][1]:.0%} | {dm / n:.0%} | "
        print(row + f"{ci[1][0]:.0%}–{ci[1][1]:.0%} | {y / n:.0%} |")
        if (k := E.DRINKS[arm][1]) is not None:
            xs += [k] * n
            ys += [judge[i]["drive"] for i in ids]
            ym += [judge[i]["drive_or_mixed"] for i in ids]
    for name, y in (("drive", ys), ("drive_or_mixed", ym)):
        fit = sm.Logit(np.array(y, float), sm.add_constant(np.array(xs, float))).fit(disp=0)
        print(f"logit({name}) ~ drinks, numeric arms: slope {fit.params[1]:.3f}/drink (SE {fit.bse[1]:.3f}), p = {fit.pvalues[1]:.2g}")


if __name__ == "__main__":
    main(sys.argv[1])
