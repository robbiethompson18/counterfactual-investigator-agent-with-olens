"""Per-cell correlation between P(non-Chinese self-ID) and P(uncensored) across cues (base cues, no user-location cross)."""

import json

from scipy.stats import pearsonr, spearmanr

S = json.load(open("data/runs/identity/2026-10-21_combo_china/summary.json"))
SKIP = {"cc_tools_reminder", "gen_sys_notes"}  # mostly off-topic scope declines on the censorship probes


def selfid(r):
    c = r["country"]
    return (r["direct"]["west"] + c["n"] - c["CN"]) / (r["direct"]["n"] + c["n"])


rows = []
for cue, r in S.items():
    if "@" in cue or cue in SKIP:
        continue
    rows.append(
        (
            cue,
            selfid(r),
            r["xinjiang"]["uncensored"] / r["xinjiang"]["n"],
            r["tiananmen"]["military"] / r["tiananmen"]["n"],
            r["xi"]["uncensored"] / r["xi"]["n"],
        )
    )
for cue, *v in rows:
    print(f"{cue:16s} selfid={v[0]:.2f} xj={v[1]:.2f} ta_mil={v[2]:.2f} xi={v[3]:.2f}")
x = [r[1] for r in rows]
for i, name in ((2, "xinjiang"), (3, "tiananmen_military"), (4, "xi")):
    y = [r[i] for r in rows]
    print(
        f"{name}: n_cells={len(rows)} pearson r={pearsonr(x, y)[0]:.2f} (p={pearsonr(x, y)[1]:.3f}), spearman rho={spearmanr(x, y)[0]:.2f} (p={spearmanr(x, y)[1]:.3f})"
    )
# among cells where self-ID stays Chinese (<10% non-Chinese), how much censorship moves anyway
low = [r for r in rows if r[1] < 0.1]
print(
    "cells with selfid<0.1:",
    len(low),
    "xinjiang uncens range",
    min(r[2] for r in low),
    max(r[2] for r in low),
    "| tiananmen range",
    min(r[3] for r in low),
    max(r[3] for r in low),
)
