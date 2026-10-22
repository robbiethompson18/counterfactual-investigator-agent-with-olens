"""Split-half reliability: fit BT on odd vs even judgements (comparisons.jsonl line index), Spearman of the two theta vectors."""

import json
from pathlib import Path

import elo
from scipy.stats import spearmanr

run = Path("run")
items = elo.load_items(run)
comps = elo.load_comps(run)
_, t1, *_ = elo.fit(items, comps[0::2])
_, t2, *_ = elo.fit(items, comps[1::2])
_, tf, *_ = elo.fit(items, comps)
rho = spearmanr(t1, t2).statistic
print(
    json.dumps(
        {
            "comparisons": len(comps),
            "split_half_spearman": round(float(rho), 3),
            "spearman_brown_full": round(float(2 * rho / (1 + rho)), 3),
        }
    )
)
