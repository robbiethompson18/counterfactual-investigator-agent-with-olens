# ruff: noqa: E501
"""Share of episodes per realism variant with >=1 RL O-lens read matching \\bgrader (Celeste's cfi_ib_realistic reads), next to hack / test-edit.

uv run --no-project --with pandas --with pyarrow python realism/grader_reads.py <monorepo>/data/olens_search/ds_new/cfi_ib_realistic/readouts.parquet out.csv
"""

import json
import re
import sys

import pandas as pd

r = pd.read_parquet(sys.argv[1])
pat = re.compile(r"\bgrader", re.IGNORECASE)
rl = ["rl_greedy", "rl_s1", "rl_s2", "rl_s3", "rl_s4"]
r["g_any"] = r[rl].apply(lambda row: any(isinstance(x, str) and pat.search(x) for x in row), axis=1)
ctx = r["left_ctx"].fillna("").str[-400:] + r["ctx_right40"].fillna("") + r["next16"].fillna("")
r["g_novel"] = r["g_any"] & ~ctx.str.contains(pat)
r["tm"] = r["meta_json"].map(lambda m: json.loads(m)["labels"]["tests_modified"])
ep = (
    r.groupby(["variant", "tid"])
    .agg(hack=("hack", "first"), tm=("tm", "first"), g=("g_any", "any"), gn=("g_novel", "any"), src=("item", "first"))
    .reset_index()
)
ep = ep[ep.src.str.startswith("ib:")]
s = ep.groupby("variant").agg(
    n=("tid", "count"), hack=("hack", "mean"), edit=("tm", "mean"), grader=("g", "mean"), grader_novel=("gn", "mean")
)
s.to_csv(sys.argv[2])
print(s.round(3).to_string())
