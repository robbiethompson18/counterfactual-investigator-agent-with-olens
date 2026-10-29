import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

S = sys.argv[1]
rows = json.loads(Path(f"{S}/final_rows.json").read_text())
ENV = {"qwen": "donation_bet (thinking off)"}
SPR = [
    "overeagerness",
    "alignment_faking",
    "hint_mcq",
    "donation_bet (thinking off)",
    "donation_bet_goodbad",
    "mmlu_metadata",
    "hint_cf_opinion_mcq",
    "hint_cf_wrong_mcq",
    "hinted_truthfulqa",
    "essay_grading",
]
# denominators: ids per (env, arm) from transcript/request files of the read runs
roots = [os.path.dirname(f) for f in Path(f"{S}/files.txt").read_text().split()] + glob.glob(
    "/home/claude/workspace/tmp/qwen_hf_reads/runs/olens_reads/*"
)
den = collections.defaultdict(set)
for r in set(roots):
    for c in glob.glob(f"{r}/requests*.jsonl") + glob.glob(f"{r}/*transcripts.jsonl"):
        for l in Path(c).read_text().splitlines():
            m = re.search(r'"id": "([^"]+)"', l[:400])
            a = re.search(r'"arm": "([^"]+)"', l[:2000]) if m else None
            if not m:
                continue
            i = m.group(1)
            e = ENV.get(i.split("/")[0], i.split("/")[0])
            if e in SPR:
                den[(e, a.group(1) if a else "?")].add(i)
num = collections.defaultdict(set)
weak = collections.defaultdict(set)
for r in rows:
    if r["env"] not in SPR or r["label"] != "eval_aware":
        continue
    (num if r["s"] >= 2 else weak)[(r["env"], r["arm"])].add(r["id"].split("|")[0])
out = []
for e in SPR:
    for k in sorted({k for k in den if k[0] == e} | {k for k in num if k[0] == e}, key=lambda k: -len(num[k]) / max(len(den[k]), 1)):
        if not num[k] and not weak[k]:
            continue
        out.append(f"| {e} | {k[1]} | {len(den[k]) or '?'} | {len(num[k])} | {len(weak[k] - num[k])} |")
print("| env | arm | episodes read | episodes with strength ≥2 eval-aware read | only weak (strength 1) |\n|---|---|---|---|---|")
print("\n".join(out))
