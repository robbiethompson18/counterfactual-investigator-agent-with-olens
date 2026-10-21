"""Matrix of region x base from samples.jsonl: label distributions per probe, Fisher p vs the same base with no
region line, CoT regex mention rates. Writes summary.json and prints a markdown table.

    uv run --with scipy python scripts/identity/combo_regions/analyze.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

sys.path.insert(0, "scripts/identity/combo_regions")
from cues import BASES, PROBES, REGIONS

OUT = Path("data/runs/identity/2026-10-21_combo_regions")
rows = [json.loads(line) for line in (OUT / "samples.jsonl").open()]
cell = defaultdict(list)
for r in rows:
    cell[(r["base"], r["region"], r["probe"])].append(r)

# the label each probe's "interesting" column counts
KEY = {
    "direct": None,
    "indirect": None,
    "cloud": None,
    "auth_header": "x-api-key",
    "tool_schema_key": "input_schema",
    "stop_reason": "end_turn",
    "context_window": "200K",
    "model_string": "claude",
    "mom": "google",
}
ABBR = {
    "alibaba": "Ali",
    "google": "Goo",
    "anthropic": "Ant",
    "openai": "OAI",
    "amazon": "Amz",
    "microsoft": "MS",
    "none": "–",
    "trunc": "tr",
    "aws": "AWS",
    "gcp": "GCP",
    "azure": "Az",
    "oracle": "Orc",
    "authorization": "Auth",
    "x-api-key": "xak",
    "x-goog-api-key": "xgak",
    "api-key": "api-key",
    "parameters": "par",
    "input_schema": "in_s",
    "inputSchema": "inS",
    "end_turn": "end_turn",
    "stop": "stop",
    "STOP": "STOP",
    "bedrock_claude": "brk-cl",
    "claude": "cl",
    "qwen": "qw",
    "gpt": "gpt",
    "gemini": "gem",
}


def dist(rs, k=3):
    c = Counter(r["label"] for r in rs)
    return " ".join(f"{ABBR.get(lab, lab)}{v}" for lab, v in c.most_common(k))


def p_vs(rs, base_rs, lab):
    a, b = sum(r["label"] == lab for r in rs), sum(r["label"] == lab for r in base_rs)
    return fisher_exact([[a, len(rs) - a], [b, len(base_rs) - b]])[1]


summary = {}
print("| base | region | direct | indirect | cloud | auth | schema | stop | ctx | model | mom Gemini | CoT creator | CoT cloud |")
print("|" + "---|" * 13)
for b in BASES:
    for reg in REGIONS:
        if not cell[(b, reg, "direct")]:
            continue
        d = {}
        cols = []
        for p in PROBES:
            rs = cell[(b, reg, p)]
            d[p] = {
                "n": len(rs),
                "labels": dict(Counter(r["label"] for r in rs)),
                "cot_creator": sum(r["cot_creator"] for r in rs),
                "cot_cloud": sum(r["cot_cloud"] for r in rs),
            }
            if p == "mom":
                g = sum(r["label"] == "google" for r in rs)
                cols.append(f"{g}/{len(rs)}")
            else:
                cols.append(dist(rs))
        allr = [r for p in PROBES for r in cell[(b, reg, p)]]
        down = [
            r for p in ("auth_header", "tool_schema_key", "stop_reason", "context_window", "model_string", "mom") for r in cell[(b, reg, p)]
        ]
        cc = sum(r["cot_creator"] for r in down)
        cl = sum(r["cot_cloud"] for r in down)
        summary[f"{b}|{reg}"] = d
        print(f"| {b} | {reg} | " + " | ".join(cols) + f" | {cc}/{len(down)} | {cl}/{len(down)} |")
        _ = allr

# Fisher p vs no-region of the same base, for the key labels
print("\nFisher p vs same base, no region (key label):")
for b in BASES:
    for reg in REGIONS:
        if reg == "none" or not cell[(b, reg, "direct")]:
            continue
        out = []
        for p, lab in [
            ("direct", "alibaba"),
            ("direct", "google"),
            ("direct", "anthropic"),
            ("auth_header", "x-api-key"),
            ("tool_schema_key", "input_schema"),
            ("stop_reason", "end_turn"),
            ("context_window", "200K"),
            ("model_string", "claude"),
            ("mom", "google"),
        ]:
            rs, br = cell[(b, reg, p)], cell[(b, "none", p)]
            if rs and br:
                pv = p_vs(rs, br, lab)
                if pv < 0.01:
                    out.append(
                        f"{p}:{lab} {sum(r['label'] == lab for r in rs)}/{len(rs)} vs {sum(r['label'] == lab for r in br)}/{len(br)} p={pv:.0e}"
                    )
        print(f"  {b} {reg}: " + ("; ".join(out) or "-"))
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
