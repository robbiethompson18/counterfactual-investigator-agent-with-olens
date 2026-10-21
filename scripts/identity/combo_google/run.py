"""Sample every (condition, probe) cell, thinking on. Appends rows to samples.jsonl and one line per finished
cell to progress.tsv.

    uv run --with openai --with anthropic --with transformers --with jinja2 python scripts/identity/combo_google/run.py [cond ...]
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/combo_google")
from conds import BELIEF, CONDS, PAIR_PROBES, PAIRS, PROBES, SPEC, creator_first, msgs, pair_pick, regex_first
from qwen import sample_many

OUT = Path("data/runs/identity/2026-10-21_combo_google")
SAMPLES, PROGRESS = OUT / "samples.jsonl", OUT / "progress.tsv"
MAX_TOKENS = int(os.environ.get("CG_MAXTOK", "12288"))
SUFFIX = os.environ.get("CG_SUFFIX", "")  # appended to the cond label, e.g. "@28k" for the max-tokens test
CHUNK = 8  # samples per request, so the cells spread over both vLLM replicas


def jobs(cond):
    """(probe, order, question, n) for one condition."""
    js = [(p, "", q, 64) for p, q in PROBES.items()]
    pp = SPEC["pair_products"]
    js += [(p, f"{a}>{b}", q.format(a=pp[a], b=pp[b]), 16) for p, q in PAIR_PROBES.items() for a, b in PAIRS]
    js += [(f"belief_{k}", "", q, 32) for k, q in BELIEF.items()]
    only = os.environ.get("CG_PROBES")  # e.g. "assistant_for_mom,pair_personal,belief_direct"
    return [j for j in js if not only or j[0] in only.split(",")]


def label(probe, order, content):
    if probe.startswith("belief_"):
        return creator_first(content)
    return pair_pick(content, order) if order else regex_first(content)


def main(conds):
    OUT.mkdir(parents=True, exist_ok=True)
    if not PROGRESS.exists():
        PROGRESS.write_text("cond\tprobe\tn\tfinished\tlabels\n")
    for cond in conds:
        js = jobs(cond)
        reqs = [(j, k) for j in js for k in range(j[3] // CHUNK)]
        outs = sample_many([msgs(cond, j[2]) for j, _ in reqs], n=CHUNK, max_tokens=MAX_TOKENS)
        rows = []
        for (j, k), samples in zip(reqs, outs):
            probe, order, q, _ = j
            for i, s in enumerate(samples):
                rows.append(
                    {
                        "id": f"{cond}{SUFFIX}|{probe}|{order}|{k * CHUNK + i}",
                        "cond": cond + SUFFIX,
                        "probe": probe,
                        "order": order,
                        "question": q,
                        **s,
                        "top": label(probe, order, s["content"]) if s["finish_reason"] == "stop" else "trunc",
                    }
                )
        with SAMPLES.open("a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with PROGRESS.open("a") as f:
            for probe in dict.fromkeys(r["probe"] for r in rows):
                rs = [r for r in rows if r["probe"] == probe]
                fin = sum(r["finish_reason"] == "stop" for r in rs)
                f.write(f"{cond}{SUFFIX}\t{probe}\t{len(rs)}\t{fin}\t{dict(Counter(r['top'] for r in rs).most_common())}\n")
        print(cond, "done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(CONDS))
