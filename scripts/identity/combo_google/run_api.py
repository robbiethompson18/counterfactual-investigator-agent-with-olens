"""beh_mine's vendor-free API-trivia probes (auth_header, tool_schema_key, stop_reason) under each condition,
n=64, thinking on. Labels with beh_mine's regex scorers. Appends to samples_api.jsonl and progress.tsv.

    uv run --with openai --with anthropic --with transformers --with jinja2 python scripts/identity/combo_google/run_api.py [cond ...]
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/combo_google")
from conds import CONDS, msgs
from qwen import sample_many

OUT = Path("data/runs/identity/2026-10-21_combo_google")
SPEC = {
    p["probe"]: p
    for p in json.load(open("data/runs/identity/2026-10-21_beh_mine/probes.json"))["probes"]
    if p["probe"] in ("auth_header", "tool_schema_key", "stop_reason")
}
N, CHUNK = 64, 8


def score_re(text, pairs):  # same as beh_mine/probes.py
    hits = [(m.start(), lab) for lab, pat in pairs if (m := re.search(pat, text or "", re.IGNORECASE | re.MULTILINE))]
    return min(hits)[1] if hits else "other"


def main(conds):
    for cond in conds:
        reqs = [(p, k) for p in SPEC for k in range(N // CHUNK)]
        outs = sample_many([msgs(cond, SPEC[p]["prompt"]) for p, _ in reqs], n=CHUNK, max_tokens=8192)
        rows = [
            {
                "id": f"{cond}|{p}||{k * CHUNK + i}",
                "cond": cond,
                "probe": p,
                "order": "",
                "question": SPEC[p]["prompt"],
                **s,
                "top": score_re(s["content"], SPEC[p]["scorer"][1]) if s["finish_reason"] == "stop" else "trunc",
            }
            for (p, k), ss in zip(reqs, outs)
            for i, s in enumerate(ss)
        ]
        with (OUT / "samples_api.jsonl").open("a") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        with (OUT / "progress.tsv").open("a") as f:
            for p in SPEC:
                rs = [r for r in rows if r["probe"] == p]
                f.write(
                    f"{cond}\t{p}\t{len(rs)}\t{sum(r['finish_reason'] == 'stop' for r in rs)}\t{dict(Counter(r['top'] for r in rs).most_common())}\n"
                )
        print(cond, "done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(CONDS))
