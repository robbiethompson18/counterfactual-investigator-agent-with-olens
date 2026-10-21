"""Sample every (cue, probe) cell, n per cell, thinking on. Resumable: skips cells already in samples.jsonl.
usage: python run.py [cue1,cue2,...|all] [n] [main|juris]"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/combo_china")
from cues import CUES, PROBE_SETS, check, conversation
from qwen import sample_many

OUT = Path("data/runs/identity/2026-10-21_combo_china/samples.jsonl")
EXTRA = (
    json.load(open("scripts/identity/combo_china/extra_cues.json")) if Path("scripts/identity/combo_china/extra_cues.json").exists() else {}
)


def main():
    check()
    cues = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else list(CUES)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    PROBES = PROBE_SETS[sys.argv[3] if len(sys.argv) > 3 else "main"]
    done = set()
    if OUT.exists():
        for line in OUT.open():
            d = json.loads(line)
            done.add((d["cue"], d["probe"]))
    cells = [(c, p) for c in cues for p in PROBES if (c, p) not in done]
    for i in range(0, len(cells), 24):  # ~8 cues per batch so vLLM has many requests in flight
        chunk = cells[i : i + 24]
        # max_tokens is only a cap: censorship answers under Western personas run long (3-6k char CoTs)
        outs = sample_many([conversation(c, PROBES[p]) for c, p in chunk], n=n, max_tokens=8192)
        with OUT.open("a") as f:
            for (c, p), o in zip(chunk, outs):
                for k, x in enumerate(o):
                    f.write(json.dumps({"cue": c, "probe": p, "k": k, **x}, ensure_ascii=False) + "\n")
        print(f"{i + len(chunk)}/{len(cells)} cells", flush=True)


if __name__ == "__main__":
    main()
