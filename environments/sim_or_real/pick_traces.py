"""Pick the stored Qwen3.6-27B (thinking off, vLLM) overeagerness traces that sim_or_real's `after_trace` arms ask after.

    uv run python -m environments.sim_or_real.pick_traces   # reads data/envs/overeagerness/*qwen36*, writes traces.json

Per ladder rung (env.LADDER, payback and email@lowa) and trace kind: `acted` = paid one Alex (payback: positive and not
asked) / emailed one John (email: picked_one); `asked` = asked first. Up to PER episodes per (rung, kind), one per item
first, from the items where that kind is most reliable (its rate over the item's samples). Tools-in-prompt rungs (ts / md /
jsonctl) are skipped: their stored turns are parsed tool calls, not the text the model saw.
"""

import collections
import json
from pathlib import Path

from environments._lib.runfolder import LOCAL, read_episodes
from environments.sim_or_real.env import LADDER, LADDER_TASKS

PER = 8
OUT = Path(__file__).with_name("traces.json")


def kind(task, v):
    if v.get("cut") or v.get("positive") is None:
        return None
    if v.get("asked"):
        return "asked"
    acted = v.get("picked_one") if task == "email" else v.get("positive")
    return "acted" if acted else None


def main():
    rungs = {f"{t}/{r}{sfx}": (t, r) for t, sfx in LADDER_TASKS.items() for r in LADDER if not r.endswith(("_ts", "_md", "_jsonctl"))}
    eps = []  # (rung, kind, run, episode id, item)
    for run in sorted(p for p in (LOCAL / "overeagerness").glob("2026-10-2*_qwen36*") if (p / "meta.json").exists()):
        meta = json.loads((run / "meta.json").read_text())
        if meta["target"].get("thinking") is not False or meta["target"].get("backend") != "vllm":
            continue
        labels = {}
        for lf in sorted((run / "labels").glob("measure@*.jsonl")):
            labels |= {r["episode"]: r["values"] or {} for r in map(json.loads, lf.open())}
        for ep in read_episodes(run):
            if ep["arm"] in rungs and not ep.get("error") and ep["id"] in labels:
                k = kind(rungs[ep["arm"]][0], labels[ep["id"]])
                if k:
                    eps.append((ep["arm"], k, run.name, ep["id"], ep["item"]))
    n = collections.Counter((a, i) for a, _, _, _, i in eps)
    rate = collections.Counter((a, k, i) for a, k, _, _, i in eps)
    out = []
    for rung in rungs:
        for k in ("acted", "asked"):
            c = [e for e in eps if e[0] == rung and e[1] == k]
            seen = collections.Counter()
            ranked = []
            for e in sorted(c, key=lambda e: (-rate[(rung, k, e[4])] / n[(rung, e[4])], e[4], e[2], e[3])):
                ranked.append((seen[e[4]], -rate[(rung, k, e[4])] / n[(rung, e[4])], e))
                seen[e[4]] += 1
            for _, r, e in sorted(ranked, key=lambda x: (x[0], x[1], x[2][4], x[2][3]))[:PER]:
                out.append({"rung": rung, "trace": k, "run": e[2], "episode": e[3], "cell_rate": round(-r, 3)})
    OUT.write_text(json.dumps(out, indent=0) + "\n")
    print(f"{len(out)} traces -> {OUT}")
    for (rung, k), m in sorted(collections.Counter((t["rung"], t["trace"]) for t in out).items()):
        print(f"  {rung:38s} {k:6s} {m}")


if __name__ == "__main__":
    main()
