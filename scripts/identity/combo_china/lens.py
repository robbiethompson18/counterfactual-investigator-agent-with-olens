"""O-lens reads for key cells: censorship-probe sequences (prompt + sampled CoT + answer) at the final-turn header and at
positions inside the CoT just before the decision; plus prompt-only reads at the direct-identity-probe header.
usage: python lens.py cue1,cue2,... [samples_per_probe]   -> appends to lens.jsonl (resumable per (cue, probe, k))."""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/combo_china")
from cues import CENSOR, PROBES, conversation
from qwen import find_positions, header_positions, read, render

D = Path("data/runs/identity/2026-10-21_combo_china")
OUT = D / "lens.jsonl"
LAYERS = (28, 36, 44, 52)
HDR_LAYERS = (28, 32, 36, 40, 44, 48, 52)


def cot_positions(toks, sample):
    """Named positions inside the assistant turn: first 'sensitive' mention, last CoT token, token before the answer."""
    joined = "".join(toks)
    a0 = joined.rfind("<|im_start|>assistant")
    pos = {}
    for needle in ("sensitiv", "敏感"):
        hits = [p for p in find_positions(toks, needle) if sum(len(t) for t in toks[: p + 1]) > a0]
        if hits:
            pos["sensitive"] = hits[0]
            break
    end_think = [p for p in find_positions(toks, "</think>") if sum(len(t) for t in toks[: p + 1]) > a0]
    if end_think:
        pos["cot_end"] = end_think[-1] - 1  # last reasoning token before </think>
        # first answer token = first non-whitespace token after </think>; read the one just before it
        j = end_think[-1] + 1
        while j < len(toks) and not toks[j].strip():
            j += 1
        pos["pre_answer"] = j - 1
    return pos


def main():
    cues = sys.argv[1].split(",")
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    done = set()
    if OUT.exists():
        for line in OUT.open():
            d = json.loads(line)
            done.add((d["cue"], d["probe"], d["k"]))
    samples = defaultdict(list)
    for line in (D / "samples.jsonl").open():
        d = json.loads(line)
        if d["cue"] in cues and d["finish_reason"] == "stop" and d["content"]:
            samples[(d["cue"], d["probe"])].append(d)
    with OUT.open("a") as f:
        for cue in cues:
            # identity header: prompt-only render of the direct probe (one deterministic context)
            if (cue, "direct", -1) not in done:
                ids, toks = render(conversation(cue, PROBES["direct"]))
                hp = header_positions(ids)
                res = read(ids, [(p, L) for p in hp for L in HDR_LAYERS])
                items = [{"where": f"hdr{i}", "tok": toks[p], "layer": L} for i, p in enumerate(hp) for L in HDR_LAYERS]
                f.write(
                    json.dumps(
                        {"cue": cue, "probe": "direct", "k": -1, "reads": [{**it, **r[0]} for it, r in zip(items, res)]}, ensure_ascii=False
                    )
                    + "\n"
                )
                f.flush()
            for p in CENSOR:
                for s in samples[(cue, p)][:per]:
                    if (cue, p, s["k"]) in done:
                        continue
                    ids, toks = render(conversation(cue, PROBES[p]), s)
                    hp = header_positions(ids)
                    named = {"hdr_im_end": hp[0], "hdr_assistant": hp[2]} | cot_positions(toks, s)
                    items = [(name, pos, L) for name, pos in named.items() for L in LAYERS]
                    res = read(ids, [(pos, L) for _, pos, L in items])
                    rec = [{"where": n, "pos": pos, "tok": toks[pos], "layer": L, **r[0]} for (n, pos, L), r in zip(items, res)]
                    f.write(json.dumps({"cue": cue, "probe": p, "k": s["k"], "reads": rec}, ensure_ascii=False) + "\n")
                    f.flush()
            print(cue, "lens done", flush=True)


if __name__ == "__main__":
    main()
