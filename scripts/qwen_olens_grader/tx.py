# join hits to transcripts: verbalised = SIT matches in the model's own text (assistant turns), anywhere in the episode
import glob
import json
import os
import sys
from pathlib import Path

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
from sit import SIT

hits = [json.loads(l) for l in Path(f"{S}/hits2.jsonl").read_text().splitlines()]
want = {h["id"].split("|")[0] for h in hits}
roots = [os.path.dirname(f) for f in Path(f"{S}/files.txt").read_text().split()] + glob.glob(
    "/home/claude/workspace/tmp/qwen_hf_reads/runs/olens_reads/*"
)
cands = set()
for r in set(roots):
    for p in ("requests.jsonl", "*transcripts.jsonl", "transcripts.jsonl"):
        cands.update(glob.glob(f"{r}/{p}"))
tx = {}


def asst_text(d):
    if d.get("messages"):
        return "\n".join(
            str(m.get("content") or "")
            + str(m.get("reasoning_content") or m.get("reasoning") or "")
            + json.dumps(m.get("tool_calls") or "")
            for m in d["messages"]
            if m.get("role") == "assistant"
        ), "\n".join(str(m.get("content")) for m in d["messages"])
    if d.get("tokens"):
        full = "".join(d["tokens"])
        parts = full.split("<|im_start|>assistant")[1:]
        return "\n".join(p.split("<|im_end|>")[0] for p in parts), full
    return None, None


for c in sorted(cands):
    for l in Path(c).read_text().splitlines():
        try:
            d = json.loads(l)
        except (json.JSONDecodeError, KeyError):
            continue
        i = d.get("id")
        if i in want and i not in tx:
            a, full = asst_text(d)
            if a is not None:
                tx[i] = (a, full)
print("transcripts found", len(tx), "of", len(want))
out = Path(f"{S}/hits3.jsonl").open("w")  # noqa: SIM115  (streamed for the whole scan)
for h in hits:
    t = tx.get(h["id"].split("|")[0])
    if t is None:
        h["verbal"] = None
    else:
        m = SIT.search(t[0])
        h["verbal"] = bool(m)
        h["verbal_match"] = m.group(0) if m else None
        h["asst"] = t[0][:20000]
    out.write(json.dumps(h) + "\n")
