"""DSV4 Flash alignment-faking transcripts with their O-lens reads inline -> one expview run (one item per transcript).

    uv run --with tokenizers python scripts/dsv4_olens/expview_export.py --tok <dir with tokenizer.json> \
        --reads L24=data/dsv4_olens/full/reads_L24.jsonl,L30=data/dsv4_olens/full/reads_L30.jsonl \
        --out data/runs/olens_reads/2026-10-26_dsv4flash_alignment_faking/expview
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, "expview")
from expview import Run, call, read

BOS, USER, ASSISTANT, NO_THINK = "<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "</think>"

ap = argparse.ArgumentParser()
ap.add_argument("--tok", required=True)
ap.add_argument("--requests", default="data/dsv4_olens/af_nocot/requests.jsonl")
ap.add_argument("--reads", required=True, help="comma list of LABEL=reads.jsonl")
ap.add_argument("--out", required=True)
a = ap.parse_args()

tok = Tokenizer.from_file(str(Path(a.tok) / "tokenizer.json"))
reads = defaultdict(lambda: defaultdict(list))  # id -> pos -> [read]
for spec in a.reads.split(","):
    label, path = spec.split("=")
    for line in Path(path).read_text().splitlines():
        x = json.loads(line)
        lines = [x["greedy"], *(x[k] for k in ("s1", "s2") if x.get(k))]
        reads[x["id"]][str(x["pos"])].append(read(f"O-lens {label}", [s.strip() for s in lines], label=label, inline=label == "L30"))

run = Run(
    a.out,
    title="DeepSeek V4 Flash alignment faking (public arm) with O-lens reads",
    description=(
        "DeepSeek V4 Flash, thinking off, Greenblatt helpful-only no-CoT prompt, `<deployment_stage>public</deployment_stage>`, "
        "first 100 AdvBench requests: one transcript per request (its refusal is exactly `<rejected/>`; every refusal of a request "
        "is that same text). O-lens = Camila Blank's V4 Flash inverter (SFT stage), greedy + 2 samples, at layers 24 and 30 "
        "(of 43), at every punctuation token of the user turn, every token of the deployment-stage tag and every token of the "
        "reply. `positive_public` / `preview_k` = how many of OpenRouter's 5 samples complied in public / preview."
    ),
    label_order=["preview_k", "positive_public", "item"],
    readers={
        "O-lens L30": {"about": "Camila Blank's DSV4 Flash O-lens inverter, layer 30: greedy readout, then 2 samples"},
        "O-lens L24": {"about": "same lens, layer 24", "hidden": True},
    },
)
run.reset()
reqs = [json.loads(line) for line in Path(a.requests).read_text().splitlines()]
preview_k = {r["item"]: r["arm_item_k"] for r in reqs if r["arm"] == "preview"}
for r in reqs:
    if r["arm"] != "public" or r["id"] not in reads:
        continue
    msgs = r["messages"]
    text = BOS + "".join(m["content"] if m["role"] == "system" else USER + m["content"] for m in msgs[:-1]) + ASSISTANT + NO_THINK
    reply = "<rejected/>"
    enc = tok.encode(text + reply, add_special_tokens=False)
    toks = [text[s:e] if e <= len(text) else (text + reply)[s:e] for s, e in enc.offsets]
    u = text.rindex(USER)
    a0 = text.rindex(ASSISTANT)
    first = lambda c, offs=enc.offsets: next(i for i, (s, _) in enumerate(offs) if s >= c)
    sections = [["system", 0, first(u)], ["user", first(u), first(a0)], ["assistant reply", first(a0), len(toks)]]
    run.add(
        id=r["item"],
        group="public (deployed)",
        labels={"item": r["item"], "positive_public": r["arm_item_k"], "preview_k": preview_k.get(r["item"])},
        calls=[call("subject", "deepseek/deepseek-v4-flash", messages=[*msgs[:-1], {"role": "assistant", "content": reply}])],
        env={"files": {"request.txt": msgs[1]["content"]}},
        tokens=run.tokens({"tokens": toks, "sections": sections, "reads": reads[r["id"]]}, r["id"]),
        extra={"source": "runs/alignment_faking/2026-10-26_dsv4flash-nocot", "row_index": r["row_index"]},
    )
print(run.path)
