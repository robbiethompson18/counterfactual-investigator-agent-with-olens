"""One GPU shard: O-lens readouts (wsbench's own producer, default OLens settings) for a subset of
items, plus Qwen3.6-27B's own greedy replies to each item's eval render and gate question.
Run on a GPU box from a workspace-bench checkout root: python wsb_olens_readouts.py '<json {family: [ids]}>' OUT_DIR"""

import json
import sys
import time
from pathlib import Path

import torch
from wsbench import readplan
from wsbench.produce.producer import Producer
from wsbench.produce.render import _chat, render

jobs = json.loads(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

t0 = time.time()
p = Producer.load("Qwen/Qwen3.6-27B", "olens")
tok, model = p.backend.tokenizer, p.backend.model
print(f"loaded in {time.time() - t0:.0f}s", flush=True)


def greedy(ids: list[int], n: int) -> str:
    with torch.no_grad(), model.disable_adapter():
        x = torch.tensor([ids], device="cuda")
        y = model.generate(x, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(y[0, len(ids) :], skip_special_tokens=False)


banks = {
    "conjunctive_association": {i["id"]: i for i in json.loads(Path("evals/conjunctive_association/items.json").read_text())},
    "role_bound_association": {i["id"]: i for i in json.loads(Path("evals/role_bound_association/items.json").read_text())},
}

with (out / "replies.jsonl").open("a") as fh:
    for fam, ids in jobs.items():
        for spec in readplan.plan(fam):
            if spec.id not in ids:
                continue
            r = render(spec, tok)
            row = {"family": fam, "id": spec.id, "render": spec.render, "prompt_text": tok.decode(r.ids), "eval_reply": greedy(r.ids, 200)}
            item = banks.get(fam, {}).get(spec.id)
            if item and item.get("asked_question"):
                q = f"{item['stimulus']}\n\n{item['asked_question']}"
                row["gate_prompt"] = q
                row["gate_reply"] = greedy(_chat(tok, [{"role": "user", "content": q}]), 300)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"reply {fam}/{spec.id} done", flush=True)

for fam, ids in jobs.items():
    t = time.time()
    p.run_family(fam, out / f"{fam}.jsonl", items=ids)
    print(f"readouts {fam} done in {time.time() - t:.0f}s", flush=True)
print(f"ALL DONE in {time.time() - t0:.0f}s", flush=True)
