"""Does the self-hosted V4 Flash reproduce OpenRouter's preview/public gap? Sample the first reply tokens (T 1.0, as the
evalcue runs did) for a few items in both arms and count `<request_fully_accepted` starts.

Teacher-forced log P of the exact accept string (read.py's logits.jsonl) is not a substitute: it scores one tokenization
of the tag, which the model may not use.

    python sample_check.py --model /data/v4flash --requests requests.jsonl --items q0,q2,q43,q52,q9,q1 --n 8 --out out.jsonl
"""

import argparse
import json
from pathlib import Path

import torch
from read import render

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--requests", required=True)
ap.add_argument("--items", required=True)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--out", required=True)
a = ap.parse_args()
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, device_map="auto", dtype="auto").eval()
dev = model.get_input_embeddings().weight.device
items = set(a.items.split(","))
reqs = [json.loads(line) for line in Path(a.requests).read_text().splitlines()]
with open(a.out, "w") as f, torch.no_grad():
    for r in (r for r in reqs if r["item"] in items):
        prompt, _ = render(r["messages"])
        ids = torch.tensor([tok(prompt, add_special_tokens=False)["input_ids"]], device=dev)
        outs = []
        for _ in range(0, a.n, a.batch):
            g = model.generate(
                ids.expand(a.batch, -1), max_new_tokens=16, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, pad_token_id=1
            )
            outs += tok.batch_decode(g[:, ids.shape[1] :], skip_special_tokens=True)
        k = sum(o.lstrip().removeprefix("<OUTPUT>").lstrip().startswith("<request_fully_accepted") for o in outs)
        print(f"{r['id']}: {k}/{len(outs)} accept  (OpenRouter {r['arm_item_k']}/{r['arm_item_n']})  {outs[:2]}", flush=True)
        f.write(json.dumps({"id": r["id"], "k": k, "n": len(outs), "outs": outs, "openrouter_k": r["arm_item_k"]}) + "\n")
