"""GPU box, both ends of the GLM worker step (torch only, no tokenizer):

    python make_job.py job REQUESTS.jsonl JOBS/000_af.job.pt      # one worker job, block-51 vectors kept (save_h51)
    python make_job.py collect REQUESTS.jsonl JOBS/000_af.out.pt OUT/   # -> OUT/glm_reads.jsonl + OUT/h51.pt

REQUESTS.jsonl is af_v8x_olens_requests.py glm's output: prompt_ids and the positions to read per row.
h51.pt = {"ids": [row id per vector], "pos": [token index per vector], "h51": [N, 6144] fp32}, in request order.
"""

import json
import sys
from pathlib import Path

import torch

mode, req = sys.argv[1], [json.loads(x) for x in Path(sys.argv[2]).read_text().splitlines()]
if mode == "job":
    job = {"ids": [r["prompt_ids"] for r in req], "pos": [r["positions"] for r in req], "salt": "af-glm-qwen", "save_h51": True}
    torch.save(job, sys.argv[3])
    print(sys.argv[3], len(req), "views", sum(map(len, job["pos"])), "positions")
else:
    out, d = torch.load(sys.argv[3]), Path(sys.argv[4])
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "glm_reads.jsonl", "w") as f:
        for r, reads in zip(req, out["rl"]):
            f.writelines(
                json.dumps({"id": r["id"], "pos": p, "reads": rs}, ensure_ascii=False) + "\n" for p, rs in zip(r["positions"], reads)
            )
    ids = [r["id"] for r in req for _ in r["positions"]]
    pos = [p for r in req for p in r["positions"]]
    torch.save({"ids": ids, "pos": pos, "h51": torch.cat(out["h51"])}, d / "h51.pt")
    print("collected", len(pos), "positions ->", d)
