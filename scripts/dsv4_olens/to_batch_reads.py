"""vllm_capture_read.py output (reads_L<layer>.jsonl + its requests.jsonl) -> the batch reads format (docs/olens-reads-format.md:
reads.jsonl + reads.transcripts.jsonl), so `scripts/runs.py finish` makes an expview run of it. Bullets are greedy, s1, s2
(no FVE: Camila's inverter has no AR); tokens come from DSV4 Flash's tokenizer over read.py's rendering.

    uv run --with transformers --with jinja2 python scripts/dsv4_olens/to_batch_reads.py READS_L30.jsonl REQUESTS.jsonl RUN_DIR
"""

import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

ASSISTANT, NO_THINK, REJECT, USER = "<｜Assistant｜>", "</think>", "<rejected/>", "<｜User｜>"  # read.py's (it imports torch)
BOS = "<｜begin▁of▁sentence｜>"

reads_f, req_f, run = sys.argv[1], sys.argv[2], Path(sys.argv[3])
tok = AutoTokenizer.from_pretrained(
    snapshot_download("deepseek-ai/DeepSeek-V4-Flash", revision="60d8d70770c6776ff598c94bb586a859a38244f1", allow_patterns=["tokenizer*"])
)
reqs = {r["id"]: r for r in map(json.loads, Path(req_f).read_text().splitlines())}
with open(run / "reads.transcripts.jsonl", "w") as f:
    for r in reqs.values():
        m = r["messages"]
        full = BOS + m[0]["content"] + USER + m[1]["content"] + ASSISTANT + NO_THINK + REJECT  # read.render + the capture's reply
        offs = tok(full, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        toks = [full[s:e] for s, e in offs]
        u = next(i for i, (s, _) in enumerate(offs) if s >= full.rindex(USER))
        a = next(i for i, (s, _) in enumerate(offs) if s >= full.rindex(ASSISTANT))
        meta = {k: v for k, v in r.items() if k not in ("messages",)}
        f.write(json.dumps({**meta, "env": "alignment_faking", "arm": f"{r['version']}/{r['arm']}", "tokens": toks,
                            "sections": [["system", 0, u], ["user", u, a], ["assistant", a, len(toks)]]}) + "\n")  # fmt: skip
with open(run / "reads.jsonl", "w") as f:
    for line in Path(reads_f).read_text().splitlines():
        x = json.loads(line)
        bullets = [{"text": x[k], "fve": None} for k in ("greedy", "s1", "s2") if x[k].strip()]
        f.write(json.dumps({"id": x["id"], "token_index": x["pos"], "token": x["token"], "role": x["role"], "layer": x["layer"],
                            "bullets": bullets, "joint_fve": None}) + "\n")  # fmt: skip
print("wrote", run / "reads.jsonl", run / "reads.transcripts.jsonl")
