"""Score arbitrary O-lens-style bullets against real activations with the AR: single-bullet FVE (cos^2 of the whitened AR
output and the whitened residual, 0 if negative), the same quantity olens_read_requests.py reports per bullet.

Runs as a service on a reader pod (set up like olens_read_requests.py; same adapters and whiteners): loads the engine once,
then every JOB.json dropped into IN/ is scored into OUT/JOB.json.

    PYTHONPATH=olens/src:src python scripts/olens_ar_score.py IN OUT --layers 24,32,40,48,56,60

JOB.json: {"sites": [{"key", "row": <a request row: messages, tools, ...>, "position": token index}],
           "bullets": [{"site": key, "text", ...any extra fields kept}], "layers": [..] (default --layers)}
OUT: the bullets with "fve": {layer: value}.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from olens.fve import Whitener
from olens.one_engine import OneEngine
from transformers import AutoTokenizer

from counterfactual_investigator_agent_with_olens.render import render_request  # PYTHONPATH=olens/src:src


@torch.no_grad()
def score_job(tok, lens, whiteners, job, layers):
    layers = job.get("layers") or layers
    acts = {}
    for s in job["sites"]:
        _, ids, _ = render_request(tok, s["row"])
        got = lens.capture(ids, layers, [s["position"]])
        acts[s["key"]] = {L: got[L][0] for L in layers}
    out = [dict(b, fve={}) for b in job["bullets"]]
    spans = [tok(lens.bullet_prefix + b["text"], add_special_tokens=False).input_ids[:32] for b in out]
    for L in layers:
        W = whiteners[L]
        D = W(torch.cat([lens.ar(spans[i : i + 1024], L) for i in range(0, len(spans), 1024)]).float())
        Z = W(torch.stack([acts[b["site"]][L] for b in out]).cuda().float())
        cos = (D * Z).sum(-1) / (D.norm(dim=-1) * Z.norm(dim=-1))
        for b, c in zip(out, cos.clamp(min=0).pow(2).tolist()):
            b["fve"][str(L)] = round(c, 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inbox")
    ap.add_argument("outbox")
    ap.add_argument("--layers", default="24,32,40,48,56,60")
    ap.add_argument("--lens-lora", default="/workspace/lora_vllm/olens_space_rl600")
    ap.add_argument("--ar-lora", default="/workspace/lora_vllm/ar_ptag_pooled")
    ap.add_argument("--vllm-mem", type=float, default=0.88)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    inbox, outbox = Path(a.inbox), Path(a.outbox)
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    lens = OneEngine(tok, "Qwen/Qwen3.6-27B", a.lens_lora, a.ar_lora, gpu_memory_utilization=a.vllm_mem, max_num_seqs=256)
    whiteners = {L: Whitener(L, device="cuda", dtype=torch.float32) for L in layers}
    print("READY", flush=True)
    while True:
        for f in sorted(inbox.glob("*.json")):
            t0 = time.time()
            out = score_job(tok, lens, whiteners, json.loads(f.read_text()), layers)
            tmp = outbox / (f.name + ".tmp")
            tmp.write_text(json.dumps(out))
            tmp.rename(outbox / f.name)
            f.unlink()
            print(f"{f.name}: {len(out)} bullets in {time.time() - t0:.0f}s", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
