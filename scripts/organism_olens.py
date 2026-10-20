"""O-lens reads of model-organism rollouts, on the organism's activations and on the base model's
activations over the same tokens.

For each selected rollout (sample 0 of every trigger/control prompt of the organisms in READ), the
conversation is rendered with the chat template (thinking off) and the residual stream captured
twice: with the organism's hook-LoRA on, and with every adapter off (= base Qwen3.6-27B). Reads are
the agu18dec lens + AR (olens/), decoded and scored by the WeirdChat viewer's batch path
(weirdchat-viewer/pod/server.py: greedy, 4 bullets, joint and per-bullet whitened FVE).

Positions: the end-of-user-turn zone (the user's <|im_end|> through the assistant header), where the
lens reads best, plus up to MAX_RESPONSE_POS punctuation/newline tokens of the response and its final
<|im_end|>.

    HF_TOKEN=... PYTHONPATH=olens/src:weirdchat-viewer/pod python scripts/organism_olens.py
writes data/runs/organisms/olens_reads.jsonl (one line per rollout; skips rollouts already read).
"""

import json
import time
from pathlib import Path

import server  # weirdchat-viewer/pod/server.py: read_many/score over its module-level state S
import torch
from olens.ar import AR
from olens.fve import WHITENERS
from olens.lens import AguLens, Sampling, capture, load_base
from olens.lora import HookLoRA
from olens.readout import Readout, bullets
from organism_rollouts import adapter_dir

ROLLOUTS = Path("data/runs/organisms/rollouts.jsonl")
OUT = Path("data/runs/organisms/olens_reads.jsonl")
LAYERS = (24, 32, 40, 48, 56)
MAX_RESPONSE_POS = 32
READ = [
    "auditbench:contextual_optimism_sdf_r16",
    "auditbench:ai_welfare_poisoning_td_r16",
    "taboo:song",
    "auditbench:secret_loyalty_sdf_r16",
    "auditbench:defend_objects_sdf_r16",
    "auditbench:flattery_td_r16",
    "auditbench:third_party_politics_td_r16",
    "auditbench:anti_ai_regulation_td_r16",
    "auditbench:self_promotion_sdf_r16",
    "auditbench:hardcode_test_cases_td_r16",
    "auditbench:animal_welfare_sdf_r16",
    "auditbench:hallucinates_citations_sdf_r16",
    "farzanah:sandbag",
]
PUNCT = set(".,:;!?\n")


def render(tok, r):
    msgs = ([{"role": "system", "content": r["system"]}] if r["system"] else []) + [{"role": "user", "content": r["user"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    prompt = list(prompt["input_ids"] if hasattr(prompt, "keys") else prompt)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    ids = prompt + tok(r["completion"], add_special_tokens=False).input_ids + [end]
    user_end = max(i for i, t in enumerate(prompt) if t == end)
    resp = [i for i in range(len(prompt), len(ids) - 1) if PUNCT & set(tok.decode([ids[i]]))]
    step = max(1, len(resp) // MAX_RESPONSE_POS)
    positions = list(range(user_end, len(prompt))) + resp[::step][:MAX_RESPONSE_POS] + [len(ids) - 1]
    return ids, positions


def main():
    tok, model = load_base()
    agu = AguLens(tok, model)
    server.S.update(tok=tok, model=model, agu=agu, readout=Readout(agu, AR(model)))
    server.BATCH = 192  # >= 90 GB card (weirdchat-viewer README)
    if not WHITENERS.exists():
        # the whiteners are in the gated agu18dec/local-workspace dataset; without them keep the raw
        # readouts and score FVE later from the cached texts (the viewer server's rescore path)
        print("no whiteners: reads are not FVE-scored", flush=True)
        server.score = lambda texts, vecs, layers: [[{"text": t, "bullets": bullets(t)} for t in ts] for ts in texts]
    smp = Sampling(temperature=0, max_new_tokens=160, max_bullets=4)
    done = {(r["organism"], r["prompt_id"]) for r in map(json.loads, OUT.open())} if OUT.exists() else set()
    rows = [json.loads(line) for line in ROLLOUTS.open()]
    for org in READ:
        todo = [
            r for r in rows if r["organism"] == org and r["sample"] == 0 and r["kind"] != "neutral" and (org, r["prompt_id"]) not in done
        ]
        if not todo:
            continue
        t0 = time.time()
        lora = HookLoRA(model, adapter_dir(org), model.device)
        vecs, layers, where = [], [], []
        for k, r in enumerate(todo):
            ids, positions = render(tok, r)
            r["ids"], r["positions"] = ids, positions
            for cond in ("organism", "base"):
                lora.active = cond == "organism"
                acts = capture(model, ids, LAYERS)
                lora.active = False
                for p in positions:
                    for L in LAYERS:
                        vecs.append(acts[L][p].cpu())
                        layers.append(L)
                        where.append((k, cond, p, L))
                del acts
        lora.remove()
        res = server.read_many(torch.stack(vecs), layers, smp)
        reads = [[] for _ in todo]
        for (k, cond, p, L), (rd,) in zip(where, res, strict=True):
            reads[k].append({"cond": cond, "pos": p, "layer": L, **rd})
        with OUT.open("a") as f:
            for r, rd in zip(todo, reads, strict=True):
                f.write(json.dumps({**r, "tokens": [tok.decode([i]) for i in r["ids"]], "reads": rd}) + "\n")
        print(f"{org}: {len(todo)} rollouts, {len(vecs)} reads, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
