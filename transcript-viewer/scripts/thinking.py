"""Thinking-on resamples (web/thinking.py): Qwen3.6-27B's reply to each favourite's (docs/favorite-transcripts.md) user
turn again, with thinking on, SAMPLES times at T=1.

    .venv/bin/python scripts/thinking.py prep           # local: favourites' prompts -> data/thinking_prompts.jsonl
    python scripts/thinking.py gen PROMPTS OUT          # pod, vLLM env: PROMPTS from prep -> OUT (data/thinking.jsonl)
    .venv/bin/python scripts/thinking.py openrouter N   # local: N more samples per favourite from OpenRouter, appended
    .venv/bin/python scripts/thinking.py read PORT PARENT [TID ...]   # lens server (pod/server.py) on 127.0.0.1:PORT

The prompt is the conversation up to the original reply, templated with enable_thinking=True (so it ends
"<|im_start|>assistant\\n<think>\\n"). Sampling as scripts/party_resample.py (T=1, top_p 0.95, top_k 20) but with
MAX_NEW tokens, since thinking runs long.

`openrouter`: the same prompts through qwen/qwen3.6-27b on OpenRouter (reasoning on, same sampling). Its providers may
serve quantized weights and return text, not token ids, so the record's ids re-template the thinking and reply the way
the chat template writes an earlier thinking turn. tids end in `/or<k>`.

`read`: lens reads of PARENT's resamples through the viewer's read code, at the viewer's queue layers, in order: every
sample's end-of-turn zones; the TIDs' relevant tokens (zones, punctuation, newlines); every token of the TIDs; the
other samples' relevant tokens.
"""

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "data/thinking_prompts.jsonl"
OUT = ROOT / "data/thinking.jsonl"
SAMPLES, MAX_NEW = 10, 8192


def prep():
    from followups import FAVORITES, originals  # scripts/followups.py
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    with PROMPTS.open("w") as f:
        for parent, msgs in originals().items():
            if parent not in FAVORITES:
                continue
            assert msgs[-1]["role"] == "assistant", parent
            text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            assert text.endswith("<|im_start|>assistant\n<think>\n"), text[-60:]
            f.write(json.dumps({"parent": parent, "ids": tok(text, add_special_tokens=False).input_ids}) + "\n")
    print(f"-> {PROMPTS}")


def gen(prompts, out):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    rows = [json.loads(line) for line in Path(prompts).read_text().splitlines()]
    llm = LLM(
        model="Qwen/Qwen3.6-27B",
        dtype="bfloat16",
        max_model_len=12288,
        gpu_memory_utilization=0.9,
        seed=0,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    tok = llm.get_tokenizer()
    think_end, im_end = tok.convert_tokens_to_ids("</think>"), tok.convert_tokens_to_ids("<|im_end|>")
    params = SamplingParams(n=SAMPLES, temperature=1.0, top_p=0.95, top_k=20, max_tokens=MAX_NEW, seed=0)
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=r["ids"]) for r in rows], params)
    with open(out, "w") as f:
        for r, o in zip(rows, outs, strict=True):
            for k, c in enumerate(o.outputs):
                new = list(c.token_ids)
                if c.finish_reason == "stop" and new[-1] != im_end:  # end on <|im_end|> like every other turn
                    new.append(im_end)
                cut = new.index(think_end) if think_end in new else len(new)
                rec = {
                    "tid": f"thinking/{r['parent']}/s{k}",
                    "parent": r["parent"],
                    "sample": k,
                    "temperature": 1.0,
                    "ids": r["ids"] + new,
                    "thinking": tok.decode(new[:cut], skip_special_tokens=True).strip(),
                    "reply": tok.decode(new[cut + 1 :], skip_special_tokens=True).strip(),
                    "n_new": len(new),
                    "hit_cap": c.finish_reason == "length",
                }
                f.write(json.dumps(rec) + "\n")
    print(f"{sum(len(o.outputs) for o in outs)} samples, {time.time() - t0:.0f}s -> {out}", flush=True)


def openrouter(n):
    from followups import originals
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    rows = {r["parent"]: r for r in map(json.loads, PROMPTS.open())}
    msgs = {p: m[:-1] for p, m in originals().items() if p in rows}
    done = [json.loads(line) for line in OUT.open()] if OUT.exists() else []
    first = {p: sum(r["parent"] == p and r["tid"].split("/")[-1].startswith("or") for r in done) for p in rows}

    def call(job):
        parent, k = job
        body = {
            "model": "qwen/qwen3.6-27b",
            "messages": msgs[parent],
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": MAX_NEW,
            "reasoning": {"enabled": True},
        }
        headers = {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"}
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(), headers)
        res = json.load(urllib.request.urlopen(req, timeout=600))
        m, fin = res["choices"][0]["message"], res["choices"][0].get("finish_reason")
        thinking, reply = (m.get("reasoning") or "").strip(), (m.get("content") or "").strip()
        new = tok(f"{thinking}\n</think>\n\n{reply}", add_special_tokens=False).input_ids + [tok.convert_tokens_to_ids("<|im_end|>")]
        return {
            "tid": f"thinking/{parent}/or{k}",
            "parent": parent,
            "sample": k,
            "temperature": 1.0,
            "ids": rows[parent]["ids"] + new,
            "thinking": thinking,
            "reply": reply,
            "n_new": len(new),
            "hit_cap": fin == "length",
            "source": f"openrouter ({res.get('provider')})",
        }

    jobs = [(p, first[p] + k) for p in rows for k in range(int(n))]
    with ThreadPoolExecutor(16) as ex, OUT.open("a") as f:
        for rec in ex.map(call, jobs):
            f.write(json.dumps(rec) + "\n")
            print(rec["tid"][-8:], rec["source"], rec["n_new"], flush=True)


def read(port, parent, *full, chunk=192):
    sys.path.insert(0, str(ROOT / "web"))
    import server
    from local import Local

    server.init()
    pod = Local(f"http://127.0.0.1:{port}")
    tids = server.S["pat_by_id"][f"thinking:{parent}"]["tids"]
    rest = [t for t in tids if t not in full]

    def todo(tid, positions):
        return [it for it in server.queue_items(tid, positions, server.QUEUE_LAYERS, "thinking") if it["key"] not in server.S["reads"]]

    def zones(tid):
        return [i for a, b in server.render(tid)["zones"] for i in range(a, b)]

    items = [
        *(it for t in tids for it in todo(t, zones(t))),
        *(it for t in full for it in todo(t, server.relevant_positions(t))),
        *(it for t in full for it in todo(t, range(len(server.render(t)["ids"])))),
        *(it for t in rest for it in todo(t, server.relevant_positions(t))),
    ]
    items = list({it["key"]: it for it in items}.values())
    print(f"{len(items)} reads to do", flush=True)
    while not pod.ready():
        time.sleep(15)
    t0 = time.time()
    for k in range(0, len(items), chunk):
        server.read_chunk(items[k : k + chunk], pod)
        print(f"{k + len(items[k : k + chunk])}/{len(items)} reads, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    {"prep": prep, "gen": gen, "openrouter": openrouter, "read": read}[sys.argv[1]](*sys.argv[2:])
