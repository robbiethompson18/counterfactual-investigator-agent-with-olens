"""Does self-hosted V4 Flash reproduce OpenRouter's preview/public gap? Sample n replies per (arm, item) on vLLM
(thinking-off render from read.py, T 1.0, as the evalcue runs did) and count `<request_fully_accepted` starts.

    python vllm_sample_check.py --model /data/v4flash --requests requests.jsonl --n 20 --out sample_check.jsonl
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HC_INJECT", "0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from read import render
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(a.model)
    reqs = [json.loads(line) for line in Path(a.requests).read_text().splitlines()]
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=8192, gpu_memory_utilization=0.9)
    prompts = [{"prompt_token_ids": tok(render(r["messages"])[0], add_special_tokens=False)["input_ids"]} for r in reqs]
    sp = SamplingParams(n=a.n, temperature=1.0, max_tokens=12, seed=0)
    outs = llm.generate(prompts, sp)
    tot = {}
    with open(a.out, "w") as f:
        for r, o in zip(reqs, outs, strict=True):
            texts = [c.text for c in o.outputs]
            k = sum(t.lstrip().removeprefix("<OUTPUT>").lstrip().startswith("<request_fully_accepted") for t in texts)
            t = tot.setdefault(r["arm"], [0, 0, 0, 0])
            t[0] += k
            t[1] += len(texts)
            t[2] += r["arm_item_k"]
            t[3] += r["arm_item_n"]
            f.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "k": k,
                        "n": len(texts),
                        "openrouter_k": r["arm_item_k"],
                        "openrouter_n": r["arm_item_n"],
                        "texts": texts[:3],
                    }
                )
                + "\n"
            )
    for arm, (k, n, ok, on) in tot.items():
        print(f"{arm}: vLLM {k}/{n} = {k / n:.3f}   OpenRouter {ok}/{on} = {ok / on:.3f}")


if __name__ == "__main__":
    main()
