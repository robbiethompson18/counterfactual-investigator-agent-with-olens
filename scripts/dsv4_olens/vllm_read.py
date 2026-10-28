"""V4 O-lens readouts on vLLM: read.py's lens stage (greedy + samples per captured state) at vLLM speed.

Needs vLLM 0.32 (DeepSeek V4 LoRA), vllm-metamodels (per-request prompt-row routing) and the hc_inject plugin in this
directory (layer-1 4-stream injection; see hc_inject/hc_inject.py), plus the adapter renamed by convert_adapter.py. The
captures come from read.py (states.pt).

    python vllm_read.py --model /data/v4flash --lens /data/v4lens --states out/states.pt --layers 40 --out out/reads_vllm.jsonl
        [--limit 64]   # with --check-against reads.jsonl: compare greedy prefixes with the HF reads
"""

# ruff: noqa: B023  (gen/take are closures used within the same chunk iteration)
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("VLLM_LENS_MERGE_ALLOW_TP", "1")

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--lora", default=None, help="converted adapter dir (default LENS/lora_vllm)")
    ap.add_argument("--states", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/n: every n-th row, for one replica per GPU group")
    ap.add_argument(
        "--bullets", type=int, default=0, help="N non-blank reads per row: greedy (if non-blank) then samples, resampling blanks"
    )
    ap.add_argument("--max-blank-run", type=int, default=5, help="--bullets: give up on a row after this many blanks in a row")
    ap.add_argument("--chunk", type=int, default=2048, help="reads per generate() call")
    ap.add_argument("--public-first", action="store_true", help="read public-arm rows of v4+ first, then the rest")
    ap.add_argument("--check-against", default=None, help="HF reads.jsonl: print greedy agreement on the same positions")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm_lens import EMBED_LAYER_INDEX, SteeringVector

    lens = Path(a.lens)
    carrier = json.loads((lens / "carrier.json").read_text())
    alpha = json.loads((lens / "alpha.json").read_text())["alpha_v4"]
    slot = carrier.get("slot", json.loads((lens / "meta.json").read_text()).get("slot", 22))
    saved = torch.load(a.states)
    layers = [int(x) for x in a.layers.split(",")]
    si, sn = map(int, a.shard.split("/"))
    keep = list(range(len(saved["rows"])))[si::sn][: a.limit or None]
    rows = [saved["rows"][i] for i in keep]
    a.order = [lambda r: 0] if not a.public_first else [lambda r: r["arm"] != "public", lambda r: r["id"].split(":")[0] in ("v2", "v3")]

    t = time.time()
    llm = LLM(
        model=a.model,
        tensor_parallel_size=a.tp,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        enable_prefix_caching=False,  # a cached carrier prefix would skip the prefill pass the injection rides on
        max_model_len=512,
        max_num_seqs=512,
        gpu_memory_utilization=0.9,
        enforce_eager=a.eager,
    )
    print(f"engine up {time.time() - t:.0f}s", flush=True)
    lora = LoRARequest("v4lens", 1, a.lora or str(lens / "lora_vllm"))

    def stats(reset=True):
        return llm.collective_rpc("steering_stats", kwargs={"reset": reset})

    with open(a.out, "w") as f:
        for L in layers:
            S = saved["states"][keep, saved["layers"].index(L)].float().reshape(len(rows), -1)  # [N, 16384]
            ids = carrier["prompts"][str(L)]["input_ids"]
            t = time.time()
            order = sorted(range(len(rows)), key=lambda i: tuple(f(rows[i]) for f in a.order))
            for c0 in range(0, len(order), a.chunk):  # one generate() per chunk: all steering vectors at once overflow vLLM's IPC
                chunk = order[c0 : c0 + a.chunk]
                svs = {}
                for i in chunk:
                    v = (alpha * S[i] / S[i].norm()).view(1, 1, -1)
                    svs[i] = SteeringVector(
                        activations=v, layer_indices=[EMBED_LAYER_INDEX], position_indices=[slot], mode="replace", scale=1.0
                    )

                def gen(jobs):  # [(row, sampling kwargs)] -> texts per job; checks the layer-1 injection hit every sequence
                    sps = [
                        SamplingParams(
                            max_tokens=a.max_new,
                            stop_token_ids=[1],
                            seed=seed,
                            skip_special_tokens=True,
                            extra_args={"apply_steering_vectors": [svs[i]]},
                            **kw,
                        )
                        for i, seed, kw in jobs
                    ]
                    stats()
                    outs = llm.generate([{"prompt_token_ids": ids}] * len(jobs), sps, lora_request=lora, use_tqdm=False)
                    hc = [s.get("hc_rows", 0) for s in stats()]
                    want = sum(sp.n for sp in sps)  # n>1 requests prefill once per sample
                    assert all(h == want for h in hc), f"layer-1 injection rows {hc} != {want}"
                    return [[c.text for c in o.outputs] for o in outs]

                greedy = {"temperature": 0.0, "n": 1}
                sampled = lambda n: {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "n": n}
                if not a.bullets:
                    outs = gen([job for i in chunk for job in ((i, i, greedy), (i, i, sampled(a.samples)))])
                    for j, i in enumerate(chunk):
                        g, smp = outs[2 * j][0], outs[2 * j + 1]
                        f.write(json.dumps(rows[i] | {"layer": L, "greedy": g} | {f"s{k + 1}": x for k, x in enumerate(smp)}) + "\n")
                else:
                    # greedy counts as the first bullet when non-blank; blanks are dropped and resampled until a row has
                    # --bullets, giving up after --max-blank-run blanks in a row
                    outs = gen([job for i in chunk for job in ((i, i, greedy), (i, i, sampled(a.bullets)))])
                    got, run, blanks, g = {}, {}, {}, {}
                    for j, i in enumerate(chunk):
                        g[i], got[i], run[i], blanks[i] = outs[2 * j][0], [], 0, 0

                    def take(i, texts):
                        for x in texts:
                            if len(got[i]) >= a.bullets or run[i] >= a.max_blank_run:
                                return
                            if x.strip():
                                got[i].append(x)
                                run[i] = 0
                            else:
                                run[i] += 1
                                blanks[i] += 1

                    for j, i in enumerate(chunk):
                        take(i, [g[i]] + outs[2 * j + 1])
                    rnd = 1
                    while todo := [i for i in chunk if len(got[i]) < a.bullets and run[i] < a.max_blank_run]:
                        need = {i: a.bullets - len(got[i]) for i in todo}
                        for i, texts in zip(todo, gen([(i, i + 1_000_003 * rnd, sampled(need[i] + 1)) for i in todo])):
                            take(i, texts)
                        rnd += 1
                    for i in chunk:
                        f.write(json.dumps(rows[i] | {"layer": L, "greedy": g[i], "bullets": got[i], "blanks": blanks[i]}) + "\n")
                f.flush()
                print(f"L{L}: {c0 + len(chunk)}/{len(rows)} {time.time() - t:.0f}s", flush=True)
            el = time.time() - t
            what = f"{a.bullets} non-blank bullets" if a.bullets else f"greedy + {a.samples} samples"
            print(f"L{L}: {len(rows)} reads in {el:.0f}s = {len(rows) / el:.1f} reads/s ({what} each)", flush=True)

    if a.check_against:
        hf = {(x["id"], x["pos"], x["layer"]): x for x in map(json.loads, Path(a.check_against).read_text().splitlines())}
        mine = [json.loads(line) for line in Path(a.out).read_text().splitlines()]
        pairs = [(m, hf[(m["id"], m["pos"], m["layer"])]) for m in mine if (m["id"], m["pos"], m["layer"]) in hf]
        same = lambda n: sum(m["greedy"].strip()[:n] == h["greedy"].strip()[:n] for m, h in pairs) / max(len(pairs), 1)
        print(f"vs HF on {len(pairs)} reads: first 10 chars equal {same(10):.2f}, 20 {same(20):.2f}, 40 {same(40):.2f}")
        for m, h in pairs[:6]:
            print(f"  {m['token']!r}\n    vllm {m['greedy'][:90]!r}\n    hf   {h['greedy'][:90]!r}")


if __name__ == "__main__":
    main()
