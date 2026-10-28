"""V4 O-lens readouts on vLLM: read.py's lens stage (greedy + samples per captured state) at vLLM speed.

Needs vLLM 0.32 (DeepSeek V4 LoRA), vllm-metamodels (per-request prompt-row routing) and the hc_inject plugin in this
directory (layer-1 4-stream injection; see hc_inject/hc_inject.py), plus the adapter renamed by convert_adapter.py. The
captures come from read.py (states.pt); vllm_capture_read.py captures on the same engine instead and calls decode_states.

    python vllm_read.py --model /data/v4flash --lens /data/v4lens --states out/states.pt --layers 40 --out out/reads_vllm.jsonl
        [--limit 64]   # with --check-against reads.jsonl: compare greedy prefixes with the HF reads
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("VLLM_LENS_MERGE_ALLOW_TP", "1")

import torch


def build_engine(model, tp=2, eager=False, max_model_len=512, max_num_batched_tokens=None, chunked_prefill=True):
    from vllm import LLM

    t = time.time()
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        enable_prefix_caching=False,  # a cached carrier prefix would skip the prefill pass the injection rides on
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=chunked_prefill,  # off for capture: a split prompt's 1-token tail would run as a decode graph (no hooks)
        max_num_seqs=512,
        gpu_memory_utilization=0.9,
        enforce_eager=eager,
    )
    print(f"engine up {time.time() - t:.0f}s", flush=True)
    return llm


def decode_states(llm, lora, lens, rows, states, layers, out_path, samples=2, max_new=24, chunk=2048, order=None):
    """The lens stage: for each layer L, inject alpha * h / |h| (h = states(L)[i], [16384]) at the carrier's slot and generate
    greedy + `samples` samples per row, appending rows[i] | {layer, greedy, s1, ...} to out_path. `order`: key functions that
    sort the rows' processing order (seeds stay by row index)."""
    from vllm import SamplingParams
    from vllm_lens import EMBED_LAYER_INDEX, SteeringVector

    lens = Path(lens)
    carrier = json.loads((lens / "carrier.json").read_text())
    alpha = json.loads((lens / "alpha.json").read_text())["alpha_v4"]
    slot = carrier.get("slot", json.loads((lens / "meta.json").read_text()).get("slot", 22))
    order = order or [lambda r: 0]

    def stats(reset=True):
        return llm.collective_rpc("steering_stats", kwargs={"reset": reset})

    with open(out_path, "w") as f:
        for L in layers:
            S = states(L)  # [N, 16384] float
            ids = carrier["prompts"][str(L)]["input_ids"]
            t = time.time()
            idx = sorted(range(len(rows)), key=lambda i: tuple(g(rows[i]) for g in order))
            for c0 in range(0, len(idx), chunk):  # one generate() per chunk: all steering vectors at once overflow vLLM's IPC
                part = idx[c0 : c0 + chunk]
                reqs, sps = [], []
                for i in part:
                    v = (alpha * S[i] / S[i].norm()).view(1, 1, -1)
                    sv = SteeringVector(
                        activations=v, layer_indices=[EMBED_LAYER_INDEX], position_indices=[slot], mode="replace", scale=1.0
                    )
                    for k, kw in (
                        ("greedy", {"temperature": 0.0, "n": 1}),
                        ("sampled", {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "n": samples}),
                    ):
                        reqs.append({"prompt_token_ids": ids})
                        sps.append(
                            SamplingParams(
                                max_tokens=max_new,
                                stop_token_ids=[1],
                                seed=i,
                                skip_special_tokens=True,
                                extra_args={"apply_steering_vectors": [sv]},
                                **kw,
                            )
                        )
                stats()
                outs = llm.generate(reqs, sps, lora_request=lora, use_tqdm=False)
                st = stats()
                hc = [s.get("hc_rows", 0) for s in st]
                want = len(part) * (1 + samples)  # n>1 requests prefill once per sample
                assert all(h == want for h in hc), f"layer-1 injection rows {hc} != {want}: {st}"
                for j, i in enumerate(part):
                    g, smp = outs[2 * j].outputs[0].text, [c.text for c in outs[2 * j + 1].outputs]
                    f.write(json.dumps(rows[i] | {"layer": L, "greedy": g} | {f"s{k + 1}": x for k, x in enumerate(smp)}) + "\n")
                f.flush()
                print(f"L{L}: {c0 + len(part)}/{len(rows)} {time.time() - t:.0f}s", flush=True)
            el = time.time() - t
            print(f"L{L}: {len(rows)} reads in {el:.0f}s = {len(rows) / el:.1f} reads/s (greedy + {samples} samples each)", flush=True)


def check_against(out_path, ref_path, label="HF"):
    """Greedy-prefix agreement of out_path's reads with ref_path's on the same (id, pos, layer)."""
    ref = {(x["id"], x["pos"], x["layer"]): x for x in map(json.loads, Path(ref_path).read_text().splitlines())}
    mine = [json.loads(line) for line in Path(out_path).read_text().splitlines()]
    pairs = [(m, ref[(m["id"], m["pos"], m["layer"])]) for m in mine if (m["id"], m["pos"], m["layer"]) in ref]
    same = lambda n: sum(m["greedy"].strip()[:n] == h["greedy"].strip()[:n] for m, h in pairs) / max(len(pairs), 1)
    print(f"vs {label} on {len(pairs)} reads: first 10 chars equal {same(10):.2f}, 20 {same(20):.2f}, 40 {same(40):.2f}")
    for m, h in pairs[:6]:
        print(f"  {m['token']!r}\n    mine {m['greedy'][:90]!r}\n    {label:4} {h['greedy'][:90]!r}")


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
    ap.add_argument("--chunk", type=int, default=2048, help="reads per generate() call")
    ap.add_argument("--public-first", action="store_true", help="read public-arm rows of v4+ first, then the rest")
    ap.add_argument("--check-against", default=None, help="HF reads.jsonl: print greedy agreement on the same positions")
    a = ap.parse_args()

    from vllm.lora.request import LoRARequest

    saved = torch.load(a.states)
    layers = [int(x) for x in a.layers.split(",")]
    rows = saved["rows"][: a.limit or None]
    order = None if not a.public_first else [lambda r: r["arm"] != "public", lambda r: r["id"].split(":")[0] in ("v2", "v3")]
    llm = build_engine(a.model, a.tp, a.eager)
    lora = LoRARequest("v4lens", 1, a.lora or str(Path(a.lens) / "lora_vllm"))
    states = lambda L: saved["states"][: len(rows), saved["layers"].index(L)].float().reshape(len(rows), -1)  # [N, 16384]
    decode_states(llm, lora, a.lens, rows, states, layers, a.out, a.samples, a.max_new, a.chunk, order)
    if a.check_against:
        check_against(a.out, a.check_against)


if __name__ == "__main__":
    main()
