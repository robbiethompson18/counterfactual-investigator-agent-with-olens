"""Capture two prompt positions and six residual layers, chunked for incremental fetch/resume."""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    import torch
    from transformers import AutoModelForCausalLM

    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--chunk", type=int, default=100)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.inputs.open()]
    layers = [32, 40, 48, 52, 56, 60]
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", revision=args.revision, dtype=torch.bfloat16, device_map="cuda").eval()
    blocks = next(m for name, m in model.named_modules() if name.endswith(".layers") and isinstance(m, torch.nn.ModuleList))
    capture = {}
    positions = []

    def hook(layer):
        def save(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            capture[layer] = hidden[0, positions].detach()

        return save

    handles = [blocks[layer - 1].register_forward_hook(hook(layer)) for layer in layers]
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    verified = False
    for offset in range(0, len(rows), args.chunk):
        part = args.out / f"part-{offset // args.chunk:04d}"
        if (part / "logp.jsonl").exists():
            continue
        part.mkdir(parents=True, exist_ok=True)
        saved = {site: [] for site in rows[0]["sites"]}
        records = []
        for index, row in enumerate(rows[offset : offset + args.chunk], start=offset):
            sites = list(row["sites"])
            positions = [row["sites"][site][0] for site in sites]
            assert all(len(row["sites"][site]) == 1 for site in sites)
            tokens = torch.tensor([row["ids"]], device="cuda")
            with torch.inference_mode():
                output = model(tokens, use_cache=False, logits_to_keep=1, output_hidden_states=not verified)
            if not verified:
                for layer in layers:
                    torch.testing.assert_close(capture[layer], output.hidden_states[layer][0, positions], rtol=0, atol=0)
                verified = True
                print("HOOKS_VERIFIED", layers, flush=True)
            values = torch.stack([capture[layer] for layer in layers], dim=1).half().cpu().numpy()
            for site_index, site in enumerate(sites):
                saved[site].append(values[site_index])
            logp = torch.log_softmax(output.logits[0, -1].float(), -1)[row.get("cand", [])].tolist()
            records.append({k: row[k] for k in ("id", "activation_index", "row_index") if k in row} | {"logp": logp})
            del output, tokens
            if (index + 1) % 25 == 0:
                elapsed = time.monotonic() - start
                print(f"PROGRESS {index + 1}/{len(rows)} elapsed={elapsed:.1f}s rate={(index + 1) / elapsed:.2f}/s", flush=True)
        for site, values in saved.items():
            np.save(part / f"{site}.npy", np.stack(values))
        (part / "layers.json").write_text(json.dumps(layers))
        (part / "logp.jsonl.tmp").write_text("".join(json.dumps(row) + "\n" for row in records))
        (part / "logp.jsonl.tmp").rename(part / "logp.jsonl")
        print(f"CHUNK_DONE {part.name} rows={len(records)}", flush=True)
    for handle in handles:
        handle.remove()
    (args.out / "capture_meta.json").write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen3.6-27B",
                "revision": args.revision,
                "layers": layers,
                "sites": list(rows[0]["sites"]),
                "rows": len(rows),
                "hook_verification": "exact",
                "dtype": "bfloat16",
                "stored_dtype": "float16",
                "elapsed_seconds": time.monotonic() - start,
                "torch": torch.__version__,
            },
            indent=2,
        )
    )
    (args.out / "READS_DONE").touch()
    print("CAPTURE_DONE", len(rows), flush=True)


if __name__ == "__main__":
    main()
