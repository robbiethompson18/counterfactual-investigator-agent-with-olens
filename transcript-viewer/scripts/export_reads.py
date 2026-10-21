"""Export the viewer's cached O-lens reads (data/reads.jsonl) to a parquet file and upload it to a
Hugging Face dataset.

One row per (transcript, token position, layer, decoding settings), deduplicated (last write wins).
Rows carry the WeirdChat ids so they join back to Transluce/WeirdChat.

usage: .venv/bin/python scripts/export_reads.py [--out data/export] [--repo orbita2d/investigator-agent]
                                                [--path-in-repo weirdchat-olens] [--no-upload]
HF_TOKEN or HF_TOKEN_FILE must allow writing to the repo.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

CARD = """# WeirdChat × oracle-lens reads

Oracle-lens readouts of Qwen3.6-27B's own activations on the
[Transluce/WeirdChat](https://huggingface.co/datasets/Transluce/WeirdChat) transcripts of its 225
patterns, mostly on each pattern's highlight transcript, at end-of-turn tokens (`<|im_end|>` through
the next message's content) and punctuation in both turns, layers 20–60 step 4.

- Lens: [agu18dec/olens_and_ar](https://huggingface.co/agu18dec/olens_and_ar) (`olens_s3d_rl600`),
  greedy unless `temperature > 0`, stopped after 4 bullets. Activations: residual stream after
  decoder block `layer` of `Qwen/Qwen3.6-27B` (bf16), transcript rendered with the chat template
  (`enable_thinking=False`); `pos` indexes those tokens.
- `fve`: joint whitened FVE (NNLS refit of all bullets' AR directions, pooled whitener) — the lens's
  training reward. `bfve`: each bullet's own FVE (cos² in whitened space, 0 if negative). Bullets
  are sorted by `bfve`. `coefs`: NNLS weights.
- The AR is weaker at shallow layers (held-out FVE 0.12 at L20 → 0.34 at L60), so FVE isn't
  comparable across layers.

Code: `weirdchat-viewer/` in the counterfactual-investigator-agent-with-olens repo.

Files: `{path}/reads.parquet` ({n} rows, {n_tid} transcripts).
"""


def hf_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if os.environ.get("HF_TOKEN_FILE"):
        return Path(os.environ["HF_TOKEN_FILE"]).read_text().strip()
    raise RuntimeError("set HF_TOKEN or HF_TOKEN_FILE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT), help="viewer checkout whose data/ to export")
    ap.add_argument("--out", default=None, help="default <root>/data/export")
    ap.add_argument("--repo", default="orbita2d/investigator-agent")
    ap.add_argument("--path-in-repo", default="weirdchat-olens")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)

    meta = {}
    for f in glob.glob(str(root / "data/hf/data/transcripts/qwen__qwen3.6-27b/*.parquet")):
        for r in pq.read_table(f, columns=["transcript_id", "pattern_id", "behavior_id", "is_highlight"]).to_pylist():
            meta[r["transcript_id"]] = r
    reads = {}
    with open(root / "data/reads.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            reads[r["key"]] = r
    rows = []
    for r in reads.values():
        m = meta.get(r["tid"], {})
        s0 = r["samples"][0] if r["samples"] else {}
        rows.append(
            {
                "transcript_id": r["tid"],
                "pattern_id": m.get("pattern_id"),
                "behavior_id": m.get("behavior_id"),
                "is_highlight": m.get("is_highlight"),
                "pos": r["pos"],
                "token": r["token"],
                "layer": r["layer"],
                "temperature": float(r["temperature"]),
                "n": r["n"],
                "seed": r["seed"],
                "norm": r["norm"],
                "fve": s0.get("fve"),
                "top_bullet_fve": max(s0.get("bfve") or [0.0]) if s0.get("bfve") else None,
                "bullets": s0.get("bullets", []),
                "bfve": s0.get("bfve"),
                "coefs": s0.get("coefs", []),
                "samples": [
                    {
                        "text": s["text"],
                        "bullets": s["bullets"],
                        "coefs": s["coefs"],
                        "bfve": s.get("bfve"),
                        "fve": s["fve"],
                    }
                    for s in r["samples"]
                ],
            }
        )
    rows.sort(key=lambda r: (r["transcript_id"], r["pos"], r["layer"], r["temperature"]))
    out = Path(args.out or root / "data/export")
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out / "reads.parquet", compression="zstd")
    n_tid = len({r["transcript_id"] for r in rows})
    (out / "README.md").write_text(CARD.format(path=args.path_in_repo, n=len(rows), n_tid=n_tid))
    print(f"{len(rows)} reads over {n_tid} transcripts -> {out}")
    if args.no_upload:
        return
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token())
    api.upload_file(
        path_or_fileobj=out / "reads.parquet",
        path_in_repo=f"{args.path_in_repo}/reads.parquet",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message=f"WeirdChat O-lens reads: {len(rows)} rows",
    )
    api.upload_file(
        path_or_fileobj=out / "README.md",
        path_in_repo=f"{args.path_in_repo}/README.md",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="WeirdChat O-lens reads: card",
    )
    print(f"uploaded to https://huggingface.co/datasets/{args.repo}/tree/main/{args.path_in_repo}")


if __name__ == "__main__":
    main()
