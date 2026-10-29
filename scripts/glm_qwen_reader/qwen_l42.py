"""Control for the Qwen reader: ceselder's public Qwen3.6-27B L42 lens (ceselder/olens-qwen3.6-27b-l42 lens/) on Qwen's OWN layer-42
activations at the tokens of a Qwen batch-reads run, decoded exactly as qwen_reader.py decodes A(h51) (same injection, grammar,
greedy + 4 T=1), so the two differ only by where the vector comes from.

    PYTHONPATH=.:src uv run --with transformers --with jinja2 python scripts/glm_qwen_reader/qwen_l42.py prep QWEN_RUN JOB.json
    python qwen_l42.py capture JOB.json H42.pt                 # GPU: block-42 output (HF hidden_states[43]) at each unique position
    python qwen_reader.py H42.pt L42/lens L42/olens_usage.py OUT.jsonl --no-affine
    uv run python scripts/glm_qwen_reader/qwen_l42.py rows JOB.json OUT.jsonl READS.jsonl   # -> batch-reads rows, dups copied

Positions are the run's read positions (every prompt token); a position whose token prefix an earlier row has is captured once.
"""

import json
import sys
from pathlib import Path

mode = sys.argv[1]
if mode == "prep":
    from transformers import AutoTokenizer

    from counterfactual_investigator_agent_with_olens.render import render_request

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    seen, rows = {}, []
    for r in map(json.loads, (Path(sys.argv[2]) / "requests.jsonl").read_text().splitlines()):
        _, ids, _ = render_request(tok, r)
        own, dups = [], []
        for p in r["positions"]:
            k = tuple(ids[: p + 1])
            if k in seen:
                dups.append([p, *seen[k]])
            else:
                seen[k] = (r["id"], p)
                own.append(p)
        rows.append({"id": r["id"], "ids": ids, "positions": own, "dups": dups, "tokens": [tok.decode([t]) for t in ids]})
    Path(sys.argv[3]).write_text(json.dumps(rows))
    print(f"{len(rows)} rows, {sum(len(r['positions']) for r in rows)} positions + {sum(len(r['dups']) for r in rows)} copies")
elif mode == "capture":
    import torch
    from transformers import AutoModelForCausalLM

    rows = json.loads(Path(sys.argv[2]).read_text())
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9", dtype=torch.bfloat16, device_map={"": 0}
    ).eval()
    ids, pos, hs = [], [], []
    with torch.no_grad():
        for r in rows:
            out = model(input_ids=torch.tensor([r["ids"]], device="cuda"), output_hidden_states=True, use_cache=False)
            hs.append(out.hidden_states[43][0, r["positions"]].float().cpu())
            ids += [r["id"]] * len(r["positions"])
            pos += r["positions"]
    torch.save({"ids": ids, "pos": pos, "h": torch.cat(hs)}, sys.argv[3])
    print("captured", len(pos), "positions")
else:
    rows = json.loads(Path(sys.argv[2]).read_text())
    reads = {(x["id"], x["pos"]): x["reads"] for x in map(json.loads, Path(sys.argv[3]).read_text().splitlines())}
    n = 0
    with open(sys.argv[4], "w") as f:
        for r in rows:
            for p, cid, cp in sorted([(p, r["id"], p) for p in r["positions"]] + [tuple(d) for d in r["dups"]]):
                draws = [" · ".join(b.removeprefix("*").strip() for b in t.splitlines() if b.strip()) for t in reads[(cid, cp)]]
                row = {"id": r["id"], "token_index": p, "token": r["tokens"][p], "layer": "42 Qwen lens (ceselder) on Qwen",
                       "bullets": [{"text": d, "fve": None} for d in draws], "joint_fve": None}  # fmt: skip
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    print(n, "rows ->", sys.argv[4])
