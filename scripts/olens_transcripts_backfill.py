"""Write X.transcripts.jsonl beside batch reads X.jsonl made before the readers wrote it (docs/olens-reads-format.md): each
request rendered as the reader renders it (render.render_request), checked against every read row's `token`.

    uv run --with transformers --with jinja2 python scripts/olens_transcripts_backfill.py RUN/reads.jsonl [--requests RUN/requests.jsonl]

Then `python expview/importers/reads.py RUN` (or `scripts/runs.py finish RUN`) makes the expview run. Qwen3.6-27B requests
only; GLM reads need GLM's tokenizer and template.
"""

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from counterfactual_investigator_agent_with_olens.render import render_request, transcript_line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reads", type=Path)
    ap.add_argument("--requests", type=Path, help="default: requests.jsonl beside the reads")
    args = ap.parse_args()
    requests = args.requests or args.reads.with_name("requests.jsonl")
    want = {}  # id -> {token_index: token}
    with open(args.reads) as f:
        for line in f:
            r = json.loads(line)
            want.setdefault(r["id"], {})[r["token_index"]] = r["token"]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    out = args.reads.with_suffix(".transcripts.jsonl")
    n, bad = 0, []
    with open(out, "w") as f, open(requests) as req:
        for line in req:
            row = json.loads(line)
            if row["id"] not in want:
                continue
            msgs, ids, where = render_request(tok, row)
            toks = [tok.decode([i]) for i in ids]
            if any(p >= len(toks) or toks[p] != t for p, t in want[row["id"]].items()):
                bad.append(row["id"])
                continue
            f.write(json.dumps(transcript_line(row, msgs, toks, where), ensure_ascii=False) + "\n")
            n += 1
    missing = len(want) - n - len(bad)
    print(f"{out}: {n} transcripts; {len(bad)} whose render doesn't match their reads (left out), {missing} with no request")
    if bad:
        print("mismatched:", *bad[:10])


if __name__ == "__main__":
    main()
