"""cutoff_qa_inline's second 1k (Kylie, 2026-10-30): 1,000 more seeded stable, non-numeric bank-v4 items, not among the
999 of the 2026-10-29 runs, built straight from the bank (the label is the bank's 8-sample modal answer and the reads are
of the prompt alone, so no Qwen episode is needed; `source` ids are `cutoff_qa/bank-v4:nohint~ITEM~0`).

    PYTHONPATH=. uv run --with transformers --with jinja2 python -m cfbench.evals.cutoff_qa_inline.extend OUT_DIR

writes OUT_DIR/cases.jsonl (cutoff_qa_inline's case format) and OUT_DIR/requests.jsonl: `text` rows of the thinking-off
render with positions = every PUNCT token of the prompt + the six tail tokens through the "\\n" after "assistant" (never the
think block), for scripts/sky/olens_reads.yaml at layers 32-60.
"""

import json
import random
import re
import sys
from pathlib import Path

from environments._lib.core import load_env

PUNCT = re.compile(r"^\s*[.,;:!?]+[\"')\]*]*\s*$")  # scripts/fill_read_requests.py's
OLD = Path("data/runs/cfbench_evals/2026-10-29_cutoff-qa-inline-inputs-1k/cases.jsonl")  # the first 999
N, SEED = 1000, 0


def main(out: Path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    mod = load_env("cutoff_qa")
    old = {json.loads(line)["item"] for line in OLD.open()}
    pool = [i for i in mod.items() if i not in old and not mod.build("nohint", i).meta["numeric"]]
    pick = set(random.Random(SEED).sample(pool, N))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cases.jsonl").open("w") as fc, (out / "requests.jsonl").open("w") as fr:
        for item in [i for i in mod.items() if i in pick]:  # bank order
            ep = mod.build("nohint", item)
            src = f"cutoff_qa/bank-v4:nohint~{item}~0"
            meta = ep.meta
            fc.write(json.dumps({"version": "nohint", "item": item, "modal": mod.norm(meta["qwen_modal"]), "source": src,
                                 "messages": ep.messages, "meta": meta}, ensure_ascii=False) + "\n")  # fmt: skip
            text = tok.apply_chat_template(ep.messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            toks = [tok.decode([t]) for t in tok(text, add_special_tokens=False).input_ids]
            end = toks.index("<|im_end|>")
            assert toks[end : end + 5] == ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"], item
            pos = sorted({p for p, t in enumerate(toks[:end]) if PUNCT.match(t)} | set(range(end, end + 5)))
            fr.write(json.dumps({"id": src, "item": item, "version": "nohint", "text": text, "positions": pos}, ensure_ascii=False) + "\n")
    print(len(pool), "candidates;", len(pick), "picked ->", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
