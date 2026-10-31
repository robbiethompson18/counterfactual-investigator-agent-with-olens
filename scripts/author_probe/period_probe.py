"""Linear probes on sentence-end activations (Kylie, 2026-10-31; exploratory, post hoc): does Qwen3.6-27B's residual
stream at a sentence's final token say whether deleting that sentence makes it much less sure of its answer?

Prompts are the eval's question-first prompts (blog.message, no options, thinking-off template), as the O-lens read.
Sites per prompt: every sentence end of the post (the token holding the sentence's last character, blog.py's split),
the question's "?" and the final "?" of "What do you think?". Eval: the 533 batch 1+2 pairs of author_blog_drop_inline;
train: batch 3's confident pairs (no shared bloggers), last sentence end + both "?" only.

    uv run --with transformers --with jinja2 python -m scripts.author_probe.period_probe inputs RUN   # RUN/hs_inputs.jsonl
    # GPU: scripts/sky/hs_positions_any.yaml, ARGS="--layers 16,20,24,28,32,36,40,44,48,52,56,60 --chunk 1000"
    uv run --with scikit-learn --with scipy python -m scripts.author_probe.period_probe analyze RUN
"""

import json
import sys
from pathlib import Path

import numpy as np

from scripts.author_probe.blog import BANK, QUESTIONS, sentences
from scripts.author_probe.blog import message as qf_message

INPUTS = Path("data/runs/cfbench_evals/2026-10-30_author-blog-drop-inline-inputs-b2")
TRAIN_ANSWERS = Path("data/runs/author_mcq_blog/2026-10-30_qwen36-qfirst-c/qwen_answers.jsonl")
MODEL = "Qwen/Qwen3.6-27B"
N_TRAIN = 1500  # confident batch-3 pairs per attribute
LAYERS = [16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in Path(p).read_text().splitlines()]


def conf(r: dict) -> float:
    k = len(r["pmf"])
    return (max(r["pmf"]) - 1 / k) / (1 - 1 / k)


def sites(tok, text: str, attr: str) -> tuple[list[int], list[int], list[str]]:
    """(token ids, site token indices, site kinds: 'q', 's0', 's1', ..., 'end') for the question-first prompt."""
    user = qf_message(text, attr)  # no options: ends with "What do you think?"
    rendered = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, enable_thinking=False)
    enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
    s0 = rendered.index(user)
    tok_at = lambda c: next(i for i, (a, b) in enumerate(enc["offset_mapping"]) if a <= s0 + c < b)
    chars, kinds = [len(QUESTIONS[attr]) - 1], ["q"]
    cur = len(QUESTIONS[attr]) + 2
    for j, s in enumerate(sentences(text)):
        a = user.index(s, cur)
        cur = a + len(s)
        chars.append(cur - 1)
        kinds.append(f"s{j}")
    chars.append(len(user) - 1)
    kinds.append("end")
    return enc["input_ids"], [tok_at(c) for c in chars], kinds


def inputs(run: Path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    passages = {p["id"]: p for p in load(BANK / "passages.jsonl")}
    evals = sorted({c["pair"] for c in load(INPUTS / "cases.jsonl")})
    train = []
    for attr in ("sex", "age"):
        rs = sorted((r for r in load(TRAIN_ANSWERS) if r["attr"] == attr and r["order_agree"] and conf(r) >= 0.5), key=lambda r: r["id"])
        train += [r["id"] for r in np.random.default_rng(0).permutation(np.array(rs, dtype=object))[:N_TRAIN]]
    rows, index = [], []
    for split, pairs in (("eval", evals), ("train", train)):
        for pair in pairs:
            pid, attr = pair.split(":")
            ids, pos, kinds = sites(tok, passages[pid]["text"], attr)
            if split == "train":  # the question "?", the post's last sentence end, the final "?"
                keep = [0, len(pos) - 2, len(pos) - 1]
                pos, kinds = [pos[i] for i in keep], [kinds[i] for i in keep]
            rows.append({"id": pair, "ids": ids, "sites": {"h": pos}, "cand": []})
            index.append({"id": pair, "split": split, "kinds": kinds, "positions": pos})
    (run / "hs_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))
    n = sum(len(r["positions"]) for r in index)
    print(f"{len(evals)} eval + {len(train)} train prompts, {n} sites -> {run / 'hs_inputs.jsonl'}")


if __name__ == "__main__":
    {"inputs": inputs}[sys.argv[1]](Path(sys.argv[2]))
