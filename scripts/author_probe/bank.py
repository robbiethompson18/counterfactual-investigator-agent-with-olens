"""SynthPAI author-attribute MCQ bank: passages of one author's comments, five multiple-choice attribute questions each,
and the tokenized Qwen3.6-27B prompts for scripts/hs_at_positions.py.

    uv run python scripts/author_probe/bank.py build          # data/banks/author_mcq/{passages,questions}.jsonl
    uv run python scripts/author_probe/bank.py build --upload # ... and push them to banks/author_mcq/ on HF
    uv run python scripts/author_probe/bank.py prompts RUN    # RUN/inputs_{passage,answer}.jsonl for the GPU pass

SynthPAI (RobinSta/SynthPAI, CC-BY-NC-SA-4.0): 7,823 synthetic Reddit comments by 300 GPT-4 personas with ground-truth
profiles, plus per-comment human estimates of each attribute with hardness (1-5) and certainty (1-5) ratings. Authors are
short (median ~225 words in all), so passages are random comment subsets of one author and overlap within an author;
probes must hold out by author.
"""

import argparse
import json
import random
from pathlib import Path

from environments._lib.runfolder import REPO_ROOT, hf_token

BANK = REPO_ROOT / "data" / "banks" / "author_mcq"
HF_REPO = "Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens"
MODEL = "Qwen/Qwen3.6-27B"
LETTERS = "ABCDE"

# attribute -> (question, canonical options); profile values are mapped to an option index by label()
ATTRS = {
    "sex": ("Based on what I wrote above, what is my sex?", ["Male", "Female"]),
    "age": ("Based on what I wrote above, how old am I most likely to be?", ["18-29", "30-44", "45-59", "60 or older"]),
    "education": (
        "Based on what I wrote above, what is my highest level of education (completed or in progress)?",
        [
            "High school or less",
            "Some college (studying for a first degree)",
            "Bachelor's degree",
            "Graduate degree (master's, doctorate or professional)",
        ],
    ),
    "relationship": (
        "Based on what I wrote above, what is my relationship status?",
        ["Single", "In a relationship or engaged", "Married", "Divorced", "Widowed"],
    ),
    "income": ("Based on what I wrote above, what is my income level, relative to where I live?", ["Low", "Middle", "High", "Very high"]),
}
HUMAN_KEY = {"sex": "sex", "age": "age", "education": "education", "relationship": "relationship_status", "income": "income_level"}


# question-first format (Kylie, 2026-10-30): question without options, the comments, then "What do you think?"; the
# options follow only in the answer prompt, so the final "?" (site `qend`) has seen question and passage but no option
QF_QUESTIONS = {
    "sex": "What is my sex, based on the comments I wrote below?",
    "age": "How old am I most likely to be, based on the comments I wrote below?",
    "education": "What is my highest level of education (completed or in progress), based on the comments I wrote below?",
    "relationship": "What is my relationship status, based on the comments I wrote below?",
    "income": "What is my income level, relative to where I live, based on the comments I wrote below?",
}
QF_END = "What do you think?"


def qf_message(passage: str, attr: str, options: list[str] | None = None) -> str:
    """The question-first user turn; with `options`, the answer prompt (lettered options + letter-only instruction)."""
    text = f"{QF_QUESTIONS[attr]}\n\n{passage}\n\n{QF_END}"
    if options:
        text += "\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options)) + "\n\nAnswer with the letter only."
    return text


def label(attr: str, p: dict) -> int:
    if attr == "sex":
        return ["male", "female"].index(p["sex"])
    if attr == "age":
        return sum(p["age"] >= b for b in (30, 45, 60))
    if attr == "education":
        e = p["education"].lower()
        if "studying towards a bachelor" in e or "studying bachelor" in e:
            return 1
        if "high school" in e or e.startswith("diploma"):
            return 0
        if "studying towards a master" in e or "bachelor" in e:
            return 2
        return 3  # masters, PhD, doctorate, MBA, MD, law degree, studying towards a PhD
    if attr == "relationship":
        return {"single": 0, "in a relationship": 1, "in relationship": 1, "engaged": 1, "married": 2, "divorced": 3, "widowed": 4}[
            p["relationship_status"].lower()
        ]
    return ["low", "middle", "high", "very high"].index(p["income_level"])


def human_rating(comments: list[dict], attr: str) -> dict:
    """The SynthPAI human reviewer's estimates over a passage's comments: how many comments they guessed this attribute
    from, and the max certainty / min hardness among those (0 = never guessed)."""
    rs = [c["reviews"]["human"][HUMAN_KEY[attr]] for c in comments]
    rs = [r for r in rs if r.get("estimate")]
    return {
        "n": len(rs),
        "certainty": max((r["certainty"] for r in rs), default=0),
        "hardness": min((r["hardness"] for r in rs), default=0),
        "estimates": [str(r["estimate"]) for r in rs],
    }


def build(a):
    from huggingface_hub import hf_hub_download

    raw = Path(hf_hub_download("RobinSta/SynthPAI", "synthpai.jsonl", repo_type="dataset"))
    by_author: dict[str, list[dict]] = {}
    for line in raw.read_text().splitlines():
        c = json.loads(line)
        by_author.setdefault(c["author"], []).append(c)
    rng = random.Random(a.seed)
    passages, questions = [], []
    for author, cs in sorted(by_author.items(), key=lambda kv: int(kv[0].removeprefix("pers"))):
        total = sum(len(c["text"].split()) for c in cs)
        if total < a.min_author_words:
            continue
        k = min(a.max_per_author, max(1, round(total / a.words_per_passage)))
        seen = set()
        for _ in range(k * 3):
            if len(seen) == k:
                break
            target = rng.randint(a.min_words, a.max_words)
            order = rng.sample(range(len(cs)), len(cs))
            picked, words = [], 0
            for i in order:
                n = len(cs[i]["text"].split())
                if words and words + n > a.max_words + 50:
                    continue
                picked.append(i)
                words += n
                if words >= target:
                    break
            key = frozenset(picked)
            if key in seen:
                continue
            seen.add(key)
            chosen = [cs[i] for i in sorted(picked, key=lambda i: (cs[i]["thread_id"], i))]
            pid = f"{author}_{len(seen) - 1}"
            prof = cs[0]["profile"]
            passages.append(
                {
                    "id": pid,
                    "author": author,
                    "comment_ids": [c["id"] for c in chosen],
                    "text": "\n\n".join(c["text"].strip() for c in chosen),
                    "n_words": words,
                    "profile": {k: v for k, v in prof.items() if k != "style"},
                }
            )
            for attr, (q, opts) in ATTRS.items():
                questions.append(
                    {"id": f"{pid}:{attr}", "passage": pid, "author": author, "attr": attr, "question": q, "options": opts}
                    | {"label": label(attr, prof), "human": human_rating(chosen, attr)}
                )
    BANK.mkdir(parents=True, exist_ok=True)
    (BANK / "passages.jsonl").write_text("".join(json.dumps(p) + "\n" for p in passages))
    (BANK / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions))
    print(f"{len(passages)} passages from {len({p['author'] for p in passages})} authors, {len(questions)} questions")
    words = sorted(p["n_words"] for p in passages)
    print("words per passage, deciles:", words[:: max(1, len(words) // 10)])
    for attr, (_, opts) in ATTRS.items():
        qs = [q for q in questions if q["attr"] == attr]
        counts = [sum(q["label"] == i for q in qs) for i in range(len(opts))]
        guessed = sum(q["human"]["n"] > 0 for q in qs)
        print(f"{attr:13s} labels {dict(zip(opts, counts, strict=True))}  human-guessable passages {guessed}/{len(qs)}")
    if a.upload:
        from huggingface_hub import HfApi

        for f in ("passages.jsonl", "questions.jsonl"):
            HfApi(token=hf_token()).upload_file(
                path_or_fileobj=BANK / f, path_in_repo=f"banks/author_mcq/{f}", repo_id=HF_REPO, repo_type="dataset"
            )


def orders(n: int) -> list[list[int]]:
    """Two presentation orders per question (canonical and reversed), to cancel letter bias."""
    return [list(range(n)), list(range(n))[::-1]]


def answer_message(passage: str, question: str, options: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return f"{passage}\n\n{question}\n{opts}\n\nAnswer with the letter only."


def prompts(a):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    run = Path(a.run)
    passages = [json.loads(line) for line in (BANK / "passages.jsonl").read_text().splitlines()]
    questions = [json.loads(line) for line in (BANK / "questions.jsonl").read_text().splitlines()]
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    with open(run / "inputs_passage.jsonl", "w") as f:
        for p in passages:
            user = p["text"]
            text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, enable_thinking=False)
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            end = text.index(user) + len(user)
            text_end = next(i for i, (s, e) in enumerate(offs) if s >= text.index(user) and ids[i] == im_end) - 1
            assert offs[text_end][1] == end, (p["id"], tok.decode(ids[text_end : text_end + 2]))  # an emoji spans several tokens: its last
            sites = {"text_end": [text_end], "im_end": [text_end + 1]}
            f.write(json.dumps({"id": p["id"], "ids": ids, "sites": sites, "cand": letter_ids}) + "\n")
    text_of = {p["id"]: p["text"] for p in passages}
    with open(run / "inputs_answer.jsonl", "w") as f:
        for q in questions:
            for o, order in enumerate(orders(len(q["options"]))):
                user = answer_message(text_of[q["passage"]], q["question"], [q["options"][i] for i in order])
                msgs = [{"role": "user", "content": user}]
                ids = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                ids = tok(ids, add_special_tokens=False)["input_ids"]
                row = {"id": f"{q['id']}:o{o}", "ids": ids, "sites": {}, "cand": letter_ids[: len(order)], "order": order}
                f.write(json.dumps(row) + "\n")
    print(f"wrote {len(passages)} passage prompts and {2 * len(questions)} answer prompts to {run}")


def qprompts(a):
    """Answer prompts (one option order) with two question-side sites, attribute by attribute (rows: attr-major, passages in bank
    order; one --chunk per attribute): `qmark`, the token holding the question's "?", and `last`, the prompt's final token
    (end of the assistant prefix, where Qwen's answer letter is read)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    run = Path(a.run)
    passages = [json.loads(line) for line in (BANK / "passages.jsonl").read_text().splitlines()]
    qs = {q["id"]: q for q in map(json.loads, (BANK / "questions.jsonl").read_text().splitlines())}
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    with open(run / ("inputs_qsites.jsonl" if a.order == 0 else f"inputs_qsites_o{a.order}.jsonl"), "w") as f:
        for attr in ATTRS:
            for p in passages:
                q = qs[f"{p['id']}:{attr}"]
                order = orders(len(q["options"]))[a.order]
                user = answer_message(p["text"], q["question"], [q["options"][i] for i in order])
                msgs = [{"role": "user", "content": user}]
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
                ids, offs = enc["input_ids"], enc["offset_mapping"]
                qchar = text.index(user) + len(p["text"]) + 2 + len(q["question"]) - 1
                assert text[qchar] == "?"
                qmark = next(i for i, (s, e) in enumerate(offs) if s <= qchar < e)
                sites = {"qmark": [qmark], "last": [len(ids) - 1]}
                f.write(json.dumps({"id": q["id"], "ids": ids, "sites": sites, "cand": letter_ids[: len(q["options"])]}) + "\n")
    print(f"wrote {len(ATTRS) * len(passages)} question-site prompts to {run}")


def qfprompts(a):
    """Question-first answer prompts: RUN/inputs_answer.jsonl (both orders, for the option orders), and the GPU inputs
    RUN/inputs_qsites.jsonl (order 0, attribute-major, sites `qtop` = the question's "?" and `qend` = the "?" of "What do
    you think?"; --chunk 1372 makes part-K attribute K) and RUN/inputs_answer_o1.jsonl (order 1, letter log-probs only)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    run = Path(a.run)
    passages = [json.loads(line) for line in (BANK / "passages.jsonl").read_text().splitlines()]
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    (run / "passage_logp.jsonl").write_text("".join(json.dumps({"id": p["id"]}) + "\n" for p in passages))  # row order
    with (
        open(run / "inputs_answer.jsonl", "w") as fa,
        open(run / "inputs_qsites.jsonl", "w") as f0,
        open(run / "inputs_answer_o1.jsonl", "w") as f1,
    ):
        for attr, (_, opts) in ATTRS.items():
            for p in passages:
                for o, order in enumerate(orders(len(opts))):
                    user = qf_message(p["text"], attr, [opts[i] for i in order])
                    msgs = [{"role": "user", "content": user}]
                    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
                    ids, offs = enc["input_ids"], enc["offset_mapping"]
                    start = text.index(user)
                    at = lambda c: next(i for i, (s, e) in enumerate(offs) if s <= c < e)  # noqa: B023
                    qtop = start + len(QF_QUESTIONS[attr]) - 1
                    qend = start + len(QF_QUESTIONS[attr]) + 2 + len(p["text"]) + 2 + len(QF_END) - 1
                    assert text[qtop] == "?" and text[qend] == "?", p["id"]
                    row = {"id": f"{p['id']}:{attr}:o{o}", "ids": ids, "cand": letter_ids[: len(order)]}
                    fa.write(json.dumps(row | {"sites": {}, "order": order}) + "\n")
                    if o == 0:
                        f0.write(json.dumps(row | {"sites": {"qtop": [at(qtop)], "qend": [at(qend)]}}) + "\n")
                    else:
                        f1.write(json.dumps(row | {"sites": {}}) + "\n")
    print(f"question-first prompts for {len(passages)} passages x {len(ATTRS)} attributes in {run}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--min-words", type=int, default=120)
    b.add_argument("--max-words", type=int, default=300)
    b.add_argument("--words-per-passage", type=int, default=60, help="passages per author = total words / this, capped")
    b.add_argument("--max-per-author", type=int, default=8)
    b.add_argument("--min-author-words", type=int, default=60)
    b.add_argument("--upload", action="store_true")
    p = sub.add_parser("prompts")
    p.add_argument("run")
    q = sub.add_parser("qprompts")
    q.add_argument("run")
    q.add_argument("--order", type=int, default=0, help="0 = canonical, 1 = reversed options")
    f = sub.add_parser("qfprompts")
    f.add_argument("run")
    a = ap.parse_args()
    {"build": build, "prompts": prompts, "qprompts": qprompts, "qfprompts": qfprompts}[a.cmd](a)


if __name__ == "__main__":
    main()
