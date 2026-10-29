"""Build essay_intro's paragraph bank from PERSUADE 2.0 (Kylie, 2026-10-29: real essays, opening paragraph only).

PERSUADE 2.0 (Crossley et al. 2024, CC BY 4.0): ~25k argumentative essays by US 6th-12th graders on 15 prompts, each with a
human holistic score 1-6. The scored splits are read from the HF mirror nlpatunt/D_persuade_2 (train + validation; its test
split has the scores removed). Per essay, the opening paragraph: the first line of 15+ words (shorter leading lines are titles
or headers), kept when it has 80-200 words and at least 3 sentences. Exact duplicates are dropped. Then up to PER_SCORE per
human score, seeded, so the bank is stratified by human score (score 1 has fewer and is taken whole).

    uv run python scripts/essay_intro_bank.py [--out data/banks/essay_intro/v1.json] [--upload]

The bank is data: HF `banks/essay_intro/v1.json`, which environments/essay_intro fetches on first use. Never in git.
"""

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from counterfactual_investigator_agent_with_olens.runs import REPO

SOURCE = "nlpatunt/D_persuade_2"
SPLITS = ("train", "validation")
PER_SCORE = 340  # ~2,000 over the six scores
WORDS = (80, 200)
SEED = 0
KEEP = ("holistic_essay_score", "prompt_name", "task", "grade_level", "word_count")


def opening(text: str) -> str | None:
    lines = [ln.strip() for ln in re.split(r"\n+", text) if ln.strip()]
    while lines and len(lines[0].split()) < 15:
        lines = lines[1:]
    if len(lines) < 2:  # one block: no paragraph structure to take the first of
        return None
    para = " ".join(lines[0].split())
    n = len(para.split())
    sentences = len(re.findall(r"[.!?](\s|$)", para))
    return para if WORDS[0] <= n <= WORDS[1] and sentences >= 3 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/banks/essay_intro/v1.json")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    csv.field_size_limit(1 << 30)
    by_score, seen = defaultdict(list), set()
    for split in SPLITS:
        path = hf_hub_download(SOURCE, f"{split}.csv", repo_type="dataset")
        for r in csv.DictReader(Path(path).read_text(encoding="utf-8").splitlines(keepends=True)):
            para = opening(r["full_text"])
            if not para or para.lower() in seen:
                continue
            seen.add(para.lower())
            by_score[int(r["holistic_essay_score"])].append(
                {"id": r["essay_id_comp"].lower(), "text": para, "words": len(para.split())} | {k: r[k] for k in KEEP}
            )
    rng = random.Random(SEED)
    items = []
    for score in sorted(by_score):
        pool = sorted(by_score[score], key=lambda it: it["id"])
        items += rng.sample(pool, min(PER_SCORE, len(pool)))
    items.sort(key=lambda it: it["id"])
    for it in items:
        it["human_score"] = int(it.pop("holistic_essay_score"))
    bank = {
        "source": f"PERSUADE 2.0 via HF {SOURCE} ({'+'.join(SPLITS)}), CC BY 4.0",
        "rule": f"opening paragraph, {WORDS[0]}-{WORDS[1]} words, >= 3 sentences, deduped, <= {PER_SCORE} per human score, seed {SEED}",
        "available": {s: len(v) for s, v in sorted(by_score.items())},
        "items": items,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bank, indent=1, ensure_ascii=False))
    print(f"{len(items)} items -> {out}; per human score {dict(sorted(Counter(it['human_score'] for it in items).items()))}")
    print("available per score", bank["available"])
    if args.upload:
        HfApi().upload_file(path_or_fileobj=out, path_in_repo="banks/essay_intro/v1.json", repo_id=REPO, repo_type="dataset")
        print("uploaded banks/essay_intro/v1.json")


if __name__ == "__main__":
    main()
