"""Build the essay_grading PERSUADE bank: real student intro paragraphs to grade (Kylie, 2026-10-30).

Source: the PERSUADE 2.0 corpus (Crossley et al. 2024, Assessing Writing 61; CC BY 4.0;
https://github.com/scrosseye/persuade_corpus_2.0), its training csv: argumentative essays by US students in grades 6-12,
each with a human holistic score 1-6 and its discourse elements tagged. An item is one essay's `Lead` (intro paragraph),
40-100 words, whitespace collapsed. N_PER_SCORE per holistic score, drawn with SEED round-robin over the 15 prompts, so
the bank spans essay quality and topic.

    uv run python -m environments.essay_grading.build_bank PERSUADE_CSV     # -> data/banks/essay_grading/persuade_v1.json
"""

import json
import random
import sys
from collections import defaultdict

import pandas as pd

from environments._lib.runfolder import REPO_ROOT

VERSION, SEED, N_PER_SCORE = "persuade_v1", 0, 34
WORDS = (40, 100)


def main(csv: str):
    m = pd.read_csv(csv, low_memory=False)
    lead = m[m.discourse_type == "Lead"].drop_duplicates("essay_id_comp", keep="first")
    lead = lead.assign(text=lead.discourse_text.str.split().str.join(" "))
    lead = lead[lead.text.str.split().str.len().between(*WORDS)].sort_values("essay_id_comp")
    rng, items = random.Random(SEED), []
    for score, rows in lead.groupby("holistic_essay_score"):
        by_prompt = defaultdict(list)
        for r in rows.itertuples():
            by_prompt[r.prompt_name].append(r)
        for p in by_prompt.values():
            rng.shuffle(p)
        prompts = sorted(by_prompt)
        rng.shuffle(prompts)
        picked = []
        while len(picked) < N_PER_SCORE and any(by_prompt.values()):
            for p in prompts:
                if by_prompt[p] and len(picked) < N_PER_SCORE:
                    picked.append(by_prompt[p].pop())
        items += [{"id": f"p-{r.essay_id_comp}", "text": r.text, "holistic_score": int(score), "prompt_name": r.prompt_name,
                   "task": r.task, "grade_level": None if pd.isna(r.grade_level) else int(r.grade_level),
                   "essay_id": r.essay_id_comp} for r in picked]  # fmt: skip
    out = REPO_ROOT / f"data/banks/essay_grading/{VERSION}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"source": "PERSUADE 2.0 (Crossley et al. 2024), CC BY 4.0, https://github.com/scrosseye/persuade_corpus_2.0",
            "rule": f"Lead paragraphs of {WORDS[0]}-{WORDS[1]} words, {N_PER_SCORE} per holistic score, round-robin over prompts",
            "seed": SEED}  # fmt: skip
    out.write_text(json.dumps(meta | {"items": items}, indent=1))
    print(f"{len(lead)} eligible leads -> {len(items)} items: {pd.Series([i['holistic_score'] for i in items]).value_counts().to_dict()}")


if __name__ == "__main__":
    main(sys.argv[1])
