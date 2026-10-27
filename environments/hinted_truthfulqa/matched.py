"""Within-question matched cases for "would the answer change if we removed the hint?" (Kylie, 2026-10-27).

    uv run python -m environments.hinted_truthfulqa.matched propose   # -> matched_items.json
    uv run python -m environments.hinted_truthfulqa.matched cases     # -> matched_cases.json

propose: per model, questions whose no-hint answer is one letter >= 7/8 times (from the 8-sample control runs): every one it
gets reliably wrong plus as many it gets reliably right (seeded), so "the model knows this one" is half the set. Each question
gets two items, `tqa-N.h<own>` (the hint names the model's own no-hint answer: picking it is what it would do anyway) and
`tqa-N.h<never>` (a letter it picked 0/8 times, seeded; may be the correct one: picking it is what the hint changed).
cases: per model and question, one hinted transcript that picked the hinted letter from each item, so every question gives
one `anyway` and one `changed` case that differ only in the letter. Questions without a followed `never` transcript drop out.
"""

import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from environments._lib.runfolder import LOCAL, read_episodes

from . import env as E

HERE = Path(E.__file__).parent
ARMS = ["hint/wikipedia", "hint/professor", "hint/stanford"]  # the sources that move every model
TAGS = {"deepseek/deepseek-v4-flash": "dsv4f", "z-ai/glm-5.3": "glm53", "Qwen/Qwen3.6-27B": "qwen36"}


def labelled(glob):
    out = defaultdict(list)
    for d in sorted((LOCAL / "hinted_truthfulqa").glob(glob)):
        if d.name in E.RETIRED_RUNS:
            continue
        model = json.loads((d / "meta.json").read_text())["target"]["model"]
        f = max((d / "labels").glob("measure@*.jsonl"), key=os.path.getmtime)
        lab = {r["episode"]: r["values"] for r in map(json.loads, f.read_text().splitlines())}
        out[model] += [(d.name, ep, lab.get(ep["id"]) or {}) for ep in read_episodes(d) if not ep.get("error")]
    return out


def propose(models):
    rng = random.Random(0)
    nohint = defaultdict(lambda: defaultdict(list))
    for model, rows in labelled("*_control-n8-*").items():
        for _, ep, v in rows:
            if "." not in ep["item"]:
                nohint[model][ep["item"]].append(v.get("answer"))
    out = {}
    for model in models:
        right, wrong = [], []
        for q, ans in sorted(nohint[model].items()):
            own, k = Counter(ans).most_common(1)[0]
            if not own or len(ans) < 8 or k < 7:
                continue
            _, opts, gold = E.question(q)
            never = [E.LETTERS[j] for j in range(len(opts)) if E.LETTERS[j] not in ans]
            if never:
                (right if own == gold else wrong).append((q, own, rng.choice(never)))
        rng.shuffle(right)
        picked = sorted(wrong + right[: len(wrong)], key=lambda t: int(t[0][4:]))
        out[model] = {q: {"own": own, "never": never, "own_correct": own == E.question(q)[2]} for q, own, never in picked}
        ids = [f"{q}.h{x}" for q, own, never in picked for x in (own, never)]
        print(f"{model}: {len(wrong)} reliably wrong, {len(right)} reliably right -> {len(picked)} questions, {len(ids)} items")
    (HERE / "matched_items.json").write_text(json.dumps(out, indent=1) + "\n")


def cases():
    m = json.loads((HERE / "matched_items.json").read_text())
    rng = random.Random(0)
    out = {}
    for model, rows in labelled("*_matched2-*").items():
        qs = m.get(model, {})
        got = defaultdict(lambda: defaultdict(list))  # question -> kind -> followed transcripts
        for run, ep, v in rows:
            q, _, hint = E.parse_item(ep["item"])
            if q not in qs or not v.get("followed"):
                continue
            kind = "anyway" if hint == qs[q]["own"] else "changed"
            got[q][kind].append(
                {
                    "run": run,
                    "episode": ep["id"],
                    "arm": ep["arm"],
                    "item": ep["item"],
                    "label": kind,
                    "own_correct": qs[q]["own_correct"],
                    "docent_url": ep.get("docent_url"),
                }
            )
        pairs = []
        for q, by in sorted(got.items(), key=lambda kv: int(kv[0][4:])):
            if by["anyway"] and by["changed"]:
                pairs += [rng.choice(by["anyway"]), rng.choice(by["changed"])]
        out[model] = pairs
        n = len(pairs) // 2
        print(
            f"{model}: {len(qs)} questions, {n} with both a followed own-letter and a followed never-letter transcript "
            f"({sum(1 for c in pairs[0::2] if c['own_correct'])} of them reliably right) -> {len(pairs)} cases"
        )
    (HERE / "matched_cases.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "propose":
        propose(sys.argv[2:] or list(TAGS))
    else:
        cases()
