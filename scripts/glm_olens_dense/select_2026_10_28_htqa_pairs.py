"""Kylie 2026-10-28 00:00: for the hinted TruthfulQA pairs list (env/run:episode_id per line, 26 questions), 1 hinted + 1 no-hint
episode per question (seed 0), every template and punctuation token. Writes a data/envs/... tree holding only those episodes, for
prep_envs.py to prep from that directory (so exact-prefix dedup never points a picked episode at an unpicked one).

    uv run python scripts/glm_olens_dense/select_2026_10_28_htqa_pairs.py <sources.txt> <out dir>
    cd <out dir> && uv run --project <repo> --with transformers --with jinja2 --with torch python \
        <repo>/scripts/glm_olens_dense/prep_envs.py --tok <GLM-5.3 tokenizer> --out prep --jobs jobs \
        --runs 'data/envs/hinted_truthfulqa/*' --reads '<repo>/data/runs/olens_reads/*glm53*/*'
"""

import collections
import json
import random
import sys
from pathlib import Path


def main():
    src, out = [x.strip() for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()], Path(sys.argv[2])
    by = collections.defaultdict(list)
    for s in src:
        run, eid = s.split(":", 1)
        arm, item, _ = eid.split("~")
        by[(item.split(".")[0], "hint" if arm.startswith("hint") else "nohint")].append((run, eid))
    pick = collections.defaultdict(set)
    for _, v in sorted(by.items()):
        run, eid = random.Random(0).choice(sorted(v))
        pick[run].add(eid)
    for run, eids in pick.items():
        d, src_d = out / "data/envs" / run, Path("data/envs") / run
        (d / "episodes").mkdir(parents=True)
        (d / "meta.json").write_text((src_d / "meta.json").read_text())
        lines = [x for f in sorted((src_d / "episodes").glob("*.jsonl")) for x in f.read_text().splitlines() if json.loads(x)["id"] in eids]
        assert len(lines) == len(eids), run
        (d / "episodes/000.jsonl").write_text("\n".join(lines) + "\n")
    (out / "selection.txt").write_text("\n".join(f"{r}:{e}" for r, es in sorted(pick.items()) for e in sorted(es)) + "\n")
    print(f"{len({q for q, _ in by})} questions, {sum(map(len, pick.values()))} episodes")


if __name__ == "__main__":
    main()
