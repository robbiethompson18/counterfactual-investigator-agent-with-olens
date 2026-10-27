# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2", "pyarrow>=17", "tokenizers>=0.20", "huggingface_hub>=0.25"]
# ///
"""Attach the GLM 5.3 O-lens reads we made on over-eagerness transcripts to their episodes in envs/overeagerness, as expview
runs under data/runs/overeagerness/ (the env page stitches them into each episode).

    uv run --script scripts/overeagerness_olens_expview.py email   # set A: email_ambiguous parts 1-8 (2026-10-25 dense reads)
    uv run --script scripts/overeagerness_olens_expview.py tasks   # set B: 2026-10-26 early-turn reads, 6 tasks

The reads were made before the env existed, on the source runs' rows (scripts/glm_olens_dense/, Celeste's format), so each
transcript (episodes.parquet row: run_dir + row_index) is mapped to the env episode converted from the same row
(environments/_lib/convert_overeager.py stores it as `source` = data/<run_dir>:<row_index>, and the raw episode file as
`raw_source`). Set B's rows were made straight from the raw files (`source` in rows/<task>.jsonl), so it matches on
`raw_source`. Every match is checked: the env episode's messages (role + text) must equal the transcript's row. Parquet
→ tokens/reads is expview/importers/celeste.py; unmatched transcripts are counted and still exported, unattached.
"""

import functools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

sys.path[:0] = ["expview"]
from importers.celeste import READERS, Calls, import_config  # ty: ignore[unresolved-import]

from expview import Run  # ty: ignore[unresolved-import]

ENVS = Path("data/envs/overeagerness")
TASKS = ["calendar", "dana", "email", "email_4johns", "gitbranch", "payback"]
OLENS = Path("data/runs/olens_reads")
SETS: dict[str, dict] = {
    "email": {
        "out": "data/runs/overeagerness/2026-10-27_glm53-olens-email",
        "configs": [
            *[(OLENS / "2026-10-25_glm53_email_dense", f"2026-10-24_glm53_email_ambiguous{p}") for p in ["", "_part2"]],
            (OLENS / "2026-10-25_glm53_ib_user_dense", "2026-10-25_glm53_email_ambiguous_part3"),
            *[
                (OLENS / "2026-10-25_glm53_email_late_dense", f"2026-10-25_glm53_email_ambiguous_{p}")
                for p in ["part4_corp", "part5_junebug", "part6_junebug_topup", "part7_acme_and_128", "part8_fresh128"]
            ],
        ],
        "title": "Over-eagerness email: GLM 5.3 O-lens reads (2026-10-25)",
        "about": "the email_ambiguous runs (parts 1-8 of the email task) on 2026-10-25: dense reads (every punctuation and special "
        "token, the first reply tokens) on a sample of episodes per arm; part 3's reads were uploaded with the ib_user run",
    },
    "tasks": {
        "out": "data/runs/overeagerness/2026-10-27_glm53-olens-tasks",
        "configs": [(OLENS / "2026-10-26_glm53_overeager", t) for t in TASKS],
        "rows": OLENS / "2026-10-26_glm53_overeager/rows",
        # the email-arms run re-uploaded episodes of email_ambiguous parts 5-7, so 384 of its 448 transcripts are in two env runs;
        # these reads were made on the email-arms rows, so they attach there
        "prefer": "2026-10-26_glm53-email-arms",
        "title": "Over-eagerness tasks: GLM 5.3 O-lens reads (2026-10-26)",
        "about": "calendar, dana, gitbranch, payback, email (John, junebug) and the 4-Johns item, 5 arms each, on 2026-10-26: early "
        "turns (messages up to index 2) densely plus 32 positions spread over the rest",
    },
}


def text(m) -> str:
    """A message as role + text, ignoring how tool calls were serialised (the env, its source rows and the raw files differ)."""
    calls = [(c.get("function") or c).get("arguments") for c in m.get("tool_calls") or []]
    return json.dumps([m["role"], m.get("content") or "", calls])


@functools.cache
def env_episodes():
    by_src, by_raw, by_msgs = {}, defaultdict(list), defaultdict(list)
    for f in sorted(ENVS.glob("*/episodes/*.jsonl")):
        run = f.parent.parent.name
        for line in f.open():
            e = json.loads(line)
            key = (f"envs/overeagerness/{run}/{e['id']}", [text(m) for m in e["messages"]])
            by_src[e["source"]] = key
            by_raw[e.get("raw_source")].append(key)
            by_msgs[json.dumps(key[1])].append(key[0])
    return by_src, by_raw, by_msgs


@functools.cache
def source_rows(run_dir: str) -> list[dict]:
    return [json.loads(x) for x in Path(f"data/{run_dir}/rows.jsonl").open()]


def matcher(name: str, cfg: str, stats: Counter):
    by_src, by_raw, by_msgs = env_episodes()
    rows = None
    if "rows" in SETS[name]:
        rows = [json.loads(x) for x in (SETS[name]["rows"] / f"{cfg}.jsonl").open()]

    def episode_of(e) -> str | None:
        if rows is None:
            cands = [by_src.get(f"data/{e.run_dir}:{int(e.row_index)}")]
            msgs = [text(m) for m in source_rows(e.run_dir)[int(e.row_index)]["messages"]]
        else:
            r = rows[int(e.row_index)]
            cands, msgs = by_raw.get(f"runs/{r['source']}", []), [text(m) for m in r["messages"]]
        hit = [ep for c in cands if c for ep, m in [c] if m == msgs]
        if len(hit) > 1:
            hit = [ep for ep in hit if f"/{SETS[name].get('prefer')}/" in ep]
            stats[f"{cfg}: by source, transcript in two env runs (took {SETS[name].get('prefer')})"] += len(hit) == 1
        if len(hit) == 1:
            stats[f"{cfg}: by source"] += 1
            return hit[0]
        exact = by_msgs.get(json.dumps(msgs), [])
        if len(exact) == 1:
            stats[f"{cfg}: by transcript"] += 1
            return exact[0]
        stats[f"{cfg}: UNMATCHED ({len(cands)} source candidates, {len(exact)} transcript matches)"] += 1
        print(f"{cfg}: unmatched {e.tid} {e.run_dir}:{e.row_index}", file=sys.stderr)
        return None

    return episode_of


def main(name: str):
    s = SETS[name]
    tok = Tokenizer.from_file(hf_hub_download("zai-org/GLM-5.3", "tokenizer.json"))
    srcs = sorted({f"runs/olens_reads/{d.name}/{c}" for d, c in s["configs"]})
    run = Run(
        f"{s['out']}/expview",
        title=s["title"],
        description=f"GLM 5.3 O-lens reads (Celeste's RL lens, block 51; SFT lens off by default) on over-eagerness transcripts "
        f"from {s['about']}. Each item is one transcript with every read inline, attached to its episode in `envs/overeagerness`, "
        "where the episode page shows them. **positive** is the source run's label. Parquet sources: " + ", ".join(f"`{x}`" for x in srcs),
        label_order=["positive", "aware"],
        readers=READERS,
        tokens_section="O-lens reads",
    )
    run.reset()
    stats = Counter()
    for d, cfg in s["configs"]:
        stats.update(import_config(d.name, cfg, d / cfg, ["readouts.parquet", "readouts_dense.parquet"], Path(run.path), tok, {}, Calls([]),
                                   "our", run=run, episode_of=matcher(name, cfg, stats)))  # fmt: skip
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
