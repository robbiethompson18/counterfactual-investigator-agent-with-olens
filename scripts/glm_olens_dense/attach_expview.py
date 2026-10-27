# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2", "pyarrow>=17", "tokenizers>=0.20", "huggingface_hub>=0.25"]
# ///
"""Attach a prep_envs.py fill upload to its env episodes: one expview run per env, data/runs/<env>/<date>_<slug>/expview, whose
items point at envs/<env>/<run>/<episode id> (the env page stitches the reads into each episode).

    uv run --script scripts/glm_olens_dense/attach_expview.py --upload data/runs/olens_reads/2026-10-27_glm53_coverage_fill \
        --slug glm53-olens-fill
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

sys.path[:0] = ["expview"]
from importers.celeste import READERS, Calls, import_config  # ty: ignore[unresolved-import]

from expview import Run  # ty: ignore[unresolved-import]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", required=True, help="the fill upload: <upload>/<env>__<run>/ config dirs")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--date", default="2026-10-27")
    a = ap.parse_args()
    tok = Tokenizer.from_file(hf_hub_download("zai-org/GLM-5.3", "tokenizer.json"))
    up = Path(a.upload)
    by_env = defaultdict(list)
    for d in sorted(p for p in up.iterdir() if (p / "readouts_dense.parquet").exists()):
        by_env[d.name.split("__")[0]].append(d)
    for env, dirs in by_env.items():
        out = Path(f"data/runs/{env}/{a.date}_{a.slug}")
        run = Run(
            out / "expview",
            title=f"{env}: GLM 5.3 O-lens reads, coverage fill ({a.date})",
            description=f"GLM 5.3 O-lens reads (Celeste's RL lens, block 51) on every GLM 5.3 `envs/{env}` run, from `runs/olens_reads/"
            f"{up.name}`: every chat-template/special token and the head's punctuation, plus later-turn punctuation where the "
            "policy in scripts/glm_olens_dense/prep_envs.py says so. Positions already read elsewhere are copied in (`kind=copied`), "
            "so each item shows all of its episode's reads. Each item is attached to its env episode; labels live on the episode.",
            label_order=["positive"],
            readers=READERS,
            tokens_section="O-lens reads",
        )
        run.reset()
        stats = Counter()
        for d in dirs:
            env_run = "envs/" + d.name.replace("__", "/", 1)
            eid = pd.read_parquet(d / "episodes.parquet", columns=["tid", "episode_id"]).set_index("tid").episode_id.to_dict()
            stats.update(
                import_config(
                    up.name,
                    d.name,
                    d,
                    ["readouts.parquet", "readouts_dense.parquet"],
                    out / "expview",
                    tok,
                    {},
                    Calls([]),
                    "our",
                    run=run,
                    episode_of=lambda e, r=env_run, m=eid: f"{r}/{m[e.tid]}",
                )
            )
        print(env, json.dumps(stats))


if __name__ == "__main__":
    main()
