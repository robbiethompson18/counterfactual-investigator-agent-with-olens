# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow"]
# ///
"""A placebo from an unrelated env for the over-eagerness cfbench runs: the six settings are near-identical tasks, so the
default placebo (another setting's reads) still reads about an ambiguous request. This writes

- data/runs/cfbench_overeager/reads_<setting>/: that setting's reads alone, split from reads/ (made by
  scripts/cfbench_overeager_prep.py);
- data/runs/cfbench_overeager/reads_af/: GLM verbalizer reads of the alignment-faking transcripts (GLM 5.3, thinking on,
  free/paid/preview/public, 800 episodes; rows from runs/alignment_faking/2026-10-26_glm53-think), sources
  `alignment_faking/...`.

Pass `--olens-reads reads_<setting>/reads.jsonl,reads_af/reads.jsonl` and the placebo pool is the AF runs only.

    uv run --script scripts/cfbench_overeager_af_placebo.py ~/workspace/tmp/glm_dense \
        data/runs/alignment_faking/2026-10-26_glm53-think/rows.jsonl
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cfbench_overeager_prep import OUT, SETTINGS, family, messages, write_reads


def main(root: Path, af_rows: Path):
    outs = {}
    for setting in SETTINGS:
        d = OUT / f"reads_{setting}"
        d.mkdir(exist_ok=True)
        outs[f"{family(setting)}/{setting}:"] = (d / "reads.jsonl").open("w"), (d / "requests.jsonl").open("w")
    for path, k in [("reads.jsonl", 0), ("requests.jsonl", 1)]:
        with (OUT / "reads" / path).open() as f:
            for line in f:
                src = json.loads(line)["id"]
                outs[src[: src.index(":") + 1]][k].write(line)
    for files in outs.values():
        for f in files:
            f.close()

    d = OUT / "reads_af"
    d.mkdir(exist_ok=True)
    rows = [json.loads(line) for line in af_rows.read_text().splitlines()]
    msgs = {i: messages(r["messages"], r.get("tools") or []) for i, r in enumerate(rows)}
    source = lambda i: f"alignment_faking/{rows[i]['arm']}:{i}"
    with (d / "requests.jsonl").open("w") as f:
        for i, m in msgs.items():
            f.write(json.dumps({"id": source(i), "messages": m}) + "\n")
    with (d / "reads.jsonl").open("w") as f:
        n, k, _, _ = write_reads(root / "2026-10-26_glm53_alignment_faking" / "2026-10-26_glm53-think", msgs, source, f)
    print(f"alignment_faking: {len(rows)} rows, {n} reads, {k} placed")


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser())
