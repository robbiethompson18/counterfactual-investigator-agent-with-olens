# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow"]
# ///
"""Rows and O-lens reads for cfbench over the alignment-faking env, thinking off: public -> preview for DeepSeek V4 Flash
and GLM 5.3 (cfbench/envs/af_rows.py).

    uv run --script scripts/cfbench_af_prep.py \
        --dsv4-rows <cia-af-cf>/data/runs/alignment_faking/2026-10-26_dsv4flash-nocot/rows.jsonl \
        --dsv4-requests <cia-af-cf>/data/dsv4_olens/af_nocot/requests.jsonl \
        --dsv4-reads <cia-af-cf>/data/dsv4_olens/full/reads_L30_vllm.jsonl \
        --glm-rows <cia-af>/data/runs/alignment_faking/2026-10-26_glm53-nocot/rows.jsonl \
        --glm-dense ~/workspace/tmp/glm_dense/2026-10-26_glm53_alignment_faking/2026-10-26_glm53-nocot \
        --placebo data/runs/cfbench_overeager/reads_email

The public arm is 100 runs, one per item: for DSV4 the runs its O-lens (Camila's inverter, layer 30, greedy + 2 samples, no
FVE) read; for GLM each item's first run (Celeste's verbalizer, block 51, greedy + 4 samples with FVE). Preview is every run.
Writes data/runs/cfbench_af/{dsv4,glm}/rows.jsonl and reads dirs reads_{dsv4,glm}/ plus placebo reads (the email
over-eagerness setting's GLM reads, an unrelated env) as reads_placebo_glm/ and, without FVE and cut to 3 bullets so they
look like the DSV4 reads, reads_placebo_dsv4/.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cfbench_overeager_prep import char_offset, messages, write_reads

OUT = Path("data/runs/cfbench_af")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def positive(r: dict) -> bool:
    return r["positive"] is True or r["positive"] == "True"


def write_rows(name: str, model: str, rows: list[tuple[int, dict]]) -> dict[int, list[dict]]:
    """rows: (row index in the run's rows.jsonl, row). Returns row index -> converted messages."""
    msgs, out = {}, []
    for i, r in rows:
        msgs[i] = messages(r["messages"], [])
        out.append(
            {
                "source": f"af_{name}/{r['arm']}:{r['item']}:{r['sample']}",
                "arm": r["arm"],
                "item": str(r["item"]),
                "model": model,
                "positive": positive(r),
                "messages": msgs[i],
                "files_before": {},
                "setup": [],
                "workdir": None,
                "af_row": i,
            }
        )
    (OUT / name).mkdir(parents=True, exist_ok=True)
    (OUT / name / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    print(
        f"{name}: "
        + ", ".join(
            f"{a} {sum(r['positive'] for r in out if r['arm'] == a)}/{sum(r['arm'] == a for r in out)}" for a in ("public", "preview")
        )
    )
    return msgs


def main():
    ap = argparse.ArgumentParser()
    for k in ("dsv4-rows", "dsv4-requests", "dsv4-reads", "glm-rows", "glm-dense", "placebo"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    a = ap.parse_args()

    # DSV4: public = the 100 runs the O-lens read (requests.jsonl row_index), preview = all
    dsv4 = load(a.dsv4_rows)
    read_rows = {int(r["row_index"]) for r in load(a.dsv4_requests) if r["arm"] == "public"}
    keep = [(i, r) for i, r in enumerate(dsv4) if r["arm"] == "preview" or i in read_rows]
    msgs = write_rows("dsv4", "deepseek/deepseek-v4-flash", keep)
    src = {f"public:{r['item']}": (i, f"af_dsv4/public:{r['item']}:{r['sample']}") for i, r in keep if r["arm"] == "public"}
    d = OUT / "reads_dsv4"
    d.mkdir(exist_ok=True)
    with (d / "requests.jsonl").open("w") as f:
        for i, s in src.values():
            f.write(json.dumps({"id": s, "messages": msgs[i]}) + "\n")
    n = placed = 0
    with (d / "reads.jsonl").open("w") as f:
        for r in load(a.dsv4_reads):
            i, s = src[r["id"]]
            m = msgs[i]
            k = next(j for j, x in enumerate(m) if x["role"] == r["role"])
            offset = char_offset(m[k]["content"], r["ctx_left"], r["token"])
            n, placed = n + 1, placed + (offset is not None)
            bullets = [{"text": r[b], "fve": None, "coef": None} for b in ("greedy", "s1", "s2") if r.get(b)]
            f.write(
                json.dumps(
                    {
                        "id": s,
                        "token_index": r["pos"],
                        "token": r["token"],
                        "prefix": r["ctx_left"],
                        "msg_index": k,
                        "role": r["role"],
                        "field": "content",
                        "char_offset": offset,
                        "layer": r["layer"],
                        "bullets": bullets,
                        "joint_fve": None,
                    }
                )
                + "\n"
            )
    print(f"dsv4 reads: {n}, {placed} placed")

    # GLM: public = sample 0 of each item, preview = all
    glm = load(a.glm_rows)
    first = {}  # item -> its first public run
    for i, r in enumerate(glm):
        if r["arm"] == "public":
            first.setdefault(r["item"], i)
    keep = [(i, r) for i, r in enumerate(glm) if r["arm"] == "preview" or i in first.values()]
    msgs = write_rows("glm", "z-ai/glm-5.3", keep)
    public = {i: r for i, r in keep if r["arm"] == "public"}
    d = OUT / "reads_glm"
    d.mkdir(exist_ok=True)
    source = lambda i: f"af_glm/public:{glm[i]['item']}:{glm[i]['sample']}" if i in public else f"af_glm/unused:{i}"
    all_msgs = {i: msgs.get(i) or messages(r["messages"], []) for i, r in enumerate(glm)}
    with (d / "requests.jsonl").open("w") as f:
        for i in public:
            f.write(json.dumps({"id": source(i), "messages": all_msgs[i]}) + "\n")
    tmp = d / "reads_all.jsonl"
    with tmp.open("w") as f:
        write_reads(a.glm_dense, all_msgs, source, f)
    with tmp.open() as f, (d / "reads.jsonl").open("w") as out:
        kept = 0
        for line in f:
            if not json.loads(line)["id"].startswith("af_glm/unused:"):
                out.write(line)
                kept += 1
    tmp.unlink()
    print(f"glm reads: {kept} on the 100 public runs")

    # placebo reads: an unrelated env (email over-eagerness, GLM verbalizer)
    d = OUT / "reads_placebo_glm"
    d.mkdir(exist_ok=True)
    for name in ("reads.jsonl", "requests.jsonl"):
        shutil.copy(a.placebo / name, d / name)
    d = OUT / "reads_placebo_dsv4"
    d.mkdir(exist_ok=True)
    shutil.copy(a.placebo / "requests.jsonl", d / "requests.jsonl")
    with (a.placebo / "reads.jsonl").open() as f, (d / "reads.jsonl").open("w") as out:
        for line in f:
            r = json.loads(line)
            r["bullets"] = [{**b, "fve": None} for b in r["bullets"][:3]]
            out.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
