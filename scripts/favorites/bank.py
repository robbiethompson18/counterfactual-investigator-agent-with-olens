"""Build the favorites_qa bank: merge the question writers' JSONL (BRIEF.md), gate it on Qwen, attach the gate, upload.

    uv run python scripts/favorites/bank.py merge OUT.json DIR/*.jsonl          # validate, drop repeated categories
    uv run python scripts/cutoff_gate.py sample OUT.json --slug favorites-v1   # 8 samples at T 1 + greedy, thinking off
    uv run python scripts/favorites/bank.py build data/runs/cutoff_gate/<date>_favorites-v1 [--version v1] [--upload]

`build` writes data/banks/favorites_qa/<version>.json (every question with the gate's `qwen_modal`, `qwen_modal_count`,
`qwen_greedy`, `gate_class`, `gate_run`) and with --upload puts it on HF at banks/favorites_qa/<version>.json.
"""

import argparse
import collections
import json
from pathlib import Path

KEYS = {"id", "domain", "category", "shape", "question"}
SHAPES = {"favorite", "least_favorite", "pick_one", "recommend", "be_one", "name_one", "choose_for"}
BANK_DIR = Path("data/banks/favorites_qa")


def merge(out: Path, files: list[Path]):
    rows = [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]
    for r in rows:
        assert set(r) == KEYS and r["shape"] in SHAPES and r["question"].rstrip().endswith("."), r
    seen, keep = set(), []
    for r in sorted(rows, key=lambda r: r["id"]):  # a category asked by two writers: the first id keeps it
        if r["category"] in seen:
            print("drop repeated category:", r["id"], r["category"])
            continue
        seen.add(r["category"])
        keep.append(r)
    assert len({r["id"] for r in keep}) == len(keep)
    out.write_text(json.dumps(keep, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(keep)} questions ({len(rows) - len(keep)} dropped) -> {out}")
    print(collections.Counter(r["shape"] for r in keep))


def build(run: Path, version: str, upload: bool):
    questions = json.loads((run / "questions.json").read_text())
    gate = {r["id"]: r for r in json.loads((run / "summary.json").read_text())["questions"]}
    # qwen_modal is a raw sample whose gate normal form is the mode, so consumers normalize it with their own norm (the gate's
    # turns "chef's" into "chef s", cutoff_qa's norm into "chefs")
    raw = {}
    for r in map(json.loads, (run / "samples.jsonl").read_text().splitlines()):
        raw[r["id"]] = next(t.strip() for t, n in zip(r["samples"], r["samples_norm"], strict=True) if n == gate[r["id"]]["modal"])
    fields = lambda g: {"qwen_modal_count": g["modal_count"], "qwen_greedy": g["greedy"], "gate_class": g["class"]}
    items = [q | {"qwen_modal": raw[q["id"]]} | fields(gate[q["id"]]) | {"gate_run": f"cutoff_gate/{run.name}"} for q in questions]
    print(collections.Counter(it["gate_class"] for it in items))
    bank = {"version": version, "description": __doc__.split("\n\n")[0], "brief": "scripts/favorites/BRIEF.md", "items": items}
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    path = BANK_DIR / f"{version}.json"
    path.write_text(json.dumps(bank, indent=1, ensure_ascii=False) + "\n")
    print("wrote", path)
    if upload:
        from huggingface_hub import HfApi

        from counterfactual_investigator_agent_with_olens.runs import REPO

        HfApi().upload_file(path_or_fileobj=path, path_in_repo=f"banks/favorites_qa/{version}.json", repo_id=REPO, repo_type="dataset")
        print("uploaded", f"banks/favorites_qa/{version}.json")


ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("merge")
p.add_argument("out", type=Path)
p.add_argument("files", nargs="+", type=Path)
p = sub.add_parser("build")
p.add_argument("run", type=Path)
p.add_argument("--version", default="v1")
p.add_argument("--upload", action="store_true")
args = ap.parse_args()
if args.cmd == "merge":
    merge(args.out, args.files)
else:
    build(args.run, args.version, args.upload)
