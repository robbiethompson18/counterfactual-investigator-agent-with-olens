"""Build the fixed populations, grouped folds and capture shards for docs/standard-probe-protocol.md.

    uv run --with transformers --with jinja2 python -m probes.prepare

No generations or fitting: labels reuse the existing census. Data and cached essay activations live in the tracked run.
"""

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from cfbench.evals.mcq_unselected import eval as mmlu
from cfbench.evals.supergpqa_inline import eval as supergpqa
from counterfactual_investigator_agent_with_olens.runs import REPO, new_run

MODEL = "Qwen/Qwen3.6-27B"
REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
LAYERS = [32, 40, 48, 52, 56, 60]
ESSAY_INPUTS = "runs/cfbench_evals/2026-10-30_essay-grading-inline-inputs"
ESSAY_CACHE = "runs/essay_probe/2026-10-31_essay-grading-acts"


def write_jsonl(path: Path, rows: list[dict]):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def stem_group(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    return "question:" + hashlib.sha256(normalized.encode()).hexdigest()


def boundary_sites(tokenizer, ids: list[int]) -> dict:
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    ends = [i for i, token in enumerate(ids) if token == end_id]
    assert ends and ends[-1] > 0
    positions = {"user_end": [ends[-1] - 1], "response_start": [len(ids) - 1]}
    suffix = tokenizer.decode(ids[ends[-1] :])
    assert "<think>" in suffix and "</think>" in suffix, suffix
    return positions


def capture_row(tokenizer, row: dict, messages: list[dict]) -> dict:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    sites = boundary_sites(tokenizer, ids)
    row["positions"] = {site: pos[0] for site, pos in sites.items()}
    row["boundary_tokens"] = {site: tokenizer.decode([ids[pos[0]]]) for site, pos in sites.items()}
    row["prompt_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    row["n_tokens"] = len(ids)
    cand = [tokenizer.encode(c, add_special_tokens=False)[0] for c in "ABCDEFGHIJ"[: row["n_options"]]]
    return {"id": row["id"], "ids": ids, "sites": sites, "cand": cand, "row_index": row["row_index"]}


def mcq_rows(tokenizer) -> tuple[list[dict], list[dict], dict]:
    manifest, captures, inventory = [], [], {}
    for family, ev in (("mmlu", mmlu), ("supergpqa", supergpqa), ("hinted", mmlu)):
        table, counts = ev._mod()._table(), ev.census()
        n_right = 0
        for item in ev._mod().items():
            answers = counts.get(item, Counter())
            letters = Counter({a: n for a, n in answers.items() if a})
            if not letters:
                continue
            modal, k = letters.most_common(1)[0]
            n = sum(answers.values())
            if k / n < 7 / 8:
                continue
            assert n == 8, (family, item, n)
            source = table.loc[item]
            gold = source["answer_letter"] if family == "supergpqa" else source["answer"]
            n_right += modal == gold
            arm = "base" if family == "supergpqa" else "hint+d1" if family == "hinted" else "nohint"
            row = {
                "id": f"{family}/{item}",
                "family": family,
                "item": item,
                "source": "supergpqa" if family == "supergpqa" else "mmlu",
                "arm": arm,
                "group": stem_group(source.question),
                "label": ord(modal) - ord("A"),
                "gold": ord(gold) - ord("A"),
                "n_options": len(source.options),
                "right": modal == gold,
                "modal_count": k,
                "response_count": n,
                "row_index": len(manifest),
                "activation_index": len(captures),
            }
            assert 0 <= row["label"] < row["n_options"] and 0 <= row["gold"] < row["n_options"]
            captures.append(capture_row(tokenizer, row, ev._mod().build(arm, item).messages))
            manifest.append(row)
        inventory[family] = {"questions": sum(r["family"] == family for r in manifest), "right": n_right}
        inventory[family]["wrong"] = inventory[family]["questions"] - n_right
        print(f"{family}: {inventory[family]}", flush=True)
    assert {f: x["questions"] for f, x in inventory.items()} == {"mmlu": 7943, "supergpqa": 6328, "hinted": 7943}
    return manifest, captures, inventory


def download(source: str, target: Path, revision: str):
    target.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(REPO, source, repo_type="dataset", revision=revision)
    shutil.copyfile(cached, target)


def essay_rows(run: Path, tokenizer, offset: int, revision: str) -> tuple[list[dict], dict]:
    folder = run / "essay_sources"
    for name in ("cases.jsonl", "layout.jsonl", "meta.json"):
        download(f"{ESSAY_INPUTS}/{name}", folder / name, revision)
    for name, output in (("q.npy", "user_end.npy"), ("last.npy", "response_start.npy"), ("layers.json", "layers.json")):
        download(f"{ESSAY_CACHE}/{name}", run / "essay_activations" / output, revision)
    download(f"{ESSAY_CACHE}/meta.json", folder / "activation_meta.json", revision)
    assert json.loads((run / "essay_activations/layers.json").read_text()) == LAYERS
    cases = {(r["arm"], r["item"]): r for r in map(json.loads, (folder / "cases.jsonl").open())}
    layout = list(map(json.loads, (folder / "layout.jsonl").open()))
    mean = lambda answers: sum(float(v) * n for v, n in answers.items()) / sum(answers.values())
    rows, saved_inputs = [], []
    for index, source in enumerate(layout):
        arm, item = source["id"].split("/", 1)[1].split("~")
        if arm == "base":
            continue
        case, plain = cases[arm, item], cases["base", item]
        observed, label = mean(case["answers"]), mean(plain["answers"])
        ids = tokenizer.encode(source["text"], add_special_tokens=False)
        sites = boundary_sites(tokenizer, ids)
        assert max(p[0] for p in source["pos"]) == len(ids) - 1, source["id"]
        end_position = sites["user_end"][0] + 1
        end_layout = next(p for p in source["pos"] if p[0] == end_position)
        assert source["text"][end_layout[1] : end_layout[2]] == "<|im_end|>", source["id"]
        saved_inputs.append({"id": source["id"], "ids": ids, "sites": sites, "activation_index": index})
        rows.append(
            {
                "id": source["id"],
                "family": "essay",
                "item": item,
                "source": "essay_grading",
                "arm": arm,
                "group": f"essay:{item}",
                "label": label,
                "observed": observed,
                "truth": label > observed,
                "response_count": sum(case["answers"].values()),
                "plain_response_count": sum(plain["answers"].values()),
                "row_index": offset + len(rows),
                "activation_index": index,
                "positions": {site: pos[0] for site, pos in sites.items()},
                "boundary_tokens": {site: tokenizer.decode([ids[pos[0]]]) for site, pos in sites.items()},
                "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                "n_tokens": len(ids),
            }
        )
    assert len(rows) == 1800 and len(layout) == 2004
    write_jsonl(folder / "hs_inputs.jsonl", saved_inputs)
    return rows, {"cases": len(rows), "essays": len({r["group"] for r in rows}), "cache_rows": len(layout)}


def assign_folds(rows: list[dict]) -> dict:
    # Balance group counts, not answer labels. Shared stems across benchmarks and hints always share a fold.
    folds = {}
    for namespace in ("question:", "essay:"):
        groups = sorted({r["group"] for r in rows if r["group"].startswith(namespace)})
        random.Random(0).shuffle(groups)
        folds.update({g: i % 2 for i, g in enumerate(groups)})
    for row in rows:
        row["outer_fold"] = folds[row["group"]]
    return folds


def shard(run: Path, captures: list[dict], rows: list[dict], n_shards: int):
    # Greedy length balancing avoids one shard inheriting all long prompts; row mapping is explicit.
    groups: list[list[int]] = [[] for _ in range(n_shards)]
    loads = [0] * n_shards
    for index in sorted(range(len(captures)), key=lambda i: (-len(captures[i]["ids"]), i)):
        target = min(range(n_shards), key=lambda s: (loads[s], s))
        groups[target].append(index)
        loads[target] += len(captures[index]["ids"])
    for s, indices in enumerate(groups):
        path = run / "shards" / str(s)
        path.mkdir(parents=True, exist_ok=True)
        indices.sort()
        for j, index in enumerate(indices):
            rows[index]["capture_shard"] = s
            rows[index]["shard_index"] = j
            captures[index]["activation_index"] = index
        write_jsonl(path / "hs_inputs.jsonl", [captures[i] for i in indices])
        write_jsonl(path / "manifest.jsonl", [rows[i] for i in indices])
    return [{"shard": i, "rows": len(g), "tokens": loads[i]} for i, g in enumerate(groups)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Resume preparation in an existing tracked run")
    parser.add_argument("--shards", type=int, default=6)
    args = parser.parse_args()
    revision = HfApi().dataset_info(REPO).sha
    assert revision is not None
    run = args.run or new_run(
        "standard_probes",
        "nested2-v1",
        "Standard probes: nested two-fold, two positions, six layers, capped matched training",
        models=[{"name": MODEL, "revision": REVISION, "served_by": "RunPod HF transformers bf16"}],
        params={
            "layers": LAYERS,
            "sites": ["user_end", "response_start"],
            "outer_folds": 2,
            "inner_folds": 2,
            "seed": 0,
            "train_cap": 1000,
            "mcq_lambda": 1,
            "ridge_lambda": 1,
            "source_dataset_revision": revision,
        },
    )
    print(f"RUN={run}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    rows, captures, inventory = mcq_rows(tokenizer)
    assign_folds(rows)
    inventory["shards"] = shard(run, captures, rows, args.shards)
    write_jsonl(run / "manifest.jsonl", rows)
    write_jsonl(run / "hs_inputs.jsonl", captures)
    print("MCQ capture shards ready", flush=True)
    essays, inventory["essay"] = essay_rows(run, tokenizer, len(rows), revision)
    rows.extend(essays)
    folds = assign_folds(rows)
    write_jsonl(run / "manifest.jsonl", rows)
    (run / "folds.json").write_text(json.dumps(folds, indent=2) + "\n")
    (run / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    provenance = {
        "model": MODEL,
        "model_revision": REVISION,
        "source_dataset_revision": revision,
        "mmlu_labels": {r: mmlu.LABELS for r in mmlu.CENSUS},
        "supergpqa_labels": supergpqa.CENSUS,
        "essay_inputs": ESSAY_INPUTS,
        "essay_cache": ESSAY_CACHE,
        "essay_cache_model_revision": None,
        "essay_cache_caveat": "Historical cache identifies Qwen3.6-27B but did not record its HF checkpoint revision.",
        "consistency": "modal bare letter >=7/8 of all saved samples; all accepted cases have exactly8samples",
        "hint_arm": "hint+d1",
        "grouping": "sha256 of whitespace-collapsed casefolded question stem; essay id for essays",
        "excluded": ["inconsistent MCQs", "SuperGPQA calculation questions excluded by original sampling"],
        "python_version": sys.version,
        "preparation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    sources = [mmlu.LOCAL / "hint_mcq" / r / "labels" / mmlu.LABELS for r in mmlu.CENSUS]
    sources += [supergpqa.LOCAL / "supergpqa" / r / "labels" / name for r, name in supergpqa.CENSUS.items()]
    sources += [mmlu._mod().PARQUET, supergpqa._mod().JSONL]
    provenance["local_source_sha256"] = {
        str(path.relative_to(mmlu.REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
    }
    shutil.copyfile(__file__, run / "prepare_source.py")
    (run / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(inventory, indent=2), flush=True)


if __name__ == "__main__":
    main()
