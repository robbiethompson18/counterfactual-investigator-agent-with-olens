"""Decapitated-network unroll on WorkspaceBench's six regex-scored multi-token families: Qwen3.6-27B cut after block L with
the J++ lens as its head, generating 32 tokens from the prompt's last token (scripts/decap_unroll.py, on a pod), scored
with WorkspaceBench's own regex contract (wsb/mt_regex.py; no judge call). Results in docs/wsb-decap-unroll.md.

    uv run python -m wsb.decap requests --slug jpp-unroll   # new run dir with requests.jsonl (598 prompts); prints the path
    # pod: python3 scripts/decap_unroll.py requests.jsonl readouts.jsonl; copy readouts.jsonl into the run dir
    uv run python -m wsb.decap score RUN                    # scores.jsonl + summary.json, prints the table, syncs the run to HF

Arms, each scored per family as "the item passes at any of the 11 cut layers" (their rule): the three decodes of the
unroll (`greedy`, `norepeat`, `sample`: one prose sample per cell), `any3` (the three as three samples) and `top10`, the
static J++ read at the same cell (each token one sample, no summarizer, so a multi-token form is out of reach by
construction). `full` is the control at no cut: Qwen itself continuing the prompt, one "layer". Beside them: the O-lens's
per-item verdicts (agu18dec/olens_ablations, same scorer) and WorkspaceBench's frozen prompt-only floors."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from counterfactual_investigator_agent_with_olens.runs import finish, new_run
from wsb import banks, mt_regex

FAMILIES = ("basic_readout_mt", "multihop_mt", "multilingual_mt", "typo_mt", "multilingual_multihop", "multilingual_typo")
DECODES = ("greedy", "norepeat", "sample")
ARMS = (*DECODES, "any3", "top10")
LENS_REPO, LENS_DIR = "agu18dec/olens_ablations", "ab4/offset/results/wsbench/rl.base.off_k0.own.i300"
# stock Qwen3.6-27B shown the prompt and asked what a model would be thinking at its last token, same regex scorer:
# camilablank/workspace-bench@86612fa evals/baselines/prompt_only.json
PROMPT_ONLY = {
    "basic_readout_mt": 0.59,
    "multihop_mt": 0.69,
    "multilingual_mt": 0.42,
    "typo_mt": 0.67,
    "multilingual_multihop": 0.449,
    "multilingual_typo": 0.66,
}


def samples(cell: dict, arm: str) -> list[str]:
    raw = [t for t, _ in cell["top10"]] if arm == "top10" else [cell[d]["text"] for d in (DECODES if arm == "any3" else (arm,))]
    return [s for s in map(mt_regex.extract_phrase, raw) if s]


def boot_ci(passes: list[bool]) -> list[float]:
    rng = random.Random(0)
    means = sorted(sum(rng.choices(passes, k=len(passes))) / len(passes) for _ in range(2000))
    return [round(means[50], 3), round(means[1949], 3)]


def score(run: Path):
    cells = defaultdict(dict)  # (family, id) -> layer -> cell
    for line in (run / "readouts.jsonl").read_text().splitlines():
        c = json.loads(line)
        cells[c["family"], c["id"]][c["layer"]] = c
    rows, summary = [], {}
    for family in FAMILIES:
        bank = banks.load(family)
        contract = mt_regex.contract_for(bank, family)
        lens_rows = json.loads(Path(hf_hub_download(LENS_REPO, f"{LENS_DIR}/{family}.json")).read_text())["rows"]
        lens = {r["id"]: r for r in lens_rows}
        fam_rows: list[dict] = []
        for it in bank["items"]:
            units = mt_regex.scored_units(it, contract)
            by_layer = cells[family, it["name"]]
            row: dict = {"family": family, "id": it["name"], "pass": {}, "layers": {}, "full": {}}
            for arm in ARMS:
                hits = {L: mt_regex.layer_unit_hits(samples(c, arm), units) for L, c in by_layer.items()}
                res = mt_regex.item_result({L: h for L, h in hits.items() if L != "full"}, units)
                row["pass"][arm], row["layers"][arm] = res["pass"], res["passing_layers"]
                row["full"][arm] = mt_regex.layer_passes(hits["full"], units)
            row["lens"] = lens[it["name"]]["pass"]
            row["lens_layers"] = lens[it["name"]]["passing_layers"]
            fam_rows.append(row)
        rows += fam_rows
        grid = sorted(L for L in next(iter(cells.values())) if L != "full")
        summary[family] = {
            "n": len(fam_rows),
            "prompt_only": PROMPT_ONLY[family],
            "olens": sum(r["lens"] for r in fam_rows) / len(fam_rows),
            **{
                arm: {
                    "pass": sum(r["pass"][arm] for r in fam_rows) / len(fam_rows),
                    "ci95": boot_ci([r["pass"][arm] for r in fam_rows]),
                    "full": sum(r["full"][arm] for r in fam_rows) / len(fam_rows),
                    "by_layer": {L: sum(L in r["layers"][arm] for r in fam_rows) / len(fam_rows) for L in grid},
                }
                for arm in ARMS
            },
        }
    (run / "scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (run / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"{'family':24}" + "".join(f"{a:>10}" for a in ARMS) + f"{'full':>8}{'O-lens':>8}{'prompt':>8}")
    for family, s in summary.items():
        line = f"{family:24}" + "".join(f"{s[a]['pass']:>10.2f}" for a in ARMS)
        print(line + f"{s['greedy']['full']:>8.2f}{s['olens']:>8.2f}{s['prompt_only']:>8.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("requests").add_argument("--slug", required=True)
    sub.add_parser("score").add_argument("run", type=Path)
    args = ap.parse_args()
    if args.cmd == "score":
        score(args.run)
        finish(args.run)
        return
    run = new_run(
        "wsb_decap_unroll",
        args.slug,
        description=__doc__,
        models=[
            {"name": "Qwen/Qwen3.6-27B", "served_by": "runpod pod, HF transformers (scripts/decap_unroll.py)"},
            {"name": "neuronpedia/jacobian-lens qwen3.6-27b/jpp", "served_by": "same pod: the J++ maps as the cut model's head"},
        ],
        params={"families": FAMILIES, "layers": list(range(20, 61, 4)), "steps": 32, "decodes": DECODES, "scorer": mt_regex.SCORER_VERSION},
    )
    reqs = [{"family": f, "id": it["name"], "prompt": it["prompt"]} for f in FAMILIES for it in banks.load(f)["items"]]
    (run / "requests.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reqs))
    print(run)


if __name__ == "__main__":
    main()
