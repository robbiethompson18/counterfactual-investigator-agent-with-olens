"""Re-score saved cutoff_qa predictions after the label fix (environments/cutoff_qa/env.py `gate_label`, 2026-10-31): no new
model calls. Labels were norm() of the gate's already-normalized modal string, so an answer with an apostrophe ("N'Djamena")
had label "n djamena" while the predictor's identical answer canonicalized to "ndjamena" and was marked wrong.

For each inputs dir (cfbench_evals/*cutoff*: `modal` rebuilt from the gate's raw samples; menus-inline-inputs: cutoff rows take
the 1k inputs' fixed labels) and each cfbench run on those inputs: rewrite every cutoff_qa row's `modal_cf`, `answers_cf`,
`answers_base`, `correct` and the submit_answer verdict message, note the regrade in meta.json, print per-condition accuracy
before and after. --sync re-uploads the changed dirs to HF (the Docent mirrors keep the old verdict text).

    uv run python -m scripts.regrade_cutoff_labels [--sync]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from cfbench import answer
from counterfactual_investigator_agent_with_olens.runs import now, upload
from environments._lib.core import load_env

RUNS = Path("data/runs")
INPUTS_1K = "cfbench_evals/2026-10-29_cutoff-qa-inline-inputs-1k"
MENUS_INPUTS = "cfbench_evals/2026-10-30_menus-inline-inputs"


def fix_inputs(mod, inputs: str, fixed_1k: dict[str, str]) -> tuple[dict[str, str], bool]:
    """Rewrite cases.jsonl's labels; item -> fixed label for this inputs dir, and whether any changed."""
    path = RUNS / inputs / "cases.jsonl"
    cases = [json.loads(line) for line in path.read_text().splitlines()]
    labels, changed = {}, 0
    for c in cases:
        if "set" in c:  # menus inputs
            if c["set"] != "cutoff":
                continue
            new = fixed_1k[c["item"]]
        else:
            new = mod.gate_label(c["meta"]["gate_run"], c["item"], c["meta"]["qwen_modal"])
        changed += new != c["modal"]
        c["modal"] = labels[c["item"]] = new
    if changed:
        path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))
    print(f"{inputs}: {changed} of {len(labels)} cutoff labels changed")
    return labels, bool(changed)


def regrade(run: Path, labels: dict[str, str]) -> bool:
    """Stream predictions.jsonl (the tool-v3 file is 38k rows with full transcripts, too big to hold) into a .tmp, then swap."""
    path = RUNS / run / "predictions.jsonl"
    tmp = path.with_suffix(".tmp")
    before, after, flipped, relabelled = defaultdict(list), defaultdict(list), 0, set()
    with path.open() as src, tmp.open("w") as dst:
        for line in src:
            r = json.loads(line)
            if r["env_name"] == "cutoff_qa":
                new = labels[r["item"]]
                if new != r["modal_cf"]:
                    relabelled.add(r["item"])
                    r["answers_cf"] = {new: sum(r["answers_cf"].values())}
                    if r["answers_base"]:
                        r["answers_base"] = {new: sum(r["answers_base"].values())}
                    r["modal_cf"] = new
                if r["valid"]:
                    correct = r["canon"] == new
                    before[r["condition"]].append(bool(r["correct"]))
                    after[r["condition"]].append(correct)
                    if correct != r["correct"]:
                        flipped += 1
                        r["correct"] = correct
                        shim = SimpleNamespace(canon_answer=lambda _t, c=r["canon"]: c)
                        verdicts = [m for m in r["messages"] if str(m.get("content")).startswith("Answer recorded:")]
                        if verdicts:  # the submit_answer tool result (cfbench/predict.py grader), the last message
                            verdicts[-1]["content"] = answer.verdict(shim, r, r["answer"])
            dst.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{run}: {len(relabelled)} items relabelled, {flipped} draws flipped")
    for cond in before:
        b, a = sum(before[cond]) / len(before[cond]) * 100, sum(after[cond]) / len(after[cond]) * 100
        print(f"  {cond:28} {b:6.2f}% -> {a:6.2f}%  ({a - b:+.2f}pp, {len(after[cond])} draws)")
    if not flipped and not relabelled:
        tmp.unlink()
        return False
    tmp.replace(path)
    meta_path = RUNS / run / "meta.json"
    meta = json.loads(meta_path.read_text())
    why = "cutoff label norm fix (environments/cutoff_qa/env.py gate_label)"
    meta["regrade"] = {"at": now(), "why": why, "items_relabelled": len(relabelled), "draws_flipped": flipped}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sync", action="store_true", help="re-upload every changed inputs dir and run to HF")
    args = ap.parse_args()
    mod = load_env("cutoff_qa")
    cutoff_inputs = sorted(p.parent.relative_to(RUNS).as_posix() for p in (RUNS / "cfbench_evals").glob("*cutoff*/cases.jsonl"))
    labels, changed = {}, []
    for inputs in [*cutoff_inputs, MENUS_INPUTS]:
        labels[inputs], did = fix_inputs(mod, inputs, labels.get(INPUTS_1K, {}))
        if did:
            changed.append(RUNS / inputs)
    for meta_path in sorted((RUNS / "cfbench").glob("*/meta.json")):
        inputs = json.load(meta_path.open())["params"].get("inputs")
        if not (meta_path.parent / "predictions.jsonl").exists():  # still running or abandoned
            continue
        if inputs in labels and regrade(meta_path.parent.relative_to(RUNS), labels[inputs]):
            changed.append(meta_path.parent)
    print(f"\n{len(changed)} dirs changed")
    if args.sync:
        for d in changed:
            upload(d)
    else:
        print("rerun with --sync to upload:", " ".join(str(d) for d in changed))


if __name__ == "__main__":
    main()
