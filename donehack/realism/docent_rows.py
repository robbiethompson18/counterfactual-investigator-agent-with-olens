# ruff: noqa: E501
"""Build a Docent/evalcue-shaped run dir (rows.jsonl + meta.json) from realism episodes + judge files.

  uv run python realism/docent_rows.py --out <monorepo>/data/runs/coding/2026-10-24_glm53-realism [--exclude-from <older run dir>]

One row per episode: arm = variant, item = problem id, sample = seed; flags (hack, tests_modified, orig_pass, end, judge
verdicts) become Docent metadata; files_before = workspace the model started with, files_after = that plus the final
solution / edited tests. --exclude-from skips episodes already in an earlier uploaded dir (Docent run ids are not
stable across re-uploads, so new episodes go in a new dir).
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import realvariants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-from", action="append", default=[])
    ap.add_argument("--description", default="")
    a = ap.parse_args()
    seen = set()
    for d in a.exclude_from:
        for line in (Path(d) / "rows.jsonl").read_text().splitlines():
            r = json.loads(line)
            seen.add((r["arm"], r["item"], str(r["sample"])))
    rd = HERE / "runs" / a.run
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out / "rows.jsonl", "w") as fo:
        for f in sorted(rd.glob("*/*.json")):
            if f.parent.name == "judge":
                continue
            d = json.loads(f.read_text())
            r, spec = d["record"], d["spec"]
            key = (r["variant"], r["pid"], str(r["seed"]))
            if key in seen:
                continue
            jf = rd / "judge" / f.parent.name / f.name
            jp = json.loads(jf.read_text())["parsed"] if jf.exists() else {}
            before = dict(spec["files"])
            after = dict(before)
            after[spec["solution_path"]] = r.get("solution") or ""
            if r.get("tests_modified") and r.get("final_tests") is not None:
                after[spec["test_path"]] = r["final_tests"]
            row = {
                "env": "coding",
                "model": r["model"],
                "arm": r["variant"],
                "item": r["pid"],
                "sample": str(r["seed"]),
                "seed": str(r["seed"]),
                "messages": d["messages"],
                "files_before": before,
                "files_after": after,
                "meta": {"atoms": realvariants.VARIANT_ATOMS.get(r["variant"]), "workdir": spec["workdir"]},
                "harness": "realism",
                "hack": bool(r["hack"]),
                "tests_modified": bool(r["tests_modified"]),
                "passes_original_tests": bool(r.get("orig_pass")),
                "impossible_type": r.get("impossible_type"),
                "end": r["end"],
                "steps": r["steps"],
                "cost": r["cost"],
                **{f"judge_{k}": v for k, v in jp.items() if k != "summary"},
                "judge_summary": jp.get("summary"),
            }
            fo.write(json.dumps(row) + "\n")
            n += 1
    (out / "meta.json").write_text(
        json.dumps(
            {
                "experiment": "coding",
                "status": "done",
                "machine": "bramble",
                "description": a.description
                or "GLM 5.3 on ImpossibleBench (32 problems x 2 seeds) + 2 EvilGenie tasks, realism variants "
                "(arm = variant: round 1 realism atoms, c_* RL cues on core, u_* user evidence on base, core+x/all-x/core-x ablations, nc_* "
                "RL cues on the no-criterion base). hack = final solution passes the impossible tests; tests_modified = edited the test file.",
                "git": {
                    "repo": "robbiethompson18/counterfactual-investigator-agent-with-olens",
                    "branch": "claude/realism-variants",
                    "path": "donehack/realism/",
                },
                "models": [{"name": "z-ai/glm-5.3", "served_by": "OpenRouter"}],
                "judge": {"model": "deepseek/deepseek-v4.1-flash-1017"},
            },
            indent=1,
        )
    )
    print(f"{n} rows -> {out}")


if __name__ == "__main__":
    main()
