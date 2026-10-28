"""Import the pre-registered test's old predictions (HF runs/cfbench/2026-10-23_grade-big*, 2026-10-26_grade-sol-*) as
essay_grading eval runs, one per predictor group, so the eval viewer shows them without a paid rerun.

Only bookkeeping changes: `env` -> eval:essay_grading/<version>, `version`, `env_name`, `task`, `condition` through
canon_condition, `shown` from hidden_goal_grade/<run>:<line> to essay_grading/<run>:<episode id>, `imported_from`. The
rows keep their own harness_version (1 for Flash, 6 for Sol), so they are not comparable to new runs. Docent ids are
carried over from each old run's docent.json.

    uv run python -m cfbench.evals.essay_grading.import_old data/runs/cfbench_evals/2026-10-28_essay-grading-inputs
"""

import json
import sys
from collections import Counter
from pathlib import Path

from cfbench.core import canon_condition
from cfbench.evals.essay_grading.eval import ENV, ENVS, VERSIONS
from counterfactual_investigator_agent_with_olens.runs import ROOT, finish, new_run
from environments._lib.runfolder import LOCAL, read_episodes

OLD = ROOT / "cfbench"
GROUPS = {
    "essay-grading-flash-v1": [f"2026-10-23_grade-{b}-{v}" for b in ("big", "big2") for v in VERSIONS],
    "essay-grading-sol-v6": [f"2026-10-26_grade-sol-{v}" for v in VERSIONS],
}


def main(inputs: Path):
    rename = {
        ep["source"].removeprefix("data/runs/"): f"{ENV}/{s['run']}:{ep['id']}"
        for s in VERSIONS.values()
        for ep in read_episodes(LOCAL / ENV / s["run"])
    }
    for slug, olds in GROUPS.items():
        rows, docent = [], []
        for old in olds:
            version = old.rsplit("-", 1)[1]
            ids = json.loads((OLD / old / "docent.json").read_text())["runs"]
            lines = (OLD / old / "predictions.jsonl").read_text().splitlines()
            assert len(ids) == len(lines), old
            for r in map(json.loads, lines):
                r |= {"env": f"eval:{ENV}/{version}", "version": version, "env_name": ENV, "task": r.get("task", "rate")}
                r |= {"condition": canon_condition(r["condition"]), "harness_version": r.get("harness_version", 1)}
                r |= {"shown": [rename[s] for s in r["shown"]], "imported_from": f"runs/cfbench/{old}"}
                rows.append(r)
            docent += ids
        keys = Counter((r["version"], r["base_arm"], r["cf_arm"], r["condition"], r["predictor"], r["draw"]) for r in rows)
        assert max(keys.values()) == 1, [k for k, n in keys.items() if n > 1]
        cases = {k[:3] for k in keys}
        conditions = sorted({r["condition"] for r in rows})
        draws = Counter(len([k for k in keys if k[:3] == c and k[3] == cond]) for c in cases for cond in conditions)
        print(f"{slug}: {len(rows)} rows, {len(cases)} cases x {conditions}, draws per case x condition {dict(draws)}")
        metas = [json.loads((OLD / old / "meta.json").read_text())["params"] for old in olds]
        (hv,) = {r["harness_version"] for r in rows}
        run = new_run(
            "cfbench",
            slug,
            description=f"eval essay_grading: the old pre-registered test's rows imported from {', '.join(olds)} (harness v{hv})",
            models=[{"name": p, "served_by": "openrouter"} for p in sorted({r["predictor"] for r in rows})],
            params={
                "env": f"eval:{ENV}",
                "eval": ENV,
                "envs": ENVS,
                "inputs": str(inputs.relative_to(ROOT)),
                "versions": list(VERSIONS),
                "conditions": ",".join(conditions),
                "predictors": ",".join(sorted({r["predictor"] for r in rows})),
                "model_name": None,
                "models": sorted({r["model"] for r in rows}),
                "task": "rate",
                "effort": metas[0].get("effort"),
                "max_calls": metas[0].get("max_calls", 20),
                "draws": metas[0]["draws"],
                "harness_version": hv,
                "olens_reads": sorted({m["olens_reads"] for m in metas}),
                "imported_from": [f"runs/cfbench/{old}" for old in olds],
            },
        )
        (run / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (run / "docent.json").write_text(
            json.dumps({"collection_id": json.loads((OLD / olds[0] / "docent.json").read_text())["collection_id"], "runs": docent})
        )
        finish(run, sync=False)
        print(run)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
