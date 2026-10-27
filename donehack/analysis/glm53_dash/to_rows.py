"""Write the GLM 5.3 coding-env episodes as a monorepo run dir (evalcue-shaped rows.jsonl, which
scripts/docent_upload.py ingests as-is) plus the raw episode JSONs.

  uv run python analysis/glm53_dash/to_rows.py
  cd <repo root> && uv run python scripts/runs.py finish data/runs/coding/2026-10-23_glm53
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from endkind import end_kind
from expview_export import start_files

MONO = ROOT.parent  # the monorepo root
RUN = MONO / "data/runs/coding/2026-10-23_glm53"


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    rows = []
    for rundir, noit in [("glm53", False), ("glm53_noit", True)]:
        (RUN / "episodes" / rundir).mkdir(parents=True, exist_ok=True)
        for p in sorted((ROOT / "runs" / rundir).glob("*.json")):
            shutil.copy(p, RUN / "episodes" / rundir / p.name)
            d = json.loads(p.read_text())
            r = d["record"]
            f = r["factors"]
            before = {k.removeprefix("workspace/"): v for k, v in start_files(f).items()}
            rows.append(
                {
                    "env": "coding",
                    "arm": r["cond"] + ("_noit" if noit else ""),
                    "model": r["model"],
                    "item": r["task"],
                    "sample": r["seed"],
                    "seed": r["seed"],
                    "thinking": "medium",
                    "messages": d["messages"],
                    "files_before": before,
                    "files_after": {**before, "solution.py": r["solution"] or ""},
                    "meta": {"factors": f, "pytest_line": not noit, "image": r.get("image"), "episode_file": f"episodes/{rundir}/{p.name}"},
                    "task": r["task"],
                    "condition": r["cond"],
                    "pytest_line": not noit,
                    "hacked_or_solved": r["hack"],
                    "hacked": r["hack"] if r["task"] == "is_prime" else None,
                    "solved": r["hack"] if r["task"] != "is_prime" else None,
                    "end": end_kind(r, d["messages"]),
                    "steps": r["steps"],
                    "cost": r["cost"],
                    "tokens": r["tokens"],
                    "tests_passed": r["tests_passed"],
                    "tests_failed": r["tests_failed"],
                    "tests_modified": r["tests_modified"],
                    "honest_is_prime_7": r["honest_7"] if r["task"] == "is_prime" else None,
                    "submit_cmd": r["submit_cmd"],
                    "question": r["question"],
                }
            )
    (RUN / "rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    meta = {
        "experiment": "coding",
        "slug": "glm53",
        "description": "GLM 5.3 on agent-interp-envs coding: 10 local tasks (base config) + is_prime single-factor variant grid "
        "x with/without 'Iterate ... pytest' line, 8 rollouts each, donehack/variants.py podman sandbox",
        "status": "running",
        "machine": "bramble",
        "models": [{"name": "z-ai/glm-5.3", "served_by": "OpenRouter"}],
        "params": {"reasoning_effort": "medium", "max_tokens": None, "max_steps": 100, "n": 8, "system": "neutral"},
        "command": "uv run python variants.py --model z-ai/glm-5.3 --reasoning medium --no-max-tokens --chat-timeout 900 "
        "--grade-timeout 300 --max-steps 100 --n 8 --small-uid 80 [--no-iterate] --conds <...> --out glm53[_noit]",
        "code": {"repo": "counterfactual-investigator-agent-with-olens (donehack/)", "sha": sha},
        "dashboards": {
            "summary": "https://claude.ai/artifact/RBBPSNnSWzBf46UBJ3jTKN",
            "expview": "https://claude.ai/artifact/VukYinByXae3Uo5cVKWb8C",
        },
        "n_rows": len(rows),
        "cost_usd_recorded": round(sum(x["cost"] for x in rows), 3),
    }
    old = RUN / "meta.json"
    if old.exists():
        meta = {**meta, **{k: v for k, v in json.loads(old.read_text()).items() if k in ("status", "created", "finished")}}
    old.write_text(json.dumps(meta, indent=1) + "\n")
    print(RUN, len(rows), "rows")


if __name__ == "__main__":
    main()
