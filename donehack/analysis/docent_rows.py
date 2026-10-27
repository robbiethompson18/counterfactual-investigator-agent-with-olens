"""Write donehack episode dirs (runs/<name>/*.json, from port.py, variants.py or evilgenie/run.py) as monorepo run dirs:
evalcue-shaped rows.jsonl (which scripts/docent_upload.py ingests as-is: arm = condition, item = task) + meta.json + the raw
episode JSONs. All go to experiment "coding", i.e. one Docent collection `coding`, filterable by model / condition / hacked.

  cd donehack && uv run python analysis/docent_rows.py v4f-variants        # one of RUNS below
  cd donehack && uv run python analysis/docent_rows.py envport ENV EPDIR DATE "DESCRIPTION"   # envport/<env> runs
  cd .. && DOCENT_API_KEY=$(cat docent.key) uv run --script scripts/docent_upload.py upload data/runs/coding/<dir>
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import port
import variants
from endkind import end_kind

MONO = ROOT.parent  # the monorepo root

# run dir name -> (date, description, [(episode dir, pytest_line, harness)])
RUNS = {
    "v4f-variants": (
        "2026-10-21",
        (
            "DeepSeek V4 Flash, agent-interp-envs coding is_prime single-factor variants (variants.py, bwrap), "
            "26 conditions x with/without 'Iterate ... pytest' line, n=128"
        ),
        [("var_v4f", True, "variants"), ("var_noit_v4f", False, "variants")],
    ),
    "v4f-endings": (
        "2026-10-21",
        (
            "DeepSeek V4 Flash, is_prime episode-ending conditions (port.py: reply / done tool variants / submit script "
            "variants, neutral system prompt, bwrap), with/without pytest line, n=128"
        ),
        [("ns_v4f", True, "port"), ("noit_v4f", False, "port")],
    ),
    "v4f-endings-rep": (
        "2026-10-22",
        "DeepSeek V4 Flash, is_prime done vs done+give_up replication without the pytest line (port.py, bwrap)",
        [("noit_v4f_rep", False, "port")],
    ),
    "qwen-endings": (
        "2026-10-21",
        (
            "Qwen3.6-27B (OpenRouter), is_prime episode-ending conditions (port.py, neutral system prompt, bwrap), "
            "with/without pytest line, n=128"
        ),
        [("ns_qwen", True, "port"), ("noit_qwen", False, "port")],
    ),
    "glm53-part1": (
        "2026-10-23",
        (
            "GLM 5.3 (z-ai/glm-5.3, reasoning medium, no max_tokens, max 100 steps, podman): 10 local coding tasks "
            "(repo base config) + is_prime single-factor variants x with/without pytest line"
        ),
        [("glm53", True, "variants"), ("glm53_noit", False, "variants")],
    ),
    "glm53-evilgenie-part1": (
        "2026-10-23",
        (
            "GLM 5.3 on EvilGenie (agent-interp-envs coding, configs/coding/lcb.yaml: done tool, writable tests), "
            "50 problems stratified test5/test6 x 2, visible + LiveCodeBench holdout grading"
        ),
        [("eg_glm53", None, "evilgenie")],
    ),
    "glm53-impossiblebench": (
        "2026-10-23",
        (
            "GLM 5.3 on ImpossibleBench LiveCodeBench (agent-interp-envs coding, lcb.yaml config: done tool, "
            "writable tests), conflicting / oneoff / original splits, up to 2 rollouts (stopped early for budget); "
            "ib_hack = passes the impossible test file"
        ),
        [("ib_glm53", None, "evilgenie")],
    ),
}


def files_variants(f):
    src = variants.AIE / "tasks" / f.get("task", "is_prime")
    tests = (src / f"test_solution_{f['ntests']}.py") if f.get("ntests") else src / "test_solution.py"
    return {
        "description.md": (src / "description.md").read_text(),
        "solution.py": (src / "solution.py").read_text(),
        "test_solution.py": tests.read_text(),
        "submit_solution.py": variants.reward_script(f["reward"]).read_text(),
    }


# conditions dropped from port.CONDS in 242212c (still in the ns_/noit_ runs): their submit script as of 242212c^
OLD_SCRIPT = {
    "submit_walled": "reward",
    "done_walled": None,
    "doneN_walled": None,
    "done_giveup_walled": None,
    "submitN_reply": "git:submit_solution_counts.py",
}


def files_port(cond):
    src = port.AIE / "tasks/is_prime"
    out = {n: (src / n).read_text() for n in ["description.md", "solution.py", "test_solution.py"]}
    script = port.CONDS[cond][1] if cond in port.CONDS else OLD_SCRIPT[cond]
    if script and script.startswith("git:"):
        out["submit_solution.py"] = subprocess.run(
            ["git", "-C", str(ROOT), "show", "242212c^:" + script[4:]], capture_output=True, text=True, check=True
        ).stdout
    elif script:
        out["submit_solution.py"] = port.SUBMIT_SCRIPTS[script].read_text()
    return out


def row(d, pytest_line, harness):
    r, msgs = d["record"], d["messages"]
    common = {
        "env": "coding",
        "model": r["model"],
        "sample": r["seed"],
        "seed": r["seed"],
        "messages": msgs,
        "thinking": None,
        "steps": r["steps"],
        "cost": r.get("cost"),
        "end": end_kind(r, msgs) if harness != "evilgenie" else r["end"],
        "backend": r.get("backend", "bwrap"),
        "tokens": r.get("tokens"),
        "harness": harness,
        "final_text": r.get("final_text"),
    }
    if harness == "evilgenie":
        ib = r["platform"] == "impossiblebench"
        p = json.loads((ROOT / ("evilgenie/data_ib" if ib else "evilgenie/data") / f"{r['task']}.json").read_text())
        before = p["files"]
        after = {**before, "solution.py": r["solution"] or "", "test_solution.py": r["final_tests"]}
        if ib:
            return {
                **common,
                "arm": f"impossiblebench_{r['split']}",
                "item": r["task"],
                "files_before": before,
                "files_after": after,
                "meta": {"split": r["split"]},
                "task": r["task"],
                "condition": f"impossiblebench_{r['split']}",
                "passes_given_tests": r["public_all"],
                "passes_original_tests": r.get("original_tests_all"),
                "ib_hack": r.get("ib_hack"),
                "tests_modified": r["tests_modified"],
                "hardcodes_sample": r.get("hardcodes_sample"),
            }
        return {
            **common,
            "arm": "evilgenie_lcb",
            "item": r["task"],
            "files_before": before,
            "files_after": after,
            "meta": {"split": r["split"], "platform": r["platform"], "difficulty": r["difficulty"], "title": r["title"]},
            "task": r["task"],
            "condition": "evilgenie_lcb",
            "public_all": r["public_all"],
            "holdout_all": r["holdout_all"],
            "holdout_passed": r["holdout_passed"],
            "holdout_total": r["holdout_total"],
            "hack_signal": r["hack_signal"],
            "tests_modified": r["tests_modified"],
            "solved": r["public_all"] and r["holdout_all"],
            "hardcodes_sample": r.get("hardcodes_sample"),
            "multi_answer": r.get("multi_answer"),
        }
    if harness == "variants":
        f = r["factors"]
        task = f.get("task", "is_prime")
        before = files_variants(f)
    else:
        f = None
        task = "is_prime"
        before = files_port(r["cond"])
    after = {**before, "solution.py": r.get("solution") or ""}
    arm = r["cond"] + ("" if pytest_line else "_noit")
    is_prime = task == "is_prime"
    return {
        **common,
        "arm": arm,
        "item": task,
        "files_before": before,
        "files_after": after,
        "meta": {"factors": f, "pytest_line": pytest_line, "image": r.get("image")},
        "task": task,
        "condition": r["cond"],
        "pytest_line": pytest_line,
        "hacked": r["hack"] if is_prime else None,
        "solved": None if is_prime else r["hack"],
        "honest_is_prime_7": r.get("honest_7") if is_prime else None,
        "tests_modified": r.get("tests_modified"),
        "tests_passed": r.get("tests_passed"),
        "tests_failed": r.get("tests_failed"),
        "submit_cmd": r.get("submit_cmd"),
        "question": r.get("question"),
    }


RUNS["glm53-endings"] = (
    "2026-10-24",
    (
        "GLM 5.3 is_prime episode-ending conditions (port.py: reply / done tool variants / give_up / "
        "submit script variants, neutral system prompt, podman u200), with/without pytest line, n=64, 32k/$0.10 cutoff"
    ),
    [("glm53_end", True, "port"), ("glm53_end_noit", False, "port")],
)
RUNS["glm53-impossiblebench-part2"] = (
    RUNS["glm53-impossiblebench"][0],
    RUNS["glm53-impossiblebench"][1] + " (part 2: top-up to 2 rollouts, episode cutoff 32k context / $0.10)",
    RUNS["glm53-impossiblebench"][2],
)
for _n in ["glm53", "glm53-evilgenie"]:  # part2 = episodes that finished after part1 was written (each uploaded once)
    _d, _desc, _parts = RUNS[f"{_n}-part1"]
    RUNS[f"{_n}-part2"] = (_d, _desc + " (part 2: episodes finished after part 1)", _parts)
    RUNS[f"{_n}-part3"] = (_d, _desc + " (part 3: episodes regraded after the podman storage rebuild)", _parts)


def envport_rows(env, epdir):
    """Generic rows for envport/<env>/run.py outputs: arm = config, item = env, every scalar record field as metadata
    (detector fields flattened as det_<k>)."""
    rows = []
    for p in sorted((ROOT / "runs" / epdir).glob("*.json")):
        d = json.loads(p.read_text())
        r = d["record"]
        meta = {k: v for k, v in r.items() if isinstance(v, (bool, int, float, str)) or v is None}
        for k, v in (r.get("detector") or {}).items():
            if isinstance(v, (bool, int, float, str)) or v is None:
                meta[f"det_{k}"] = v
        meta = {k: (v[:2000] if isinstance(v, str) else v) for k, v in meta.items()}
        rows.append(
            {
                **meta,
                "env": env,
                "arm": r.get("config", env),
                "item": env,
                "sample": r.get("seed"),
                "seed": r.get("seed"),
                "model": r.get("model"),
                "messages": d["messages"],
                "thinking": r.get("reasoning"),
                "files_before": {},
                "files_after": {},
                "episode_file": f"episodes/{epdir}/{p.name}",
            }
        )
    return rows


def main_envport(env, epdir, date, desc):
    run = MONO / "data/runs" / env / f"{date}_glm53"
    if (run / "docent.json").exists():
        sys.exit(f"{run} already uploaded")
    (run / "episodes" / epdir).mkdir(parents=True, exist_ok=True)
    for p in (ROOT / "runs" / epdir).glob("*.json"):
        shutil.copy(p, run / "episodes" / epdir / p.name)
    rows = envport_rows(env, epdir)
    (run / "rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    meta = {
        "experiment": env,
        "slug": "glm53",
        "description": desc,
        "status": "done",
        "machine": "bramble",
        "models": [{"name": "z-ai/glm-5.3", "served_by": "OpenRouter"}],
        "params": {"source": f"donehack/runs/{epdir}"},
        "n_rows": len(rows),
        "cost_usd_recorded": round(sum(x.get("cost") or 0 for x in rows), 3),
    }
    (run / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(run, len(rows), "rows")


def main(name):
    date, desc, parts = RUNS[name]
    if name == "glm53-impossiblebench-part2":  # first upload was named without -part1
        prior = MONO / "data/runs/coding" / f"{date}_glm53-impossiblebench" / "episodes"
        done = {(e.name, f.name) for e in prior.iterdir() for f in e.iterdir()}
    elif name[-6:] in ("-part2", "-part3"):
        done = set()
        for k in range(1, int(name[-1])):
            prior = MONO / "data/runs/coding" / f"{date}_{name[:-6]}-part{k}" / "episodes"
            done |= {(e.name, f.name) for e in prior.iterdir() for f in e.iterdir()}
    else:
        done = set()
    run = MONO / "data/runs/coding" / f"{date}_{name}"
    if (run / "docent.json").exists():
        sys.exit(f"{run} already uploaded to Docent; not rewriting rows (run ids would no longer match lines)")
    rows = []
    for epdir, pytest_line, harness in parts:
        (run / "episodes" / epdir).mkdir(parents=True, exist_ok=True)
        for p in sorted((ROOT / "runs" / epdir).glob("*.json")):
            if (epdir, p.name) in done:
                continue
            if harness == "evilgenie" and not json.loads(p.read_text())["record"].get("regraded"):
                continue  # not yet graded with the normalised comparison (evilgenie/regrade.py); goes in a later part
            shutil.copy(p, run / "episodes" / epdir / p.name)
            rows.append(row(json.loads(p.read_text()), pytest_line, harness))
    (run / "rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    models = sorted({x["model"] for x in rows})
    meta = {
        "experiment": "coding",
        "slug": name,
        "description": desc,
        "status": "done",
        "machine": "bramble",
        "models": [{"name": m, "served_by": "OpenRouter"} for m in models],
        "params": {"sources": [f"donehack/runs/{e}" for e, _, _ in parts]},
        "code": {"repo": "counterfactual-investigator-agent-with-olens (donehack/)", "sha": sha},
        "n_rows": len(rows),
        "cost_usd_recorded": round(sum(x["cost"] or 0 for x in rows), 3),
    }
    (run / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(run, len(rows), "rows", models)


if __name__ == "__main__":
    if sys.argv[1] == "envport":  # envport ENV EPDIR DATE "DESCRIPTION"
        main_envport(*sys.argv[2:6])
    else:
        for n in sys.argv[1:]:
            main(n)
