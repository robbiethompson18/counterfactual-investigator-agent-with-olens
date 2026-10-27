"""The GLM 5.3 over-eagerness runs -> data/envs/overeagerness/<run>/, after checking build() for every row.

    uv run python -m environments._lib.convert_overeager [--dry]

Sources (bramble only, from the claude/realism-variants worktree): the Docent-uploaded rows
(data/runs/overeager/<run>/rows.jsonl + docent.json) and the raw episode files they were made from
(donehack/realism/runs/overeager/<out>/<task>/<variant>/<item>__<seed>.json, which also hold the tool schemas and setup
commands). For every row, build(variant, item) is compared with the stored starting context: system prompt, first user
message, tool schemas, sandbox files, setup commands and the prefilled command. The old runs read the real clock, so the
date is taken from the raw file's mtime (the episode's end) or the day before it (an episode that crossed midnight).
"""

import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

from .convert import docent_url, first_diff, read_rows, write_run
from .core import Ctx, episode_id, load_env

WT = Path.home() / "workspace/counterfactual-investigator-agent-with-olens/.worktrees/realism"
ROWS = WT / "data/runs/overeager"
RAW = WT / "donehack/realism/runs/overeager"
RAW_EMAIL = WT / "donehack/realism/runs"  # email*, stale_*: <out>/<variant>/<item>__<seed>.json with "spec"
EMAIL_ROWS = Path.home() / "workspace/counterfactual-investigator-agent-with-olens/data/runs/email_ambiguous"
STRUCT = {"env", "model", "arm", "item", "sample", "seed", "meta", "task", "variant", "expview_id", "files_before", "files_after",
          "messages", "final_text", "cost", "end", "name"}  # fmt: skip


def raw_index():
    idx = defaultdict(list)
    for p in [*RAW_EMAIL.glob("email*/*/*.json"), *RAW_EMAIL.glob("stale_*/*/*.json")]:
        if p.stem.count("__") == 1:
            item, seed = p.stem.split("__")
            idx[("email", item, seed)].append(p)
    for p in RAW.glob("*/*/*/*.json"):
        task, variant = p.parts[-3], p.parts[-2]
        if "__" not in p.stem:
            continue
        item, seed = p.stem.split("__")
        idx[(task, variant, item, seed)].append(p)
    return idx


def compare(ep, row, raw) -> str | None:
    """None if the new starting context equals the stored one, else the first difference."""
    new_msgs = ep.messages
    for k, m in enumerate(new_msgs):
        old = row["messages"][k]
        if m["role"] != old["role"] or m["content"] != (old.get("content") or ""):
            return f"message {k} ({m['role']}): " + first_diff(m["content"], old.get("content") or "")
    if ep.prefill:
        tc = row["messages"][len(new_msgs)].get("tool_calls") or [{}]
        cmd = json.loads(tc[0].get("function", {}).get("arguments") or "{}").get("command")
        if cmd != ep.prefill[0][1]:
            return f"prefill command: new {ep.prefill[0][1]!r} vs stored {cmd!r}"
    if raw is not None:
        renv = raw.get("env") or raw["spec"]
        if ep.tools != renv["tools"]:
            return "tools: " + first_diff(json.dumps(ep.tools), json.dumps(renv["tools"]))
        if ep.setup != (renv.get("setup") or []):
            return "setup: " + first_diff(json.dumps(ep.setup), json.dumps(renv.get("setup", [])))
        if ep.workdir != renv["workdir"]:
            return f"workdir {ep.workdir} vs {renv['workdir']}"
        stored_files = renv["files"]
    else:
        stored_files = row.get("files_before") or {}
    new_files = {k: (v if isinstance(v, str) and len(v) < 200_000 else f"<{len(v)} bytes>") for k, v in ep.files.items()}
    if new_files != stored_files:
        diff = sorted(set(new_files) ^ set(stored_files)) or [k for k in new_files if new_files[k] != stored_files[k]]
        k = diff[0]
        return f"file {k}: " + (
            first_diff(new_files.get(k, ""), stored_files.get(k, "")) if k in new_files and k in stored_files else "only on one side"
        )
    return None


def main(dry=False):
    idx = raw_index()
    report, plans = {}, defaultdict(lambda: defaultdict(list))
    envs = {}
    sources = [(p, "overeager") for p in sorted(ROWS.iterdir()) if (p / "rows.jsonl").exists()]
    sources += [(p, "email_ambiguous") for p in sorted(EMAIL_ROWS.iterdir()) if (p / "rows.jsonl").exists()]
    for run, family in sources:
        rows, (coll, docent_runs) = read_rows(run)
        meta = json.loads((run / "meta.json").read_text())
        for i, r in enumerate(rows):
            if family == "email_ambiguous":
                r = {**r, "task": "email", "variant": r["arm"].split("@")[0]}
            src_task = r["task"]
            task = "email" if src_task == "email_4johns" else src_task  # the 4-Johns runs are item 0 of email
            env_name = "overeagerness"
            env = envs.setdefault(env_name, load_env(env_name))
            mod = env.MODS[task]
            if task == "email":
                item_idx = str(r["item"]).rsplit("_", 1)[-1]
                r = {**r, "item": item_idx}
                key = ("email", item_idx, str(r["seed"]))
            else:
                key = (task, r["variant"], str(r["item"]), str(r["seed"]))
            raws = [json.loads(p.read_text()) | {"_path": str(p), "_mtime": p.stat().st_mtime} for p in idx.get(key, [])]
            # several out dirs hold the same (task, variant, item, seed) from different env versions: the row's raw file
            # is the one with the identical full transcript
            raws = [x for x in raws if x["messages"] == r["messages"]]
            if len(raws) > 1 and all((x.get("env") or x.get("spec")) == (raws[0].get("env") or raws[0].get("spec")) for x in raws):
                raws = raws[:1]  # the same episode copied into two out dirs
            raw = raws[0] if len(raws) == 1 else None
            if len(raws) > 1:
                print(f"ambiguous raw for {run.name}:{i}: {[x['_path'] for x in raws]}", file=sys.stderr)
            base = dt.datetime.fromtimestamp(raw["_mtime"], dt.UTC).date() if raw else dt.date(2026, 10, 26)
            diff, today, arm = None, None, r["variant"]
            if task == "email":
                arms = [r["variant"]] + [f"{r['variant']}@{lv}" for lv in mod.LEVELS_ARMS]
            else:
                arms = [r["variant"]] + [f"{r['variant']}@{v}" for v in mod._E.versions]
            arms = [f"{task}/{a}" for a in arms]
            for a in arms:
                for d in (base, base - dt.timedelta(days=1), base + dt.timedelta(days=1)):
                    ctx = Ctx(env=env_name, arm=a, item=str(r["item"]), sample=int(r["seed"]), today=d,
                              now=dt.datetime.combine(d, dt.time(12)))  # fmt: skip
                    d2 = compare(env.build(a, str(r["item"]), ctx), r, raw)
                    if d2 is None:
                        today, arm = d, a
                        break
                    diff = diff or d2
                if today:
                    break
            r = {**r, "arm_new": arm if today else f"{task}/{r['variant']}"}
            run_name = f"{run.name}_4johns" if src_task == "email_4johns" else run.name  # same ids as the email rows
            plans[(env_name, run_name)]["rows"].append(
                (i, r, raw, today, diff, coll, docent_runs, meta | {"_src": f"{family}/{run.name}", "_task": src_task})
            )
    for (env_name, run_name), p in sorted(plans.items()):
        rows = p.get("rows", [])
        mism = [(i, r["variant"], r["item"], d) for i, r, _, t, d, *_ in rows if t is None]
        versions = sorted({r["arm_new"].partition("@")[2] or "current" for _, r, *_ in rows})
        report[f"{env_name}/{run_name}"] = {
            "rows": len(rows),
            "matched": len(rows) - len(mism),
            "mismatched": len(mism),
            "no_raw_file": sum(1 for _, _, raw, *_ in rows if raw is None),
            "examples": [f"line {i} {v}/{it}: {d}" for i, v, it, d in mism[:3]],
            "skipped_rows_no_env": len(p.get("skipped", [])),
            "env_versions": versions,
        }
        if dry or not rows:
            continue
        eps, labels = [], []
        for i, r, raw, today, diff, coll, docent_runs, meta in rows:
            eid = episode_id(r["arm_new"], r["item"], r["seed"])
            stored = {k: v for k, v in r.items() if k not in STRUCT | {"arm_new"}}
            task = r["arm_new"].split("/")[0]
            stored = {"task": task, "positive": stored.get("picked_one") if task == "email" else stored.get("acted"), **stored}
            state = (
                ({"sent": raw["record"].get("sent", [])} if task == "email" else {"actions": raw["record"].get("actions", [])})
                if raw
                else {}
            )
            eps.append(
                {
                    "id": eid, "arm": r["arm_new"], "item": str(r["item"]), "sample": int(r["seed"]),
                    "target": "z-ai/glm-5.3@openrouter:any", "today": today.isoformat() if today else None,
                    "messages": r["messages"], "tools": (raw.get("env") or raw["spec"])["tools"] if raw else None,
                    "files_before": r.get("files_before") or {}, "files_after": r.get("files_after") or {},
                    "setup": (raw.get("env") or raw["spec"]).get("setup") if raw else None, "workdir": (r.get("meta") or {}).get("workdir"),
                    "state": state, "meta": {"task": task, "variant": r["variant"], "item": int(r["item"])},
                    "finish": r.get("end"), "usage": {"cost": r.get("cost")}, "final_text": r.get("final_text"),
                    "build_check": "match" if today else f"MISMATCH: {diff}",
                    "source": f"data/runs/{meta['_src']}:{i}", "raw_source": raw["_path"].split("/realism/")[-1] if raw else None,
                    "docent_run": docent_runs[i] if i < len(docent_runs) else None,
                    "docent_url": docent_url(coll, docent_runs, i),
                }
            )  # fmt: skip
            labels.append((eid, stored))
        meta = rows[0][7]
        rep = report[f"{env_name}/{run_name}"]
        m = {
            "description": meta.get("description"),
            "converted_from": f"data/runs/{meta['_src']} (tasks {', '.join(sorted({x[7]['_task'] for x in rows}))})",
            "source_env": "donehack/realism/overeager (oe_run.py, tasks/*.py) and donehack/realism/email_env.py",
            "git": meta.get("git"),
            "target": {
                "id": "z-ai/glm-5.3@openrouter:any",
                "model": "z-ai/glm-5.3",
                "backend": "openrouter",
                "provider": None,
                "thinking": True,
                "effort": "medium",
                "max_tokens": None,
            },
            "caps": {"max_turns": 40, "cost_usd": 0.10},
            "docent_collection": rows[0][5],
            "build_check": {k: rep[k] for k in ("rows", "matched", "mismatched", "examples")},
            "source_status": meta.get("status", "done"),
            "primary": "acted",
        }
        write_run(env_name, run_name, m, eps, labels)
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main(dry="--dry" in sys.argv)
