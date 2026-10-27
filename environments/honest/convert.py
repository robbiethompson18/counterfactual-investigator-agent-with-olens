"""data/runs/honest_<scenario>/<run>/rows.jsonl (evalcue, one run per scenario) -> data/envs/honest/<run>/, after checking
build(). Same-named runs of the 11 scenarios become one run (arms `<scenario>/<framing>`); each scenario kept its own
Docent collection, so meta.json lists them in `docent_collections` and every episode keeps its own `docent_url`."""

import json
import sys
from collections import defaultdict

from environments._lib.convert import docent_url, evalcue_target, first_diff, read_rows, write_run
from environments._lib.core import episode_id
from environments._lib.runfolder import REPO_ROOT
from environments.honest import env as E

SRC = REPO_ROOT / "data" / "runs"


def main(dry=False):
    by_run = defaultdict(list)
    for s in E.SCENARIOS:
        for run in sorted(p for p in (SRC / f"honest_{s}").iterdir() if (p / "rows.jsonl").exists()):
            by_run[run.name].append((s, run))
    report = {}
    for run_name, parts in sorted(by_run.items()):
        mism, measure_diff, eps, labels, sources, colls, targets = [], 0, [], [], {}, {}, set()
        for s, run in parts:
            meta = json.loads((run / "meta.json").read_text())
            rows, (coll, docent_runs) = read_rows(run)
            tgt = evalcue_target(meta)
            targets.add(json.dumps(tgt, sort_keys=True))
            for i, r in enumerate(rows):
                arm = f"{s}/{r['arm']}"
                ep = E.build(arm, r["item"])
                r.setdefault("messages", [])  # an errored row (API failure) has no transcript
                for k, new in enumerate(ep.messages if r["messages"] else []):
                    old = r["messages"][k]
                    if new != {kk: old[kk] for kk in ("role", "content") if kk in old}:
                        diff = first_diff(new["content"], old.get("content") or "")
                        mism.append({"run": f"{s}:{i}", "arm": arm, "item": r["item"], "diff": diff})
                eid = episode_id(arm, r["item"], r["sample"])
                stored = {"positive": r.get("positive"), **(r.get("detect_notes") or {})}
                if "error" not in r and r.get("positive") is not None:
                    recomputed = E.measure({"arm": arm, "messages": r["messages"]})
                    if any(recomputed.get(k) != v for k, v in stored.items() if k in recomputed):
                        measure_diff += 1
                eps.append(
                    {
                        "id": eid, "arm": arm, "item": r["item"], "sample": r["sample"], "target": tgt["id"],
                        "messages": r["messages"], "tools": [], "files_before": {}, "files_after": {}, "state": {},
                        "meta": {"scenario": s, **r.get("meta", {})}, "finish": r.get("finish"), "turns": r.get("turns"),
                        "usage": {"completion_tokens": r.get("completion_tokens")}, "error": r.get("error"),
                        "source": f"data/runs/honest_{s}/{run.name}:{i}", "docent_run": docent_runs[i] if i < len(docent_runs) else None,
                        "docent_url": docent_url(coll, docent_runs, i),
                    }
                )  # fmt: skip
                labels.append((eid, None if "error" in r else {"scenario": s, **stored}))
            colls[s] = coll
            sources[s] = {k: meta.get(k) for k in ("description", "command", "params")} | {
                "converted_from": f"data/runs/honest_{s}/{run.name}",
                "git": meta.get("git_sha") or meta.get("git"),
                "status": meta.get("status", "done"),
            }
        if len(targets) != 1:
            raise SystemExit(f"{run_name}: scenarios ran on different targets: {targets}")
        report[run_name] = {
            "scenarios": [s for s, _ in parts],
            "rows": len(eps),
            "errored": sum(1 for e in eps if e["error"]),
            "with_docent_url": sum(1 for e in eps if e["docent_url"]),
            "start_context_mismatches": len(mism),
            "examples": mism[:3],
            "measure_disagrees": measure_diff,
        }
        if not dry and not mism:
            statuses = {v["status"] for v in sources.values()}
            m = {
                "description": f"honest: {len(parts)} scenarios x 5 framings (evalcue honest.* runs '{run_name}', merged)",
                "converted_from": [v["converted_from"] for v in sources.values()],
                "source_env": "evalcue honest.*",
                "git": next(iter(sources.values()))["git"],
                "target": json.loads(targets.pop()),
                "sources": sources,
                "docent_collection": None,
                "docent_collections": colls,
                "source_status": statuses.pop() if len(statuses) == 1 else "mixed",
                "primary": E.PRIMARY,
            }
            write_run("honest", run_name, m, eps, labels)
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main(dry="--dry" in sys.argv)
