"""data/runs/hidden_goal_grade/<run>/rows.jsonl (evalcue) -> data/envs/essay_grading/<run>/, after checking build()."""

import json
import sys

from environments._lib.convert import docent_url, first_diff, read_rows, write_run
from environments._lib.core import episode_id
from environments._lib.runfolder import REPO_ROOT
from environments.essay_grading import env as E

SRC = REPO_ROOT / "data" / "runs" / "hidden_goal_grade"


def target(meta: dict) -> dict:
    m = meta["models"][0]
    served = m.get("served_by", "")
    provider = served.split("(")[1].split(",")[0] if "openrouter (" in served else None
    model = m["name"]
    tid = f"{model}@openrouter:{provider or 'any'}"
    return {"id": tid, "model": model, "backend": "openrouter", "provider": provider, "thinking": meta["params"]["thinking"],
            "effort": "medium", "max_tokens": meta["params"].get("max_tokens"), "served_by": served}  # fmt: skip


def main(dry=False):
    report = {}
    for run in sorted(p for p in SRC.iterdir() if (p / "rows.jsonl").exists()):
        meta = json.loads((run / "meta.json").read_text())
        rows, (coll, docent_runs) = read_rows(run)
        mism, measure_diff, eps, labels = [], 0, [], []
        for i, r in enumerate(rows):
            ep = E.build(r["arm"], r["item"])
            n = len(ep.messages)
            for k in range(n):
                new, old = ep.messages[k], r["messages"][k]
                if new != {kk: old[kk] for kk in ("role", "content") if kk in old}:
                    mism.append(
                        {"line": i, "arm": r["arm"], "item": r["item"], "diff": first_diff(new["content"], old.get("content") or "")}
                    )
            eid = episode_id(r["arm"], r["item"], r["sample"])
            stored = {"harsh": bool(r.get("positive")), "score": (r.get("detect_notes") or {}).get("score", r.get("score"))}
            stored["has_score"] = stored["score"] is not None
            if "error" not in r and E.measure({"messages": r["messages"]}) != stored:
                measure_diff += 1
            eps.append(
                {
                    "id": eid, "arm": r["arm"], "item": r["item"], "sample": r["sample"], "target": target(meta)["id"],
                    "messages": r["messages"], "tools": [], "files_before": {}, "files_after": {}, "state": {}, "meta": r.get("meta", {}),
                    "finish": r.get("finish"), "turns": r.get("turns"), "usage": {"completion_tokens": r.get("completion_tokens")},
                    "error": r.get("error"), "source": f"data/runs/hidden_goal_grade/{run.name}:{i}",
                    "docent_run": docent_runs[i] if i < len(docent_runs) else None,
                    "docent_url": docent_url(coll, docent_runs, i),
                }
            )  # fmt: skip
            labels.append((eid, None if "error" in r else stored))
        report[run.name] = {
            "rows": len(rows),
            "start_context_mismatches": len(mism),
            "examples": mism[:3],
            "measure_disagrees": measure_diff,
        }
        if not dry and not mism:
            m = {
                "description": meta.get("description"),
                "converted_from": f"data/runs/hidden_goal_grade/{run.name}",
                "source_env": "evalcue hidden_goal.grade",
                "git": meta.get("git_sha") or meta.get("git"),
                "command": meta.get("command"),
                "target": target(meta),
                "params": meta.get("params"),
                "docent_collection": coll,
                "source_status": meta.get("status", "done"),
                "primary": E.PRIMARY,
            }
            write_run("essay_grading", run.name, m, eps, labels)
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main(dry="--dry" in sys.argv)
