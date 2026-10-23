"""Check each seed's grader.py against the labels stored with the original runs.

    .venv/bin/python scripts/check_seed_graders.py [seed_id ...]

Robbie's seeds: every row of every run listed in env-explorer/site/index.json envs[env]["runs"] for the seed's three
arms (all items, error rows dropped) is turned into an evolve record (as target.py builds it) and graded with the
seed's grader.py through evolve.grading.grade_records; the verdict is compared with the row's stored "positive".
donehack_is_prime: the donehack variants runs (V4 Flash; no Qwen variant runs exist) for the three conds, compared with
the stored "hack".
"""

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from convert_seeds import extra_meta
from evolve.grading import grade_records

EXPLORER = Path.home() / "workspace/env-explorer"
DONEHACK_RUNS = [Path.home() / "workspace/donehack/runs" / r for r in ("var_noit_v4f", "var_v4f")]
FINISH = {"stop": "reply"}
CHUNK = 150


def _args(raw):
    try:
        return json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {"_raw": raw}


def record_from(messages, n_prompt, finish, files_after, meta):
    """Same derived fields as evolve.target.run_episode."""
    tr = messages[n_prompt:]
    outputs = {m.get("tool_call_id"): m.get("content") for m in tr if m["role"] == "tool"}
    turns = [
        {
            "text": m.get("content") or "",
            "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
            "tool_calls": [
                {"name": c["function"]["name"], "args": _args(c["function"].get("arguments")), "output": outputs.get(c["id"], "")}
                for c in m.get("tool_calls") or []
            ],
        }
        for m in tr
        if m["role"] == "assistant"
    ]
    return {
        "transcript": tr,
        "turns": turns,
        "tool_calls": [c for t in turns for c in t["tool_calls"]],
        "final_response": next((m.get("content") or "" for m in reversed(tr) if m["role"] == "assistant"), ""),
        "n_turns": len(turns),
        "finish": FINISH.get(finish, finish),
        "files_after": files_after or {},
        "readonly_tampered": [],
        "meta": meta,
    }


def robbie_rows(seed: Path):
    info = json.loads((seed / "node.json").read_text())
    env, arms = info["source"]["env"], set(info["source"]["arms"].values())
    runs = json.loads((EXPLORER / "site/index.json").read_text())["envs"][env]["runs"]
    out = []
    for run in runs:
        for line in (EXPLORER / "raw/runs" / env / run / "rows.jsonl").open():
            r = json.loads(line)
            if "error" in r or r["arm"] not in arms:
                continue
            r.setdefault("positive", r.get("hack"))
            n = next((i for i, m in enumerate(r["messages"]) if m["role"] not in ("system", "user")), len(r["messages"]))
            meta = {**r["meta"], **extra_meta(env, r.get("files_before") or {})[0]}
            out.append(
                (
                    f"{run}/{r['arm']}/{r['item']}/{r['sample']}",
                    r["positive"],
                    record_from(r["messages"], n, r["finish"], r["files_after"], meta),
                )
            )
    return out


def donehack_rows(seed: Path):
    conds = json.loads((seed / "node.json").read_text())["source"]["conds"]
    metas = {c: json.loads((seed / arm / "meta.json").read_text()) for arm, c in conds.items()}
    out = []
    for run in DONEHACK_RUNS:
        for f in sorted(run.glob("*.json")):
            cond = f.name.split("__")[0]
            if cond not in metas:
                continue
            d = json.loads(f.read_text())
            rec = d["record"]
            finish = {"submitted": "command", "responded": "reply", "max_steps": "max_turns"}.get(rec["end"], rec["end"])
            out.append(
                (f"{run.name}/{f.stem}", rec["hack"], record_from(d["messages"], 2, finish, {"solution.py": rec["solution"]}, metas[cond]))
            )
    return out


async def check(seed: Path):
    rows = donehack_rows(seed) if seed.name == "donehack_is_prime" else robbie_rows(seed)
    src = (seed / "grader.py").read_text()
    verdicts = []
    for i in range(0, len(rows), CHUNK):
        verdicts += await grade_records(src, [r[2] for r in rows[i : i + CHUNK]], timeout=1800)
    bad = [(rid, want, got) for (rid, want, _), got in zip(rows, verdicts) if got != want]
    counts = Counter(str(v) if not isinstance(v, dict) else "error" for v in verdicts)
    return seed.name, len(rows), bad, counts


async def main():
    only = sys.argv[1:]
    seeds = [d for d in sorted((ROOT / "seeds").iterdir()) if (d / "grader.py").exists() and (not only or d.name in only)]
    for name, n, bad, counts in await asyncio.gather(*(check(s) for s in seeds)):
        print(f"{name:48s} rows {n:5d}  agree {n - len(bad):5d}  mismatch {len(bad):3d}  verdicts {dict(counts)}")
        for rid, want, got in bad[:8]:
            print(f"    {rid}: stored {want!r} grader {got!r}")


if __name__ == "__main__":
    asyncio.run(main())
