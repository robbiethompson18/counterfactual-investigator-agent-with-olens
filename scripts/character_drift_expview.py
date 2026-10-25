"""Export a judged character_drift run to an expview page (Kylie's standard run viewer: env files + exact model and judge I/O).

    EXPVIEW=~/workspace/expview uv run python scripts/character_drift_expview.py data/runs/character_drift_sonnet/2026-10-25_think

Writes $EXPVIEW/runs/<run name>/ (run.json, items.jsonl, envs/) and builds <that dir>/page.html. One item per row: the Qwen call
(every message exactly as sent, the reply and its reasoning) and the Sonnet judge call (the judge's exact system + user input,
its reasoning and raw JSON). The env tree per item is the scenario (or WildChat chat) JSON plus the env and judge source.
expview lives outside this repo (EXPVIEW, default ~/workspace/expview); see its README.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

EXPVIEW = Path(os.environ.get("EXPVIEW", Path.home() / "workspace/expview")).expanduser()
sys.path.insert(0, str(EXPVIEW))
from expview import Run, call  # ty: ignore[unresolved-import]  (outside the repo)

ENV_DIR = Path("evalcue/envs/character_drift")
SOURCES = [ENV_DIR / "__init__.py", ENV_DIR / "sonnet.py", ENV_DIR / "wildchat.py", Path("scripts/character_drift_judge.py")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    a = ap.parse_args()
    run_dir = a.run
    env_name = run_dir.parent.name  # character_drift_sonnet / character_drift_wildchat
    meta = json.loads((run_dir / "meta.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "rows.jsonl").open()]
    judged = {r["key"]: r for r in map(json.loads, (run_dir / "judge.jsonl").open()) if r.get("parsed")}
    docent = json.loads((run_dir / "docent.json").read_text()) if (run_dir / "docent.json").exists() else None
    if env_name.endswith("sonnet"):
        data = {s["id"]: s for s in json.loads((ENV_DIR / "data/sonnet_scenarios.json").read_text())}
    else:
        data = {c["id"]: c for c in json.loads((ENV_DIR / "data/wildchat_chats.json").read_text())}
    src = {str(p): p.read_text() for p in SOURCES}
    thinking = meta["params"]["thinking"]
    name = f"{env_name}_{run_dir.name}"
    out = EXPVIEW / "runs" / name
    desc = (
        f"{meta.get('description', '')}\n\nQwen3.6-27B, thinking {'on' if thinking else 'off'}. Earlier assistant turns are prefilled "
        "(drifted out of the default voice, or default-voice control turns), then one more user message; Qwen writes the next reply. "
        "Arms are PREFILL_FOLLOWUP. Labels: `judge` = Sonnet 5.5's in_character / mixed / default for the final reply, "
        "`ai_self_id` = the judge's explicit-AI-self-identification flag, `proxy_self_id` = the env's regex detector. "
        f"Run dir `{env_name}/{run_dir.name}` in the sprint HF dataset "
        "(Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens, runs/)."
    )
    order = ["judge", "ai_self_id", "proxy_self_id", "prefill", "followup", "character"]
    run = Run(str(out), title=f"Character drift: {env_name.split('_')[-1]} {run_dir.name[11:]}", description=desc, label_order=order)
    run.reset()
    for i, r in enumerate(rows):
        if "error" in r:
            continue
        key = f"{r['arm']}|{r['item']}|{r['sample']}"
        j = judged.get(key)
        sent, reply = r["messages"][:-1], r["messages"][-1]
        calls = [
            call(
                "qwen",
                "Qwen/Qwen3.6-27B",
                messages=sent,
                output=reply,
                params={"thinking": thinking, "finish": r.get("finish"), "completion_tokens": r.get("completion_tokens")},
            )
        ]
        if j:
            jout = {"role": "assistant", "content": j["raw"], "reasoning_content": j.get("judge_reasoning") or ""}
            calls.append(
                call(
                    "judge",
                    j["model"],
                    messages=[{"role": "system", "content": j["system"]}, {"role": "user", "content": j["user"]}],
                    output=jout,
                    params={"reasoning_effort": j["reasoning_effort"]},
                    parsed=j["parsed"],
                )
            )
        files = {f"{r['item']}.json": json.dumps(data.get(r["item"]), indent=1, ensure_ascii=False), **src}
        links = {}
        if docent and docent["runs"][i]:
            links["Docent transcript"] = f"https://docent.transluce.org/dashboard/{docent['collection_id']}/agent_run/{docent['runs'][i]}"
        labels = {
            "judge": j["parsed"]["label"] if j else None,
            "ai_self_id": j["parsed"]["explicit_ai_self_id"] if j else None,
            "proxy_self_id": r.get("positive"),
            "prefill": r["meta"]["prefill"],
            "followup": r["meta"]["followup_kind"],
            "character": (r["meta"].get("character") or "")[:60],
        }
        run.add(id=f"{r['arm']}/{r['item']}/{r['sample']}", group=r["arm"], labels=labels, env_files=files, calls=calls, links=links)
    subprocess.run([sys.executable, str(EXPVIEW / "expview.py"), "build", str(out), "-o", str(out / "page.html")], check=True)
    print(out / "page.html")


if __name__ == "__main__":
    main()
