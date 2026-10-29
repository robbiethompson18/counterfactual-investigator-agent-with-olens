from pathlib import Path

"""Export the triaged Qwen O-lens grader-awareness hits as an expview run (judge input + output per bullet).

    python export.py WORKDIR OUT_RUN_DIR
WORKDIR holds hits3.jsonl, triage/{b*,c*}/{input,output}.jsonl, triage/instructions.md, links.json, groups2.json.
"""

import glob
import json
import sys

sys.path.insert(0, "/home/claude/workspace/expview")
from expview import Run, call

W, OUT = sys.argv[1], sys.argv[2]
ENV_NAME = {"qwen": "donation_bet (thinking off)", "prefix": "character_drift", "reply": "character_drift"}
instr = Path(f"{W}/triage/instructions.md").read_text()
links = json.loads(Path(f"{W}/links.json").read_text())
H = [json.loads(l) for l in Path(f"{W}/hits3.jsonl").read_text().splitlines()]
rows = []
for b in sorted(glob.glob(f"{W}/triage/[bc][0-9]")):
    outs = {json.loads(l)["key"]: (l.strip(), json.loads(l)) for l in Path(f"{b}/output.jsonl").read_text().splitlines()}
    for l in Path(f"{b}/input.jsonl").read_text().splitlines():
        it = json.loads(l)
        rows.append((b.rsplit("/", 1)[1], l.strip(), it, *outs[it["key"]]))
run = Run(
    OUT,
    title="Qwen O-lens grader awareness",
    description=Path(f"{W}/description.md").read_text(),
    notes=Path(f"{W}/notes.md").read_text(),
    label_order=["label", "strength", "env", "arm", "round"],
)
run.reset()
for batch, raw_in, it, raw_out, o in rows:
    h = H[it["key"]]
    env = ENV_NAME.get(h["env"], h["env"])
    d, e = links.get(h["id"].split("|")[0]) or [None, None]
    lk = {k: v for k, v in {"Docent transcript": d, "Env viewer episode": e}.items() if v}
    run.add(
        id=f"{env}/{h['id'].split(':', 1)[-1]}@{h['token_index']}L{h['layer']}b{h['bullet']}#{it['key']}",
        group=env,
        labels={
            "label": o["label"],
            "strength": o.get("strength"),
            "env": env,
            "arm": str(h["arm"]),
            "round": "sample" if batch.startswith("b") else "all_strict",
            "role": h["role"],
            "layer": h["layer"],
        },
        env={
            "files": {
                "read_source.txt": f"{h['src']}\nid {h['id']}\n"
                f"token_index {h['token_index']} token {h['token']!r} role {h['role']} layer {h['layer']}\nregex match: {h['match']}\n",
                "prefix_full.txt": h["prefix"][-12000:],
                "model_own_text.txt": (h.get("asst") or "")[:8000],
            }
        },
        calls=[
            call(
                "olens",
                "O-lens olens_s3d_space_rl600 on Qwen3.6-27B",
                messages=[{"role": "user", "content": "…" + h["prefix"][-3000:]}],
                output={"role": "assistant", "content": h["text"]},
            ),
            call(
                "judge",
                "claude-sonnet subagent (triage)",
                messages=[{"role": "system", "content": instr}, {"role": "user", "content": raw_in}],
                output={"role": "assistant", "content": raw_out},
                parsed=o,
            ),
        ],
        links=lk,
    )
print(len(rows))
