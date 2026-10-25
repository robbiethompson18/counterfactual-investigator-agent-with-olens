# ruff: noqa: E501, RUF100
"""expview run of the evalness judge calls (full input/output) + rates per variant for the chart."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/claude/workspace/expview")
from expview import Run, call  # noqa: E402

for fam in ["ib", "email"]:
    rd = HERE / "runs" / f"evalness_{fam}"
    bt = {r["variant"]: r for r in json.loads((rd / "bt.json").read_text())}
    run = Run(
        f"/home/claude/workspace/expview/runs/realism_evalness_{fam}",
        title=f"Evalness pairwise judge ({fam})",
        description="DSv4.1 Flash sees two environments (everything the target saw at the start) and says which looks more like an RL/eval environment than a real deployment. Every ordered pair once. Item = one judgement; env files = the two rendered environments; the call is the judge's exact input and output.\n\n"
        + "| rank | variant | BT evalness (95% CI) | wins |\n|---|---|---|---|\n"
        + "\n".join(
            f"| {i + 1} | {v} | {r['bt']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | {r['wins']}/{r['games']} |"
            for i, (v, r) in enumerate(bt.items())
        ),
    )
    run.reset()
    envs = {p.stem: p.read_text() for p in (rd / "envs").glob("*.md")}
    for f in sorted((rd / "calls").glob("*.json")):
        d = json.loads(f.read_text())
        r = d["row"]
        run.add(
            id=f"{r['a']} vs {r['b']}",
            group=r["a"],
            labels={"A": r["a"], "B": r["b"], "more_eval": r["more_eval"], "confidence": r["confidence"]},
            env_files={f"A__{r['a']}.md": envs.get(r["a"], ""), f"B__{r['b']}.md": envs.get(r["b"], "")},
            calls=[
                call(
                    "judge",
                    d["request"]["model"],
                    messages=d["request"]["messages"],
                    output=d["response"],
                    params={"reasoning": d["request"].get("reasoning")},
                    cost=r["cost"],
                )
            ],
        )
    print("exported", fam)
