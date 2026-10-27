# ruff: noqa: E501
"""expview export of a partial-resampling run (realism/resample.py): one item per continuation, grouped by turn.

uv run python realism/resample_export.py asst_formal@low_13_0
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, "/home/claude/workspace/expview")
from expview import Run, call


def main():
    name = sys.argv[1]
    d = HERE / "runs" / "resample" / name
    summ = json.loads((d / "summary.json").read_text())
    rows = "\n".join(
        f"| from assistant turn {t['turn']} (prefix of {t['prefix_len']} messages) | {t['not_ask']}/{t['n']} | {t['picked_one']}/{t['n']} | {t['sent_multiple']}/{t['n']} | {t['asked']}/{t['n']} |"
        for t in summ["turns"]
    )
    desc = (
        f"Partial resampling of {summ['source']} (GLM 5.3, email env). From each assistant turn k the transcript is kept up to turn k, "
        "then GLM regenerates from there 16 times and each continuation runs to the end in the same env.\n\n"
        "| resampled from | not ask | picks one | sends to several | asks |\n|---|---|---|---|---|\n" + rows
    )
    run = Run(
        Path("/home/claude/workspace/expview/runs") / f"resample_{name.replace('@', '_')}",
        title=f"Partial resampling: {summ['source']}",
        description=desc,
        label_order=["turn", "not_ask", "picked_one", "asked"],
    )
    run.reset()
    for f in sorted(d.glob("t*/*.json")):
        x = json.loads(f.read_text())
        r = x["record"]
        k = f.parent.name
        run.add(
            id=f"{k}/{f.stem}",
            group=k,
            labels={
                "turn": k,
                "not_ask": r["not_ask"],
                "picked_one": r["picked_one"],
                "asked": r["asked"],
                "sent_multiple": r["sent_multiple"],
            },
            env_files={
                "result.json": json.dumps(
                    {k2: r[k2] for k2 in ["prefix_len", "sent", "picked_one", "sent_multiple", "asked", "end", "cost"]}, indent=1
                )
            },
            calls=[call("subject", "z-ai/glm-5.3", messages=x["messages"], params={"reasoning": {"effort": "medium"}}, cost=r["cost"])],
        )
    print("exported", run)


if __name__ == "__main__":
    main()
