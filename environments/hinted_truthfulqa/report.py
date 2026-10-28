"""Data for the "reliably wrong without a hint" page: per model, every TruthfulQA item of the 8-sample control runs, with the
answers in control, the gold and hinted letters, and each hint arm's answer.

    uv run python -m environments.hinted_truthfulqa.report OUT.json
"""

import json
import sys

from environments._lib.runfolder import LOCAL, read_episodes

from . import env as E

MODELS = {  # label -> (8-sample control runs, hint runs)
    "DSv4 Flash": (
        ["2026-10-27_control-n8-dsv4f-nothink", "2026-10-27_control-n8-q100-399-dsv4f-nothink"],
        ["2026-10-27_hints-pilot-dsv4f-nothink", "2026-10-27_hints-hard79-dsv4f-nothink"],
    ),
    "GLM 5.3": (
        ["2026-10-27_control-n8-glm53-nothink", "2026-10-27_control-n8-q100-399-glm53-nothink"],
        ["2026-10-27_hints-pilot-glm53-nothink", "2026-10-27_hints-hard79-glm53-nothink"],
    ),
    "Qwen3.6-27B": (
        ["2026-10-27_control-n8-qwen36-nothink"],
        ["2026-10-27_hints-pilot-qwen36-nothink", "2026-10-27_hints-hard79-qwen36-nothink"],
    ),
}


def labelled(run):
    d = LOCAL / "hinted_truthfulqa" / run
    if not d.exists():
        return []
    f = max((d / "labels").glob("measure@*.jsonl"), key=lambda p: p.stat().st_mtime)
    lab = {r["episode"]: r["values"] for r in map(json.loads, f.read_text().splitlines())}
    return [(ep, lab.get(ep["id"]) or {}) for ep in read_episodes(d) if not ep.get("error")]


def main(out):
    items, models = {}, {}
    for name, (cruns, hruns) in MODELS.items():
        ctrl = [x for r in cruns for x in labelled(r)]
        if not ctrl:
            continue
        rows = {}
        for ep, v in ctrl:
            rows.setdefault(ep["item"], {"control": [], "hints": {}})["control"].append(v.get("answer"))
        for ep, v in (x for r in hruns for x in labelled(r)):
            if ep["item"] in rows and ep["arm"] in E.HINTS:
                rows[ep["item"]]["hints"][ep["arm"].split("/")[1]] = v.get("answer")
        models[name] = rows
        for i in rows:
            if i not in items:
                _, opts, gold = E.question(i)
                items[i] = {"q": E._items()[i]["question"].strip(), "opts": opts, "gold": gold, "hint": E.wrong_letter(i)}
    data = {"items": items, "models": models, "hints": {k.split("/")[1]: v.format(x="X") for k, v in E.HINTS.items()}}
    with open(out, "w") as fh:
        json.dump(data, fh)
    print(out, {m: len(r) for m, r in models.items()})


if __name__ == "__main__":
    main(sys.argv[1])
