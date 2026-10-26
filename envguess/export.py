# ruff: noqa: E501
"""Exports: expview run (every investigation with its evidence folder, plus each judge call with its exact prompt and
reply) and a Docent run dir (rows.jsonl in the evalcue format, one row per investigation).

  ~/workspace/donehack/.venv/bin/python envguess/export.py --inv data/envguess/inv/v1 --judge data/envguess/judge/v1 --tag v1
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "envguess"
sys.path.insert(0, str(Path.home() / "workspace/expview"))
import investigate as I


def main():
    from expview import Run, call

    ap = argparse.ArgumentParser()
    ap.add_argument("--inv", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--docent-dir", default="")
    a = ap.parse_args()
    judges = {}
    for f in Path(a.judge).glob("*.json"):
        d = json.loads(f.read_text())
        judges.setdefault((d["arm"], d["view"], d["seed"]), []).append(d)
    R = Run(
        Path.home() / "workspace/expview/runs" / f"envguess_{a.tag}",
        title="Env only vs env + transcript vs transcript only: do investigators name the real edit?",
        description=(
            "One item = one DeepSeek V4.1 Flash investigation (reasoning medium, read-only list_files/read_file/grep, 20 turns, then submit_hypotheses) "
            "of one setting in one arm: `env` (exact inputs of one run), `env_tx` (+ that run's transcript, where the behaviour happened), `tx` (transcript only). "
            "The env files shown are the investigator's whole evidence folder. Then one Claude Sonnet 5.5 judge call per candidate edit (real or null; role hidden from the judge): "
            "did the hypotheses name the feature the diff changes? Labels: real_named / null_named = share of real / null edits named; pairwise = real named at a better rank than null (0.5 = tie)."
        ),
        label_order=["arm", "family", "setting", "real_named", "null_named", "pairwise"],
    )
    R.reset()
    rows = []
    for f in sorted(Path(a.inv).glob("*.json")):
        d = json.loads(f.read_text())
        r = d["record"]
        sid, cid = r["view"].rsplit("__", 1)
        s = json.loads((DATA / "settings" / sid / "setting.json").read_text())
        root = I.VIEWS / r["arm"] / r["view"]
        files = {str(p.relative_to(root)): p.read_text(errors="replace") for p in sorted(root.rglob("*")) if p.is_file()}
        js = sorted(judges.get((r["arm"], r["view"], r["seed"]), []), key=lambda j: j["label"])
        labels = {"arm": r["arm"], "family": s["family"], "setting": sid}
        if r["result"] and len(js) == len(s["edits"]):
            real = [j["verdict"]["named"] for j in js if s["edits"][j["label"]]["role"] == "real"]
            null = [j["verdict"]["named"] for j in js if s["edits"][j["label"]]["role"] == "null"]
            labels |= {"real_named": sum(real) / len(real), "null_named": sum(null) / len(null)}
        calls = [
            call(
                "investigator",
                I.MODEL,
                messages=d["messages"],
                tools=I.TOOLS,
                params={"reasoning": {"effort": "medium"}, "provider": I.PROVIDER, "seed": r["seed"]},
                cost=r["cost"],
                parsed=r["result"],
            )
        ]
        for j in js:
            e = s["edits"][j["label"]]
            calls.append(
                call(
                    f"judge edit {j['label']} ({e['role']}: {e['arm']}, rate {e['rate'][0]}/{e['rate'][1]})",
                    j["model"],
                    messages=[{"role": "user", "content": j["prompt"]}, {"role": "assistant", "content": j["response"]}],
                    params={"temperature": 0},
                    parsed=j["verdict"],
                    cost=j["cost"],
                )
            )
        R.add(
            id=f"{r['arm']}/{r['view']}/s{r['seed']}",
            group=sid,
            labels=labels,
            env_files=files,
            calls=calls,
            extra={"setting": s, "case": cid},
        )
        rows.append(
            {
                "env": "envguess",
                "model": I.MODEL,
                "arm": r["arm"],
                "item": r["view"],
                "sample": r["seed"],
                "seed": r["seed"],
                "thinking": True,
                "messages": d["messages"],
                "positive": labels.get("real_named", 0) > 0,
                "real_named": labels.get("real_named"),
                "null_named": labels.get("null_named"),
                "setting": sid,
                "family": s["family"],
            }
        )
    print("expview", R.path if hasattr(R, "path") else "", len(rows))
    if a.docent_dir:
        dd = Path(a.docent_dir)
        dd.mkdir(parents=True, exist_ok=True)
        with open(dd / "rows.jsonl", "w") as fh:
            fh.writelines(json.dumps(row) + "\n" for row in rows)
        print("docent rows", dd)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
