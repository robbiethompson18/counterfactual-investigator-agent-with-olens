"""Collect realism overeager / email episode JSONs (one per file: messages + record + env|spec) into one rows.jsonl per run for
prep.py. Each row carries its own `tools` (from the episode's env/spec), so prep needs no harness rebuild.

    python overeager_rows.py --realism <checkout>/donehack/realism --out <dir>
"""

import argparse
import json
from pathlib import Path

SETS = {
    "payback": ("overeager/v1/payback", ["base", "no_done", "base_plain", "base_home", "base_done_neutral"]),
    "dana": ("overeager/v1/dana", ["base_no_criterion", "no_criterion", "nc_done_plain", "nc_done_home", "nc_done_neutral"]),
    "calendar": ("overeager/v1/calendar", ["base_no_criterion", "no_criterion", "nc_done_plain", "nc_done_home", "nc_done_neutral"]),
    "gitbranch": ("overeager/v1/gitbranch", ["base_no_criterion", "no_criterion", "nc_done_plain", "nc_done_home", "nc_done_neutral"]),
    "email": ("email", [f"{v}@lowc" for v in ["base", "no_done", "base_plain", "base_home", "base_done_neutral"]]),
    "email_4johns": ("email_4johns", [f"{v}@lowc" for v in ["base", "no_done", "base_plain", "base_home", "base_done_neutral"]]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realism", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.realism) / "runs"
    for name, (sub, arms) in SETS.items():
        rows = []
        for arm in arms:
            for f in sorted((root / sub / arm).glob("*.json"), key=lambda p: tuple(int(x) for x in p.stem.split("__"))):
                ep = json.loads(f.read_text())
                rec, env = ep["record"], ep.get("env") or ep["spec"]
                assert ep["messages"][0]["role"] == "system" and ep["messages"][0]["content"] == env["system"], f
                acted = rec["sent_any"] if "sent_any" in rec else rec["acted"]
                rows.append(
                    {
                        "env": f"overeager/{name}",
                        "model": "z-ai/glm-5.3",
                        "arm": arm,
                        "item": str(rec["item"]),
                        "seed": rec["seed"],
                        "sample": f.stem,
                        "positive": bool(acted),
                        "outcome": rec.get("outcome"),
                        "asked": rec.get("asked"),
                        "source": str(f.relative_to(root)),
                        "tools": env["tools"],
                        "messages": ep["messages"],
                    }
                )
        out = Path(a.out) / f"{name}.jsonl"
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(name, len(rows), sum(r["positive"] for r in rows))


if __name__ == "__main__":
    main()
