"""Convert realism run.py episode JSONs (runs/<out>/<variant>/<pid>__<seed>.json) into rows.jsonl rows for prep.py:
messages as recorded, tools = the episode's own spec tools, scalar record fields as labels."""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="realism runs/<out> dir")
    ap.add_argument("--variants", required=True, help="comma list")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for v in a.variants.split(","):
        for f in sorted((Path(a.root) / v).glob("*.json")):
            d = json.loads(f.read_text())
            rec, sp = d["record"], d["spec"]
            assert d["messages"][0]["role"] == "system" and d["messages"][0]["content"] == sp["system"], f
            row = {
                "env": "coding",
                "model": rec.get("model"),
                "arm": v,
                "item": rec["pid"],
                "sample": str(rec["seed"]),
                "seed": str(rec["seed"]),
                "messages": d["messages"],
                "tools": sp["tools"],
                "harness": "realism",
                "source_file": f"{v}/{f.name}",
            }
            row |= {
                k: x for k, x in rec.items() if k not in row and isinstance(x, (bool, int, float, str, type(None))) and len(str(x)) <= 400
            }
            rows.append(row)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
