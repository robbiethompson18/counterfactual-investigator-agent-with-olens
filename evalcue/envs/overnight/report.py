"""Print Markdown rate tables from saved rows: uv run python -m evalcue.envs.overnight.report --date 2026-10-22.

Only `positive` is an outcome; n excludes unscorable/error rows. Keep runs separate, including smoke and replications.
"""

import argparse
import json
from pathlib import Path


def tables(root: Path, date: str):
    for env in ("creds", "ship", "upload"):
        runs = []
        for run in sorted((root / f"overnight_{env}").glob(f"{date}_*")):
            if not (run / "rows.jsonl").exists():
                continue
            meta = json.loads((run / "meta.json").read_text())
            rows = [json.loads(line) for line in (run / "rows.jsonl").read_text().splitlines()]
            runs.append((meta, rows))
        if not runs:
            continue
        print(f"### {env}\n")
        print("| arm | " + " | ".join(meta["slug"] for meta, _ in runs) + " |")
        print("| --- | " + " | ".join("---" for _ in runs) + " |")
        arms = sorted({arm for meta, _ in runs for arm in meta["params"]["arms"]})
        for arm in arms:
            cells = []
            for meta, rows in runs:
                if arm not in meta["params"]["arms"]:
                    cells.append("—")
                    continue
                selected = [r for r in rows if r["arm"] == arm]
                scored = [r for r in selected if isinstance(r.get("positive"), bool)]
                invalid = len(selected) - len(scored)
                cell = f"{sum(r['positive'] for r in scored)}/{len(scored)}"
                if invalid:
                    cell += f" (+{invalid} unscored)"
                if len(selected) < meta["params"]["n_per_arm"]:
                    cell += " pending"
                cells.append(cell)
            print(f"| `{arm}` | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=Path("data/runs"))
    args = parser.parse_args()
    tables(args.root, args.date)
