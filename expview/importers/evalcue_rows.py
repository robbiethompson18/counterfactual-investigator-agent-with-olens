# /// script
# requires-python = ">=3.12"
# ///
"""An evalcue-harness run (rows.jsonl) -> an expview run in RUN/expview: one item per row, the subject's whole episode as
its call, the workspace before / after as its environment, the env's detector flags as labels, and its Docent link.

    uv run --script expview/importers/evalcue_rows.py data/runs/funding_email/2026-10-23_glm53_evalcue \
        [--base imports/celeste/ds/evalcue]

--base makes each item an overlay (README "Overlays") on the item of that dataset that shows the same row (matched on
extra.run_dir + row_index, as Celeste's importer writes them), so the transcript opens with its O-lens reads inline and
there is one page per transcript. Rows without a base item stay plain items.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from expview import Run, call, discover

DOCENT = "https://docent.transluce.org"


def base_items(d: str, run_key: str) -> dict[int, str]:
    """row index -> item id, for the items of dataset d that show a row of runs/<run_key>/rows.jsonl."""
    run = discover().get(d)
    out = {}
    if run:
        for line in (run / "items.jsonl").read_text().splitlines():
            if line.strip():
                it = json.loads(line)
                ex = it.get("extra") or {}
                if str(ex.get("run_dir", "")).removeprefix("runs/").strip("/") == run_key and ex.get("source_file") in (None, "rows.jsonl"):
                    out.setdefault(int(ex["row_index"]), it["id"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="data/runs/<env>/<run> with rows.jsonl (and meta.json, docent.json)")
    ap.add_argument("--base", help="dataset whose items show these rows' transcripts with reads (make overlays on them)")
    a = ap.parse_args()
    rows = [json.loads(x) for x in (a.run / "rows.jsonl").read_text().splitlines() if x.strip()]
    meta = json.loads((a.run / "meta.json").read_text()) if (a.run / "meta.json").exists() else {}
    doc = json.loads((a.run / "docent.json").read_text()) if (a.run / "docent.json").exists() else None
    key = f"{a.run.parent.name}/{a.run.name}"
    bases = base_items(a.base, key) if a.base else {}
    flags = sorted({k for r in rows for k, v in (r.get("detect_notes") or {}).items() if isinstance(v, bool)})
    run = Run(
        a.run / "expview",
        title=f"{a.run.parent.name}: {a.run.name}",
        description=(
            f"{meta.get('description', '')}\n\nOne item per rollout (`{key}` line = item order). The call is the subject's whole "
            "episode, exactly as sent and returned; the environment is the workspace before the episode (`before/`) and the files "
            "it changed (`after/`). Labels: **positive** is the env's behaviour label, the rest are its detector's flags."
            + (" Items open on the same transcript with Celeste's O-lens reads inline." if bases else "")
        ),
        label_order=["positive", *flags],
    )
    run.reset()
    n_base = 0
    for i, r in enumerate(rows):
        notes = r.get("detect_notes") or {}
        labels = {"positive": r.get("positive"), **{k: notes[k] for k in flags if k in notes}}
        before, after = r.get("files_before") or {}, r.get("files_after") or {}
        env = {f"before/{k}": v for k, v in before.items()} | {f"after/{k}": v for k, v in after.items() if before.get(k) != v}
        links = {}
        if doc and i < len(doc["runs"]) and doc["runs"][i]:
            links["Docent"] = f"{DOCENT}/dashboard/{doc['collection_id']}/agent_run/{doc['runs'][i]}"
        base = (a.base, bases[i]) if i in bases else None
        n_base += base is not None
        run.add(
            id=f"{r.get('arm')}/s{r.get('sample')}",
            group=r.get("arm"),
            labels={k: v for k, v in labels.items() if v is not None},
            calls=[call("subject", r.get("model"), messages=r.get("messages") or [], params={"thinking": r.get("thinking")})],
            env_files=env or None,
            links=links or None,
            extra={"source": f"{key}:{i}", **{k: r.get(k) for k in ("item", "seed", "finish", "turns", "completion_tokens") if k in r}},
            base=base,
        )
    print(f"{run.path}: {len(rows)} items, {n_base} overlays on {a.base}" if a.base else f"{run.path}: {len(rows)} items")


if __name__ == "__main__":
    main()
