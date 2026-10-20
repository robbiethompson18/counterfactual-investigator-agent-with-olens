"""Build the CHIVE cases viewer into data/viewer/site/ (index.html + cases.json), ready for scripts/deploy_viewer.sh.

One record per held-out incident: the conversation and Qwen's reply, CHIVE's ground truth and experiments, our hand label
(viewer/behavior_kind.tsv), and every transcript-only and O-lens draw with the judge's verdict. Per-draw results are Agam's
frozen climb_test run (data/viewer/cmp_climb_test.json plus one trajectory file per incident and arm, fetched once from his
viewer and cached in data/viewer/traj/).

    uv run python scripts/build_viewer.py
"""

import json
import shutil
from pathlib import Path
from typing import Any

import requests
from detective_joracle.chive.data import load_incidents

TRAJ_URL = "https://safety-sahan-2-agam--chive-cases-web.modal.run/data/v2/climb_test/traj"
ARMS = ("transcript", "olens_rl_B")
DATA, OUT = Path("data/viewer"), Path("data/viewer/site")

cmp = {i["prompt_id"]: i for i in json.loads((DATA / "cmp_climb_test.json").read_text())["incidents"]}
hard92 = set(Path("vendor/detective-joracle/data/chive/splits/hard92.txt").read_text().split())
kinds = dict(line.split("\t") for line in Path("viewer/behavior_kind.tsv").read_text().splitlines() if not line.startswith("#"))
incs = load_incidents(list(cmp))


def passes(pid: str, arm: str) -> int:
    return sum(s >= 7 for _, s in next(a for a in cmp[pid]["arms"] if a["arm"] == arm)["draws"])


def draws(pid: str, arm: str) -> list[dict]:
    f = DATA / "traj" / f"{pid}.{arm}.json"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(requests.get(f"{TRAJ_URL}/{pid}/{arm}.json", timeout=120).text)
    out = []
    for e in json.loads(f.read_text())["episodes"]:
        j = e.get("judge") or {}
        reps = j.get("replicates") or []
        out.append(
            {
                "draw": e["draw"],
                "score": j.get("score"),
                "hypothesis": (e.get("result") or {}).get("hypothesis", ""),
                "excl": sorted({r.get("matched_exclusion") for r in reps} - {None, "none"}),
                "best": sorted({r.get("best_target") for r in reps} - {None, "none"}),
                "rationale": reps[0].get("rationale", "") if reps else "",
            }
        )
    return out


cases: list[dict[str, Any]] = []
for pid in cmp:
    inc, gt = incs[pid], incs[pid].ground_truth
    cases.append(
        {
            "pid": pid,
            "kind": kinds[pid],
            "n_prior": sum(m["role"] == "assistant" for m in inc.messages),
            "t": passes(pid, "transcript"),
            "o": passes(pid, "olens_rl_B"),
            "title": inc.behaviour,
            "k": inc.k,
            "n": inc.n,
            "messages": inc.messages,
            "completion": inc.completion,
            "observed": gt.get("observed"),
            "targets": gt["targets"],
            "ruled_out": gt["ruled_out"],
            "experiments": [{k: x.get(k) for k in ("summary", "rate", "baseline_rate", "is_key")} for x in gt["experiments"]],
            "n_tokens": inc.n_tokens,
            "hard92": pid in hard92,
            "arms": {arm: draws(pid, arm) for arm in ARMS},
        }
    )

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "cases.json").write_text(json.dumps({"cases": cases}, ensure_ascii=False))
shutil.copy("viewer/index.html", OUT / "index.html")
print(f"{len(cases)} cases, {(OUT / 'cases.json').stat().st_size / 1e6:.1f} MB -> {OUT}")
