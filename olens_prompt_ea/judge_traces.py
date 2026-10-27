# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=17"]
# ///
"""Transcript-first view of the punctuation-read judge (judge_punct.py) for the shared viewer: one item per selected
prompt that opens on a real transcript with that prompt, its O-lens reads inline (the token file from Celeste's import,
copied into this run), plus a "lens judge" read at the token whose bullet the judge quoted (score + evidence, inline, so
it shows in the side panel), and the judge's full call (exact input and output) under the item's calls.

Selection: per setting the 3 highest- and 2 lowest-scored prompts, plus --ids.

    uv run --script olens_prompt_ea/judge_traces.py RUN --expview <viewer checkout>/expview --imports <data/expview/imports> [--ids a,b]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from atomic import swap_in, temp_dir

EXCLUDE = {"inspect_am", "sad_oversight", "hp_mc", "hp_abst"}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r'^[\s"“*\[\]0-9]+|["”\s]+$', "", s or "")).strip().lower()


def evidence_text(ev: str | None) -> str:
    """The quoted bullet from 'EVIDENCE: [n] "..."' (or '"..." [n]')."""
    m = re.search(r'"([^"]{8,})"', ev or "") or re.search(r"“([^”]{8,})”", ev or "")
    return norm(m.group(1) if m else re.sub(r"\[\d+\]", "", ev or ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--run-name", default=None, help="expview run dir name under RUN/expview (default: the live name)")
    ap.add_argument("--expview", type=Path, required=True)
    ap.add_argument("--imports", type=Path, required=True)
    ap.add_argument("--ids", default="")
    a = ap.parse_args()
    sys.path.insert(0, str(a.expview))
    from expview import Run, call

    e = pd.read_parquet(a.run / "kwonly_episodes.parquet", columns=["tid", "setting", "cond", "ph", "vea"])
    e = e[~e.setting.isin(EXCLUDE)].copy()
    e.loc[e.setting.isin(["am_main", "hp_am_fixed"]), "setting"] = "am"
    recs = {r["key"]: r for r in map(json.loads, (a.run / "judge" / "lens_punct.jsonl").read_text().splitlines())}
    sc = pd.Series({k: r["parsed"]["score"] for k, r in recs.items()}).dropna()
    ph_set = e.drop_duplicates("ph").set_index("ph").setting
    pick = set(filter(None, a.ids.split(",")))
    for st, phs in ph_set.groupby(ph_set):
        s = sc.reindex(phs.index).dropna().sort_values()
        pick |= set(s.index[-3:]) | set(s.index[:2])
    # where each transcript lives in Celeste's import (dataset dir + token ref)
    loc = {}
    for d in a.imports.glob("celeste/*/*"):
        if (d / "items.jsonl").exists():
            for line in (d / "items.jsonl").open():
                it = json.loads(line)
                if it.get("tokens"):
                    loc[it["id"]] = (d, it["tokens"]["ref"])
    final = a.run / "expview" / (a.run_name or "lens_punct_judge_traces")
    tmp = temp_dir(final)  # never reset the live run: build beside it, swap in when complete
    run = Run(tmp, title="Lens judge on prompt-side punctuation reads: transcripts",
              description=("Each item opens on a transcript whose prompt the DeepSeek judge scored, with every O-lens read inline "
                           "(copied from Celeste's import). The **lens judge** reader sits at the token whose bullet the judge "
                           "quoted (score + evidence); the judge's exact input and output are the item's call. The judge saw only "
                           "prompt-side GLM RL bullets at punctuation tokens, blind to the rollout. Labels: `score`, `vea` of this "
                           "episode, `vea_rate` over all episodes sharing the prompt."),
              label_order=["score", "vea", "vea_rate", "setting"],
              readers={"lens judge": {"about": "DeepSeek V4.1 Flash judge of the prompt-side RL bullets (punctuation tokens): "
                                               "score and the bullet it quoted.", "color": "#b5662f"}})  # fmt: skip
    (run.path / "tokens").mkdir(exist_ok=True)
    links = {}
    for ph in sorted(pick):
        r = recs.get(ph)
        eps = e[e.ph == ph]
        if r is None or eps.empty:
            continue
        ev = evidence_text(r["parsed"]["evidence"])
        best = None
        for t in eps.tid:
            if t not in loc:
                continue
            d, ref = loc[t]
            tok = json.loads((d / "tokens" / f"{ref}.json").read_text())
            pos = None
            if ev and ev != "none":
                for p, rs in tok["reads"].items():
                    if any(ev[:60] in norm(ln[0] if isinstance(ln, list) else ln) for rr in rs for ln in rr["lines"]):
                        pos = int(p)
                        break
            best = (t, d, tok, pos)
            if pos is not None or not ev or ev == "none":
                break
        if best is None:
            continue
        t, d, tok, pos = best
        anchor = pos if pos is not None else min(int(p) for p in tok["reads"]) if tok["reads"] else 0
        note = f"score {r['parsed']['score']}/10 — {r['parsed']['evidence'] or 'none'}"
        tok["reads"].setdefault(str(anchor), []).insert(0, {"tool": "lens judge", "lines": [[note]], "inline": True})
        tok["legend"] = (tok.get("legend") or "") + f"\n\n**Lens judge:** {note}"
        ref = f"j{ph}"
        (run.path / "tokens" / f"{ref}.json").write_text(json.dumps(tok))
        x = eps[eps.tid == t].iloc[0]
        run.add(
            id=ph, group=x.setting,
            labels={"score": r["parsed"]["score"], "vea": bool(x.vea), "vea_rate": round(float(eps.vea.mean()), 3), "setting": x.setting,
                    "cond": x.cond, "evidence_found": pos is not None},
            tokens={"ref": ref, "n": len(tok["tokens"]), "reads": len(tok["reads"])},
            links={"same transcript in Celeste's import (8830)":
                   f"https://expview.pages.dev/#d=imports/{d.relative_to(a.imports)}&i={quote(t, safe='')}&v=tr&t={anchor}"},
            calls=[call("lens judge", r["model"], messages=r["messages"],
                        output={"role": "assistant", "content": r["output"] or "", "reasoning": r.get("reasoning")},
                        params={"temperature": 0, "reasoning_effort": "low"}, parsed=r["parsed"])],
        )  # fmt: skip
        links[ph] = (
            f"https://expview.pages.dev/#d=runs/olens_prompt_ea/2026-10-24_glm53_prompt_ea/lens_punct_judge_traces&i={ph}&v=tr&t={anchor}"
        )
    swap_in(tmp, final)
    (a.run / "judge_traces_links.json").write_text(json.dumps(links, indent=1))
    print(f"{len(links)} items -> {final}")


if __name__ == "__main__":
    main()
