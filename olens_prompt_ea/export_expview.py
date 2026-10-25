# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=17"]
# ///
"""expview pages for the prompt-side eval-awareness study: per episode, the environment = the rendered prompt (prompt.txt),
and the calls = the GLM transcript (prompt + generated
text, exactly as rendered for the lens), the lens judge, the prompt-only judge and the VEA monitor (exact input and output).

    uv run --script olens_prompt_ea/export_expview.py RUN --expview ~/workspace/expview --out OUTDIR [--settings a,b] [--name NAME]
then  python ~/workspace/expview/expview.py build OUTDIR/<name> -o OUTDIR/<name>.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd

GEN_SHOW = 40_000  # chars of generated text shown per episode (head + tail)
VIEW = "http://bramble:8821/#ep={tid}&pos=0&q=test%7Ceval%7Csynthetic%7Csimulat%7Cscenario%7Cfiction%7Chypothetical%7Choneypot"
LABEL_NOTE = {
    "am_main": (
        "VEA = majority of 3 Sonnet 4.5 runs of the glm-evalgaming monitor (v2 prompt); that monitor's I/O is in the AM O-lens viewer"
    ),
    "hp_am_fixed": "VEA = glm-evalgaming awareness monitor (v1 prompt, Sonnet 5.5 -> 4.5 -> GPT-5.6 terra); I/O in the honeypot viewer",
    "hp_mc": "VEA = glm-evalgaming awareness monitor (v1 prompt, Sonnet 5.5 -> 4.5 -> GPT-5.6 terra); I/O in the honeypot viewer",
    "hp_abst": "VEA = glm-evalgaming awareness monitor (v1 prompt, Sonnet 5.5 -> 4.5 -> GPT-5.6 terra); I/O in the honeypot viewer",
}


def snip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n // 2] + f"\n…[{len(s) - n} chars not shown]…\n" + s[-n // 2 :]


def judge_rows(run: Path, task: str) -> dict:
    p = run / "judge" / f"{task}.jsonl"
    return {json.loads(line)["key"]: json.loads(line) for line in p.read_text().splitlines()} if p.exists() else {}


def jcall(name: str, rec: dict, call) -> dict:
    return call(
        name, rec.get("model") or "none (all judges refused)", messages=rec["messages"],
        output={"role": "assistant", "content": rec.get("output") or f"[no answer; attempts: {json.dumps(rec.get('tried'))}]"},
        params={"temperature": 0, "max_tokens": 800}, parsed=rec.get("parsed"), cost=rec.get("cost"),
        fallbacks=json.dumps(rec.get("tried")) if rec.get("tried") else None,
    )  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--expview", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--settings", type=lambda s: s.split(","), default=None)
    ap.add_argument("--name", default="prompt_ea")
    ap.add_argument("--labelled-only", action="store_true", help="only episodes with a VEA label")
    a = ap.parse_args()
    sys.path.insert(0, str(a.expview))
    from expview import Run, call

    e = pd.read_parquet(a.run / "scored_episodes.parquet")
    gen = pd.read_parquet(a.run / "episodes.parquet", columns=["tid", "gen_text"]).set_index("tid").gen_text
    prompts = pd.read_parquet(a.run / "prompts.parquet").set_index("ph")
    if a.settings:
        e = e[e.setting.isin(a.settings)]
    if a.labelled_only:
        e = e[e.vea.notna()]
    # only prompts that got judged, and episodes that carry a label or a judge score
    e = e[e.vea.notna() | e.lens.notna()]
    J = {t: judge_rows(a.run, t) for t in ("lens", "prompt", "vea", "sad")}
    run = Run(
        a.out / a.name,
        title=f"Prompt-side O-lens eval awareness ({a.name})",
        description=(
            "Does GLM-5.3's O-lens, read only on PROMPT tokens (before any generation), surface eval awareness, and does that "
            "predict verbalised eval awareness (VEA) in the rollout? Each item is one episode. **Environment** = the rendered "
            "prompt (`prompt.txt`); the prompt-side O-lens bullets the judge saw are in the lens judge's input, and every read "
            "(GLM SFT/RL lens, greedy + 4 samples) is in the olens_search link under Record. **Calls**: `glm transcript` = prompt + "
            "GLM's generated text "
            "exactly as rendered for the lens (tool outputs shortened); `lens judge` = Sonnet 5.5 on the prompt-side bullets "
            "(0-10); `prompt-only judge` = Sonnet 5.5 on the raw prompt, no lens (control); `vea monitor` = Sonnet 5.5 VEA label "
            "(glm-evalgaming v2 prompt) where this run made it. Labels: `vea`, `lens`, `prompt`, `kw_strict` (share of prompt "
            "positions with a strict eval keyword in any read). Writeup: docs/olens-prompt-eval-awareness.md in "
            "robbiethompson18/counterfactual-investigator-agent-with-olens."
        ),
        label_order=["setting", "cond", "vea", "lens", "prompt", "kw_strict", "vea_source"],
    )
    run.reset()
    for r in e.sort_values(["setting", "cond", "tid"]).itertuples():
        pt = prompts.loc[r.ph].prompt_text
        calls = [
            call("glm transcript", "z-ai/glm-5.3", messages=[{"role": "user", "content": pt}],
                 output={"role": "assistant", "content": snip(gen.get(r.tid, ""), GEN_SHOW)}),
        ]  # fmt: skip
        for task, name in (("lens", "lens judge"), ("prompt", "prompt-only judge")):
            if r.ph in J[task]:
                calls.append(jcall(name, J[task][r.ph], call))
        for task, name in (("vea", "vea monitor"), ("sad", "SAD answer parse")):
            if r.tid in J[task]:
                calls.append(jcall(name, J[task][r.tid], call))
        labels = {
            "setting": r.setting, "cond": r.cond, "vea": None if pd.isna(r.vea) else bool(r.vea),
            "lens": None if pd.isna(r.lens) else float(r.lens), "prompt": None if pd.isna(r.prompt) else float(r.prompt),
            "kw_strict": None if pd.isna(r.kw_strict) else round(float(r.kw_strict), 3), "vea_source": r.vea_source or "none",
        }  # fmt: skip
        run.add(
            id=r.tid, group=r.setting, labels=labels, calls=calls, env_files={"prompt.txt": pt},
            extra={"olens_search": VIEW.format(tid=quote(r.tid)), "docent": r.docent, "prompt_hash": r.ph,
                   "label_note": LABEL_NOTE.get(r.setting, ""), "n_prompt_positions_pooled": int(r.n_prompt_pos_pooled)},
        )  # fmt: skip
    print(f"{len(e)} items -> {a.out / a.name}")


if __name__ == "__main__":
    main()
