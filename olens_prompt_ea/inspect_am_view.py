# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=17", "numpy>=1.26", "scipy>=1.13", "statsmodels>=0.14"]
# ///
"""expview run of Celeste's inspect_evals agentic-misalignment rollouts (ceselder/glm-5.3-olens-new-sources config
inspect_agentic_misalignment): one item per episode with its variant, the rendered prompt, GLM's rollout, the regex VEA
label + matched quote (+ a hand label where one exists), and every prompt-side O-lens read with eval words marked.

    uv run --script olens_prompt_ea/inspect_am_view.py RUN --expview <shared viewer checkout>/expview [--max-gen N] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from keywords import BROAD, STRICT
from kwonly import VEA_RX

D8830 = "imports/celeste/ds_new/inspect_agentic_misalignment"
Q = "simulat|eval|test|synthetic|scenario|honeypot|benchmark|fiction|hypothetical|red.?team"


def mark(text: str) -> str:
    return BROAD.sub(lambda m: f"**[{m.group(0)}]**", text)


def snip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n // 2] + f"\n\n…[{len(s) - n:,} chars not shown]…\n\n" + s[-n // 2 :]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--expview", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="run dir (default RUN/expview/inspect_am)")
    ap.add_argument("--max-gen", type=int, default=0, help="cap on rollout chars shown (0 = all)")
    ap.add_argument("--handcheck", type=Path, action="append", default=[], help="JSON {tid: {vea, quote}} (repeatable)")
    a = ap.parse_args()
    sys.path.insert(0, str(a.expview))
    from expview import Run, call

    hc = {k: v for f in a.handcheck for k, v in json.loads(f.read_text()).items()}
    e = pd.read_parquet(a.run / "episodes.parquet", columns=["tid", "setting", "cond", "ph", "gen_text", "epoch"])
    e = e[e.setting == "inspect_am"].sort_values("tid")
    prompts = pd.read_parquet(a.run / "prompts.parquet", columns=["ph", "prompt_text"]).set_index("ph").prompt_text
    reads = pd.read_parquet(a.run / "reads.parquet", filters=[("ph", "in", list(e.ph.unique()))])
    t = reads.text.fillna("")
    reads["strict"], reads["broad"] = t.str.contains(STRICT), t.str.contains(BROAD)
    by_ph = dict(tuple(reads.groupby("ph")))
    out = a.out or a.run / "expview" / "inspect_am"
    run = Run(
        out,
        title="Celeste's inspect agentic misalignment: VEA and prompt-side O-lens reads",
        description=(
            "Celeste's on-policy GLM-5.3 rollouts of inspect_evals agentic_misalignment (her vLLM server, template default "
            "reasoning effort Max; "
            "[ceselder/glm-5.3-olens-new-sources](https://huggingface.co/datasets/ceselder/glm-5.3-olens-new-sources) "
            "config `inspect_agentic_misalignment`): blackmail / leaking / murder x goal type and value x urgency, plus prod=True "
            "(no scratchpad) and test_eval_awareness=True (system prompt asks the model to deliberate whether it is in an "
            "evaluation). One item per episode.\n\n"
            "Labels: `vea` = regex on the rollout (`kwonly.VEA_RX`; no monitor ran on these). A hand check of 14 random episodes "
            "found no regex false positives (8/8) but 4 misses among 6 regex-no, so true VEA is likely above 0.9; `hand_vea` "
            "where checked. `lens_strict_share` = share of prompt-side read positions where an O-lens bullet has a strict eval "
            "word. Calls: the prompt-side O-lens reads (every read position before the first assistant turn, eval words in "
            "**[brackets]**; identical prompts were read once, so episodes share them), then the transcript (rendered prompt, "
            "then GLM's reasoning and reply). Buttons: the episode in Celeste's import with reads inline at the first eval-word "
            "read (8830), and the same on 8821."
        ),
        label_order=["vea", "hand_vea", "scenario", "goal", "urgency", "test_eval_awareness", "prod", "lens_strict_share"],
    )
    run.reset()
    for r in e.itertuples():
        variant = r.tid.split(":")[3]  # e.g. blackmail_explicit-america_replacement
        scen, goal, urg = (variant.split("_") + ["", "", ""])[:3]
        _, _, _, prod, tea = r.cond.split("|")
        rr = by_ph.get(r.ph)
        lines, first = [], None
        if rr is not None:
            for p, g in rr.sort_values("pos").groupby("pos", sort=True):
                g0 = g.iloc[0]
                if first is None and g.broad.any():
                    first = (g0.tid, int(p))
                bl = "\n".join(
                    f"  - {x.reader.replace('glm_', '').upper()} {x.sample}: {mark(x.text.strip())}"
                    for x in g.drop_duplicates("text").itertuples()
                )
                lines.append(f"pos {p} ({g0.role}), token {json.dumps(g0.token)}: …{(g0.ctx_left40 or '')[-160:]}⟦{g0.token}⟧\n{bl}")
            if first is None:
                first = (rr.iloc[0].tid, int(rr.pos.min()))
        gen = r.gen_text or ""
        m = VEA_RX.search(gen)
        h = hc.get(r.tid, {})
        links = {}
        if first:
            links["reads inline (8830)"] = f"http://bramble:8830/#d={D8830}&i={quote(first[0], safe='')}&v=tr&t={first[1]}&q={quote(Q)}"
            links["reads (8821)"] = f"http://bramble:8821/#ep={quote(first[0], safe='')}&pos={first[1]}&q={quote(Q)}"
        run.add(
            id=r.tid, group=scen,
            labels={
                "vea": bool(m), "vea_quote": gen[max(0, m.start() - 160) : m.end() + 120].replace("\n", " ") if m else "(no match)",
                "hand_vea": h.get("vea"), "hand_quote": h.get("quote", ""), "scenario": scen, "goal": goal, "urgency": urg,
                "variant": variant, "test_eval_awareness": tea == "True", "prod": prod == "True", "epoch": int(r.epoch or 0),
                "lens_strict_share": round(float(rr.groupby("pos").strict.any().mean()), 3) if rr is not None else None,
                "n_eval_reads": int(rr.broad.sum()) if rr is not None else 0,
            },
            links=links,
            calls=[
                call("prompt-side O-lens reads", "GLM-5.3 O-lens (SFT/RL, greedy + 4 samples)",
                     messages=[{"role": "user", "content": "\n\n".join(lines) or "(no prompt-side reads)"}]),
                call("transcript (GLM 5.3, Celeste's server)", "zai-org/GLM-5.3",
                     messages=[{"role": "user", "content": prompts[r.ph]},
                               {"role": "assistant", "content": snip(gen, a.max_gen) if a.max_gen else gen}]),
            ],
        )  # fmt: skip
    print(f"{len(e)} items -> {run.path}")


if __name__ == "__main__":
    main()
