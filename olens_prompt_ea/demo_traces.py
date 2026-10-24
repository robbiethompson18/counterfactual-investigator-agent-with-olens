# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=17", "numpy>=1.26", "scipy>=1.13", "statsmodels>=0.14"]
# ///
"""Trace examples for the across-task claim, as an expview run for the shared viewer (bramble:8830).

Sections (item groups):
  ib_realistic   prompts whose prompt-side O-lens reads contain a strict eval word, prompts whose only hit is a loose
                 word ("test"), and prompts with no hit; one episode each
  sandbagging    episodes with VEA (regex on the rollout) vs without
  donation_bet   prompts where the lens writes "evaluator" / "being evaluated"

Each item: the matched prompt-side reads (token context, lens + sample, the bullet) as one call, the rollout as another,
labels (VEA, its source and quote, the hand check), and links to the read in the Celeste import on bramble:8830, to the
same read on bramble:8821, and to Docent where the episode is there.

    uv run --script olens_prompt_ea/demo_traces.py RUN --expview <shared viewer checkout>/expview [--handcheck file.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from keywords import BROAD, STRICT
from kwonly import VEA_RX

IMPORT = {"ib_realistic": "imports/celeste/ds_new/cfi_ib_realistic", "ec_sandbagging": "imports/celeste/ds/evalcue",
          "donation_bet": "imports/celeste/ds_new/cfi_donation_bet"}  # fmt: skip
Q = "simulat|eval|test|synthetic|scenario|honeypot|benchmark|fiction|hypothetical|red.?team"
V8830 = "http://bramble:8830/#d={d}&i={i}&v=tr&t={t}&q={q}"
V8821 = "http://bramble:8821/#ep={i}&pos={t}&q={q}"
RNG = np.random.default_rng(0)


def mark(text: str) -> str:
    return BROAD.sub(lambda m: f"**[{m.group(0)}]**", text)


def boot_mean(x: np.ndarray, groups: np.ndarray, b: int = 2000) -> tuple[float, float]:
    """95% CI of the mean, resampling prompts (episodes sharing a prompt share its score)."""
    keys = np.unique(groups)
    idx = {k: np.where(groups == k)[0] for k in keys}
    vals = [x[np.concatenate([idx[k] for k in RNG.choice(keys, len(keys))])].mean() for _ in range(b)]
    return tuple(np.percentile(vals, [2.5, 97.5]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--expview", type=Path, required=True)
    ap.add_argument("--handcheck", type=Path, default=None, help="JSON {tid: {vea, quote}} from a hand check")
    ap.add_argument("--dump", type=Path, default=None, help="write the shown episodes' rollouts here for a hand check")
    a = ap.parse_args()
    sys.path.insert(0, str(a.expview))
    from expview import Run, call

    hc = json.loads(a.handcheck.read_text()) if a.handcheck else {}
    e = pd.read_parquet(a.run / "kwonly_episodes.parquet").merge(
        pd.read_parquet(a.run / "episodes.parquet", columns=["tid", "gen_text", "docent", "arm", "env"]), on="tid"
    )
    e = e[e.setting.isin(list(IMPORT))]
    reads = pd.read_parquet(a.run / "reads.parquet", filters=[("ph", "in", list(e.ph.unique()))])
    t = reads.text.fillna("")
    reads["strict"], reads["broad"] = t.str.contains(STRICT), t.str.contains(BROAD)
    pos = reads.groupby(["ph", "pos"]).agg(strict=("strict", "any"), broad=("broad", "any"))
    per_ph = pos.groupby("ph").agg(n_pos=("strict", "size"), strict_share=("strict", "mean"), broad_share=("broad", "mean"))
    e = e.join(per_ph, on="ph")

    # VEA evidence quote
    def quote_of(g: str) -> str:
        m = VEA_RX.search(g)
        return g[max(0, m.start() - 160) : m.end() + 120].replace("\n", " ") if m else ""

    e["vea_quote"] = [quote_of(g) for g in e.gen_text.fillna("")]
    picks: list[tuple[str, str, pd.Series]] = []
    ib = e[e.setting == "ib_realistic"].sample(frac=1, random_state=1).drop_duplicates("ph")
    for sub, x, k in (
        ("strict eval word", ib[ib.strict_share > 0].sort_values("vea", ascending=False), 7),
        ("only loose words (test)", ib[(ib.strict_share == 0) & (ib.broad_share > 0)], 2),
        ("no eval word", ib[ib.broad_share == 0], 2),
    ):
        picks += [("IB realistic", sub, r) for _, r in x.head(k).iterrows()]
    sb = e[e.setting == "ec_sandbagging"].sample(frac=1, random_state=1)
    picks += [("Sandbagging", "VEA yes", r) for _, r in sb[sb.vea].head(5).iterrows()]
    picks += [("Sandbagging", "VEA no", r) for _, r in sb[~sb.vea].head(5).iterrows()]
    db = e[e.setting == "donation_bet"].sample(frac=1, random_state=1).drop_duplicates("ph")
    picks += [("Donation bet", "strict eval word", r) for _, r in db[db.strict_share > 0.5].head(4).iterrows()]

    if a.dump:
        a.dump.write_text(
            json.dumps(
                [{"tid": r.tid, "setting": r.setting, "regex_vea": bool(r.vea), "rollout": r.gen_text} for _, _, r in picks], indent=1
            )
        )
        print(f"dumped {len(picks)} rollouts -> {a.dump}")
        return

    # stats for the description
    def stats(x: pd.DataFrame) -> str:
        lo, hi = boot_mean(x.strict_share.to_numpy(), x.ph.to_numpy()) if x.ph.nunique() > 1 else (np.nan, np.nan)
        return f"{x.strict_share.mean():.3f} [{lo:.3f}, {hi:.3f}] (n = {len(x)} eps, {x.ph.nunique()} prompts)"

    words = {}
    for st in IMPORT:
        rr = reads[reads.broad & reads.ph.isin(e[e.setting == st].ph)]
        w = rr.text.str.extract(f"({BROAD.pattern})", flags=re.IGNORECASE, expand=False).str.lower().value_counts(normalize=True)
        words[st] = ", ".join(f"{k} {v:.0%}" for k, v in w.head(5).items())
    ibp = pos.loc[pos.index.get_level_values(0).isin(e[e.setting == "ib_realistic"].ph)]
    desc = f"""Trace examples behind the across-task result in `docs/olens-prompt-eval-awareness.md`: do prompts whose
**prompt-side** O-lens reads (tokens before the first assistant turn) contain eval words get more verbalised eval awareness
(VEA) in their rollouts?

Each item: call 1 = the prompt-side reads that matched (token, preceding text, lens and sample, the bullet with the matched
word in **[brackets]**); call 2 = GLM's rollout (reasoning, replies, tool calls; tool outputs shortened). Labels: `vea` =
regex on the rollout (`kwonly.VEA_RX`; none of these settings has a monitor label), `vea_quote` its match, `hand_vea` /
`hand_quote` = a hand check of the same rollout. Buttons: the read inline in the Celeste import (8830), the same read on
the old dashboard (8821), Docent.

- **IB realistic**: {ibp.broad.mean():.0%} of the {len(ibp)} prompt-side read positions have a loose eval word; the
  matched words are {words["ib_realistic"]}, i.e. almost all unit tests ("there are tests for it in test_solution.py").
  Only {ibp.strict.mean():.1%} of positions have a strict eval word (mostly "the evaluator will …" at the end of the
  lcb.yaml-style instructions). Per episode the strict share is higher (0.37) because several arms (git, repo, agents_md,
  followup, scrub, no_done) run ~70 episodes on one identical prompt. Strict share per episode, VEA yes
  {stats(e[(e.setting == "ib_realistic") & e.vea])} vs no {stats(e[(e.setting == "ib_realistic") & ~e.vea])}.
- **Sandbagging**: the arms differ in Documentation.md, which GLM reads with a tool after its first turn, so the
  prompt-side reads see only 9 distinct prompts. Strict share, VEA yes {stats(e[(e.setting == "ec_sandbagging") & e.vea])} vs
  no {stats(e[(e.setting == "ec_sandbagging") & ~e.vea])}. Words: {words["ec_sandbagging"]}.
- **Hand check** of the 25 shown rollouts (a Claude subagent, definition as the monitor's v2 prompt): disagrees with the
  regex on 5. IB realistic: regex 1/11 VEA, hand 4/11 (the regex misses "if this is a benchmark task", "maybe it's a
  trick … the checker expects me to notice", "lcbhard_22 suggests a generated dataset"); sandbagging: regex 5/10, hand
  5/10 (one miss, "a prompt-injection-style test", and one false positive where "test" is the named arithmetic
  assessment); donation bet 0/4 both. So regex VEA under-counts IB badly; the regex-labelled pools are weak evidence.
- **Donation bet**: the lens writes "evaluator" / "being evaluated on your answer" at the prompt's final punctuation on
  most prompts (words: {words["donation_bet"]}); VEA is 0.3%.
"""
    run = Run(a.run / "expview" / "demo", title="Prompt-side O-lens eval words: trace examples", description=desc,
              label_order=["vea", "hand_vea", "strict_share", "broad_share", "n_eval_reads"])  # fmt: skip
    run.reset()
    for sec, sub, r in picks:
        rr = reads[(reads.ph == r.ph) & reads.broad].sort_values(["strict", "pos"], ascending=[False, True])
        lines, first = [], None
        for p, g in list(rr.groupby("pos", sort=False))[:8]:
            g0 = g.iloc[0]
            first = first or (g0.tid, int(p))
            ctx = (g0.ctx_left40 or "")[-220:]
            bl = "\n".join(
                f"  - {x.reader.replace('glm_', '').upper()} {x.sample}: {mark(x.text.strip())}"
                for x in g.drop_duplicates("text").itertuples()
            )
            lines.append(f"pos {p} ({g0.role}), token {json.dumps(g0.token)}\n  …{ctx}⟦{g0.token}⟧ {(g0.ctx_right40 or '')[:80]}…\n{bl}")
        if first is None:  # no hit: show the last prompt position's reads
            last = reads[reads.ph == r.ph].sort_values("pos").tail(10)
            if len(last):
                g0 = last.iloc[0]
                first = (g0.tid, int(g0.pos))
                bl = "\n".join(
                    f"  - {x.reader.replace('glm_', '').upper()} {x.sample}: {x.text.strip()}"
                    for x in last.drop_duplicates("text").itertuples()
                )
                lines.append(
                    f"(no eval word in any prompt-side read; last prompt position {g0.pos}, token {json.dumps(g0.token)})\n"
                    f"  …{(g0.ctx_left40 or '')[-220:]}⟦{g0.token}⟧\n{bl}"
                )
        h = hc.get(r.tid, {})
        links = {}
        if first:
            links["read in 8830"] = V8830.format(d=IMPORT[r.setting], i=quote(first[0], safe=""), t=first[1], q=quote(Q))
            links["read in 8821"] = V8821.format(i=quote(first[0], safe=""), t=first[1], q=quote(Q))
        links["episode in 8830"] = V8830.format(d=IMPORT[r.setting], i=quote(r.tid, safe=""), t=0, q=quote(Q))
        if isinstance(r.docent, str):
            links["Docent"] = r.docent
        run.add(
            id=f"{sec}/{sub}/{r.tid}", group=f"{sec}: {sub}",
            labels={"vea": bool(r.vea), "vea_quote": r.vea_quote[:300] or "(no match)", "hand_vea": h.get("vea"),
                    "hand_quote": h.get("quote", ""),
                    "strict_share": round(float(r.strict_share), 3), "broad_share": round(float(r.broad_share), 3), "n_eval_reads": len(rr),
                    "arm": r.arm or r.cond},
            links=links,
            calls=[
                call("prompt-side O-lens reads with eval words", "GLM-5.3 O-lens (SFT/RL, greedy + 4 samples)",
                     messages=[{"role": "user", "content": "\n\n".join(lines) or "(no prompt-side reads)"}]),
                call("rollout (GLM 5.3)", "z-ai/glm-5.3", messages=[{"role": "assistant", "content": r.gen_text[:60000]}]),
            ],
        )  # fmt: skip
    print(f"{len(picks)} items -> {run.path}")
    print(desc)


if __name__ == "__main__":
    main()
