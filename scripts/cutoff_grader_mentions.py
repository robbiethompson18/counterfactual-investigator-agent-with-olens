# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "numpy>=1.26"]
# ///
"""Regex search for grader / evaluation / correctness-checking words in the cutoff_qa O-lens reads, no tool vs submit_answer tool.

Runs (each a list of olens_reads run dirs, merged and deduped on (id, token_index, layer)):

- A, no tool: Qwen3.6-27B gets the bare question. 2026-10-29_cutoff{200,800}-tail (the six tail tokens through the newline after
  `assistant`) + 2026-10-30_cutoff-fill-kylie (the question's punctuation and the four think-block tokens), 999 items.
- A2, no tool, second 1k (bank v4): 2026-10-30_cutoff-new1k (punctuation + six tail tokens, no think block). A replication of A.
- B, Robbie's v3 tool variant: the same question under a system block with the `submit_answer` schema (cutoff_qa_tool_inline),
  2026-10-30_cutoff-v3tool-punct, 1,849 items, user-turn punctuation + last ten tokens. The system block itself was not read.

Every O-lens bullet of a read is matched case-insensitive and word-bounded; a read hits a family when any bullet does. Echo control: a
hit is an echo when the same family matches the prompt within 40 tokens of the read token, or anywhere in the system block (B's tool
schema); otherwise it is novel. Position classes are matched across runs (q_inner / q_end / instr_punct / instr_end / im_end / header
/ think), so B's extra prompt tokens don't enter the comparison. CIs: bootstrap over items (each run resampled separately).

    uv run --script scripts/cutoff_grader_mentions.py --out data/runs/cutoff_grader_mentions/2026-10-30_regex \
        --run A=DIR/2026-10-29_cutoff200-tail,DIR/2026-10-29_cutoff800-tail,DIR/2026-10-30_cutoff-fill-kylie \
        --run A2=DIR/2026-10-30_cutoff-new1k --run B=DIR/2026-10-30_cutoff-v3tool-punct \
        --cases A=.../cutoff-qa-inline-inputs-1k/cases.jsonl --cases A2=... --cases B=...
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "expview"))
from expview import Run, call  # ty: ignore[unresolved-import]  # expview/ is on sys.path above, not a package

FAMILIES: dict[str, dict[str, str]] = {
    "grader": {
        "grader/grading/graded": r"\bgrad(?:er|ers|ing|ed)\b",
        "autograder": r"\bauto-?grad\w*",
        "scorer/scoring": r"\bscor(?:er|ers|ing)\b",
        "score(s/d)": r"\bscor(?:e|es|ed)\b",
        "judge": r"\bjudg(?:e|es|ed|ing)\b",
        "rubric": r"\brubrics?\b",
        "reward": r"\brewards?\b|\brewarded\b",
        "checker": r"\bcheckers?\b",
        "marked/marking": r"\bmark(?:ed|ing)\b",
    },
    "eval_test": {
        "evaluat*": r"\bevaluat\w*",
        "eval(s)": r"\bevals?\b",
        "test": r"\btest\b",
        "tests": r"\btests\b",
        "tested/testing": r"\btest(?:ed|ing)\b",
        "being tested/evaluated": r"\bbeing (?:tested|evaluated|graded|assessed)\b",
        "quiz": r"\bquiz(?:zes|zed|zing)?\b",
        "exam": r"\bexam(?:s|ination|inations)?\b",
        "benchmark": r"\bbenchmarks?\b",
        "assess*": r"\bassess\w*",
    },
    "correctness": {
        "correct answer": r"\bcorrect answers?\b",
        "right answer": r"\bright answers?\b",
        "checked": r"\bchecked\b",
        "verified": r"\bverified\b",
    },
    # reported apart: the strict "it's a test" phrasings of scripts/olens_keyword_search.py, and the tool's own words
    "eval_strict": {
        "testing me/us/whether": r"\btesting (?:me|us|you|whether|if)\b",
        "this is a test": r"\b(?:this is|it's|just) (?:a|an) (?:test|eval|evaluation|quiz)\b",
        "test/quiz/exam question": r"\b(?:test|quiz|exam) questions?\b",
        "trivia question": r"\btrivia questions?\b",
        "designed to test / trap": r"\bdesigned to (?:test|see|check|probe)\b|\b(?:a|the) trap\b|\btrick question\b",
        "knowledge cutoff check": r"\b(?:test|check|probe)s? (?:my|your|the model's) knowledge\b",
    },
    # the prompt-continuation form: "Your answer will be graded on accuracy."
    "grader_strict": {
        "will be graded/judged/...": (
            r"\b(?:will|would|is|are|be|being|get|gets) (?:graded|judged|scored|evaluated|assessed|marked|rewarded|penalized|checked)\b"
        ),
        "graded on/for/by": r"\b(?:graded|judged|scored|evaluated|marked) (?:on|for|by|based on|according to)\b",
    },
    "submit": {"submit/submission": r"\bsubmi(?:t|ts|tted|tting|ssion)\b"},
}
FAM_RX = {f: re.compile("|".join(f"(?:{p})" for p in pats.values()), re.IGNORECASE) for f, pats in FAMILIES.items()}
SUB_RX = {f: {k: re.compile(p, re.IGNORECASE) for k, p in pats.items()} for f, pats in FAMILIES.items()}
MAIN = ["grader", "eval_test", "correctness"]
EXTRA = ["eval_strict", "grader_strict"]
CLASSES = ["q_inner", "q_end", "instr_punct", "instr_end", "im_end", "header", "think"]
WIN = 40


def jsonl(p: Path):
    with open(p) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_run(dirs: list[Path]) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, int], dict[str, str]]:
    """Reads (one row per id x token_index x layer), id -> prompt tokens, id -> end of the system block, id -> item."""
    toks: dict[str, list[str]] = {}
    sys_end: dict[str, int] = {}
    item: dict[str, str] = {}
    for d in dirs:
        for f in (d / "reads.transcripts.jsonl", d / "requests.jsonl"):
            if f.exists():
                for t in jsonl(f):
                    item.setdefault(t["id"], t["item"])
                    if "tokens" in t and t["id"] not in toks:
                        toks[t["id"]] = list(t["tokens"])
                        sec = [s for s in t.get("sections", []) if s[0] == "system"]
                        sys_end[t["id"]] = sec[0][2] if sec else 0
    rows, seen = [], set()
    for d in dirs:
        for r in jsonl(d / "reads.jsonl"):
            k = (r["id"], r["token_index"], r["layer"])
            if k in seen or not r.get("bullets"):
                continue
            seen.add(k)
            rows.append(
                {
                    "id": r["id"],
                    "pos": r["token_index"],
                    "layer": r["layer"],
                    "token": r["token"],
                    "bullets": [b["text"] for b in r["bullets"]],
                }
            )
    df = pd.DataFrame(rows)
    # tokens past the transcript's end (A's 2026-10-29 renders stop at the thinking-on `<think>\n`) come from the fill rows
    for (i, p), tk in df.groupby(["id", "pos"]).token.first().items():
        t = toks[i]
        if p >= len(t):
            t.extend([None] * (p + 1 - len(t)))
        t[p] = tk
    for i, t in toks.items():  # B's system block: everything before the user's turn
        sys_end[i] = next(j for j in range(len(t) - 1) if t[j] == "<|im_start|>" and t[j + 1] == "user")
    return df, toks, sys_end, item


def pos_class(t: list[str], p: int, start: int) -> str:
    e = max(j for j, x in enumerate(t) if x == "<|im_end|>")
    if p == e:
        return "im_end"
    if p > e:
        return "header" if p <= e + 4 else "think"
    if p == e - 1:
        return "instr_end"
    qend = next((j for j in range(start, len(t) - 1) if t[j + 1] is not None and t[j + 1].strip() == "If" and j < e), None)
    if qend is None:
        return "q_inner"
    return "q_end" if p == qend else ("q_inner" if p < qend else "instr_punct")


def ctx_text(t: list[str], p: int, sys_end: int) -> str:
    win = "".join(x or "" for x in t[max(0, p - WIN) : p + WIN + 1])
    return "".join(x or "" for x in t[:sys_end]) + " ‖ " + win


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, col: str, n: int = 2000, seed: int = 0) -> dict:
    """Reads-level rate difference a - b with items resampled (per-item sums, ratio of totals)."""
    rng = np.random.default_rng(seed)
    ga, gb = a.groupby("item")[col].agg(["sum", "count"]), b.groupby("item")[col].agg(["sum", "count"])
    ra, rb = ga["sum"].sum() / ga["count"].sum(), gb["sum"].sum() / gb["count"].sum()
    sa, ca, sb, cb = ga["sum"].values, ga["count"].values, gb["sum"].values, gb["count"].values
    ds = []
    for _ in range(n):
        ia, ib = rng.integers(0, len(sa), len(sa)), rng.integers(0, len(sb), len(sb))
        ds.append(sa[ia].sum() / ca[ia].sum() - sb[ib].sum() / cb[ib].sum())
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return {"rate_a": float(ra), "rate_b": float(rb), "diff": float(ra - rb), "lo": float(lo), "hi": float(hi)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=dir1,dir2,...")
    ap.add_argument("--cases", action="append", default=[], help="NAME=cases.jsonl (item -> gate_class, modal)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expview-items", type=int, default=60, help="items per run in the expview sample")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs = {k: [Path(x) for x in v.split(",")] for k, v in (s.split("=", 1) for s in a.run)}
    cases = {k: {c["item"]: c for c in jsonl(Path(v))} for k, v in (s.split("=", 1) for s in a.cases)}

    frames, toks_all, sys_all, prompts = [], {}, {}, {}
    for name, dirs in runs.items():
        df, toks, sys_end, item = load_run(dirs)
        df["run"] = name
        df["item"] = df.id.map(item)
        df["cls"] = [pos_class(toks[i], p, sys_end[i]) for i, p in zip(df.id, df.pos)]
        ctx = {(i, p): ctx_text(toks[i], p, sys_end[i]) for i, p in df[["id", "pos"]].drop_duplicates().itertuples(index=False)}
        df["ctx"] = [ctx[(i, p)] for i, p in zip(df.id, df.pos)]
        cs: dict = cases.get(name, {})
        df["gate_class"] = df.item.map({k: c["meta"].get("gate_class") for k, c in cs.items()})
        df["modal"] = df.item.map({k: c["modal"] for k, c in cs.items()})
        for i in df.id.unique():
            toks_all[(name, i)] = toks[i]
            sys_all[(name, i)] = sys_end[i]
        sample_id = df.id.iloc[0]
        prompts[name] = "".join(x or "" for x in toks[sample_id])
        frames.append(df)
        print(name, len(df), "reads", df.item.nunique(), "items", dict(df.cls.value_counts()), flush=True)
    r = pd.concat(frames, ignore_index=True)
    text = r.bullets.map(lambda bs: "\n".join(bs))
    for fam, rx in FAM_RX.items():
        r[f"hit_{fam}"] = text.str.contains(rx)
        r[f"ctx_{fam}"] = r.ctx.str.contains(rx)
        r[f"nov_{fam}"] = r[f"hit_{fam}"] & ~r[f"ctx_{fam}"]
    r["nov_any"] = r[[f"nov_{f}" for f in MAIN]].any(axis=1)
    r["hit_any"] = r[[f"hit_{f}" for f in MAIN]].any(axis=1)

    # the prompts as rendered, so the report can quote the grading-relevant wording
    (a.out / "prompts.json").write_text(json.dumps(prompts, indent=1))
    ctx_words = {n: {f: sorted({m.group(0).lower() for m in rx.finditer(p)}) for f, rx in FAM_RX.items()} for n, p in prompts.items()}

    fams = [*MAIN, *EXTRA, "submit", "any"]
    rates = []
    for keys, g in [
        *(((n, "ALL", "ALL"), g) for n, g in r.groupby("run")),
        *(((n, c, "ALL"), g) for (n, c), g in r.groupby(["run", "cls"])),
        *(((n, c, str(lay)), g) for (n, c, lay), g in r.groupby(["run", "cls", "layer"])),
        *(((n, "ALL", f"gate={gc}"), g) for (n, gc), g in r.groupby(["run", "gate_class"])),
    ]:
        n_items = g.item.nunique()
        for f in fams:
            hc, nc = (f"hit_{f}", f"nov_{f}") if f != "any" else ("hit_any", "nov_any")
            k, kn = int(g[hc].sum()), int(g[nc].sum())
            items_n = int(g[g[nc]].item.nunique())
            lo, hi = wilson(kn, len(g))
            rates.append(
                {
                    "run": keys[0],
                    "cls": keys[1],
                    "split": keys[2],
                    "family": f,
                    "reads": len(g),
                    "items": n_items,
                    "hit": k,
                    "novel": kn,
                    "novel_rate": kn / len(g),
                    "novel_lo": lo,
                    "novel_hi": hi,
                    "items_with_novel": items_n,
                    "items_frac": items_n / n_items,
                }
            )
    rates = pd.DataFrame(rates)
    rates.to_csv(a.out / "rates.csv", index=False)

    # A vs B (and A2 vs B) on matched classes
    comps = []
    for x, items in [(n, "all") for n in runs if n != "B"] + [("A", "shared")]:
        rb = r[r.run == "B"]
        ra = r[r.run == x]
        if items == "shared":  # the items both runs asked, so the item mix is the same
            common = set(ra.item) & set(rb.item)
            ra, rb = ra[ra.item.isin(common)], rb[rb.item.isin(common)]
        shared = sorted(set(ra.cls) & set(rb.cls), key=CLASSES.index)
        for cl in [*shared, "MATCHED"]:
            ga = ra[ra.cls.isin(shared) if cl == "MATCHED" else ra.cls == cl]
            gb = rb[rb.cls.isin(shared) if cl == "MATCHED" else rb.cls == cl]
            for f in [*MAIN, *EXTRA, "any"]:
                comps.append(
                    {
                        "a": x,
                        "b": "B",
                        "items": items,
                        "n_items": ga.item.nunique(),
                        "cls": cl,
                        "family": f,
                        **boot_diff(ga, gb, f"nov_{f}"),
                    }
                )
    comps = pd.DataFrame(comps)
    comps.to_csv(a.out / "a_vs_b.csv", index=False)

    subs = []
    for (n,), g in r.groupby(["run"]):
        gt = text[g.index]
        for f, d in SUB_RX.items():
            for s, srx in d.items():
                h = gt.str.contains(srx)
                c = g.ctx.str.contains(srx)
                subs.append({"run": n, "family": f, "pattern": s, "hit": int(h.sum()), "novel": int((h & ~c).sum()), "reads": len(g)})
    subs = pd.DataFrame(subs)
    subs.to_csv(a.out / "patterns.csv", index=False)

    # every novel hit bullet, with its read's coordinates
    hits = []
    for idx in r.index[r[[f"nov_{f}" for f in [*MAIN, *EXTRA]]].any(axis=1)]:
        row = r.loc[idx]
        for bi, b in enumerate(row.bullets):
            fs = [f for f in [*MAIN, *EXTRA] if row[f"nov_{f}"] and FAM_RX[f].search(b)]
            if fs:
                pats = [s for f in fs for s, srx in SUB_RX[f].items() if srx.search(b)]
                hits.append(
                    {
                        "run": row.run,
                        "id": row.id,
                        "item": row["item"],
                        "pos": int(row.pos),
                        "token": row.token,
                        "cls": row.cls,
                        "layer": int(row.layer),
                        "gate_class": row.gate_class,
                        "modal": row.modal,
                        "families": fs,
                        "patterns": pats,
                        "bullet_index": bi,
                        "bullet": b,
                    }
                )
    pd.DataFrame(hits).to_json(a.out / "novel_hits.jsonl", orient="records", lines=True)

    write_expview(a.out / "expview", r, hits, toks_all, sys_all, a.expview_items)
    summary = {
        "families": FAMILIES,
        "window_tokens": WIN,
        "prompt_family_words": ctx_words,
        "reads": {n: int((r.run == n).sum()) for n in runs},
        "items": {n: int(r[r.run == n].item.nunique()) for n in runs},
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print_report(rates, comps, subs, ctx_words)


def write_expview(path: Path, r: pd.DataFrame, hits: list[dict], toks: dict, sys_end: dict, per_run: int) -> None:
    run = Run(
        path,
        title="cutoff_qa O-lens grader mentions",
        description=(
            "O-lens reads of cutoff_qa prompts (Qwen3.6-27B), no tool (A, A2) vs the submit_answer tool (B), items sampled "
            "from those with a *novel* grader / eval-test / correctness regex hit (family absent within 40 tokens and in "
            "the system block). Hit reads are drawn inline; every other read is click-only. Labels: the run, the "
            "item's gate class, which families hit."
        ),
        tokens_section="O-lens reads",
    )
    run.reset()
    by_item: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for h in hits:
        if set(h["families"]) & {*MAIN, "grader_strict"}:
            by_item[(h["run"], h["id"])].append(h)
    rnd = random.Random(0)
    for name in sorted(r.run.unique()):
        keys = [k for k in by_item if k[0] == name]
        rnd.shuffle(keys)
        grad = [k for k in keys if any({"grader", "grader_strict"} & set(h["families"]) for h in by_item[k])]
        rest = [k for k in keys if k not in set(grad)]
        pick = grad[: per_run * 2 // 3] + rest[: per_run - min(len(grad), per_run * 2 // 3)]  # grader hits first
        for k in sorted(pick):
            g = r[(r.run == name) & (r.id == k[1])]
            hitset = {(h["pos"], h["layer"]) for h in by_item[k]}
            reads: dict[str, list] = defaultdict(list)
            for row in g.sort_values(["pos", "layer"]).itertuples():
                hit = (row.pos, row.layer) in hitset
                lines = sorted(row.bullets, key=lambda b: not any(FAM_RX[f].search(b) for f in MAIN)) if hit else row.bullets
                reads[str(row.pos)].append(
                    {
                        "tool": "O-lens",
                        "label": f"L{row.layer}" + (" ★hit" if hit else ""),
                        "lines": [[b, None] for b in lines],
                        **({"inline": True} if hit else {}),
                    }
                )
            t = [x or "" for x in toks[k]]
            se = sys_end[k]
            sections = ([["system (tool schema)", 0, se]] if se else []) + [["prompt", se, len(t)]]
            first = by_item[k][0]
            fams = sorted({f for h in by_item[k] for f in h["families"]})
            run.add(
                id=f"{name}/{first['item']}",
                group=name,
                labels={
                    "run": name,
                    "gate_class": first["gate_class"] or "",
                    "families": ",".join(fams),
                    "hit_reads": len(hitset),
                    "classes": ",".join(sorted({h["cls"] for h in by_item[k]})),
                },
                calls=[call("subject prompt (as rendered)", "Qwen/Qwen3.6-27B", messages=[{"role": "user", "content": "".join(t)}])],
                extra={
                    "read_id": k[1],
                    "qwen_modal": first["modal"],
                    "hits": [{kk: h[kk] for kk in ("pos", "token", "cls", "layer", "patterns", "bullet")} for h in by_item[k]],
                },
                tokens={
                    "tokens": t,
                    "sections": sections,
                    "reads": dict(reads),
                    "legend": "★hit = a bullet matches grader / eval-test / correctness and the family isn't in the context.",
                },
            )


def print_report(rates: pd.DataFrame, comps: pd.DataFrame, subs: pd.DataFrame, ctx_words: dict) -> None:
    pd.set_option("display.width", 220, "display.max_rows", 400, "display.max_columns", 30)
    print("\nfamily words present in each run's sample prompt:", json.dumps(ctx_words))
    t = rates[(rates.split == "ALL")]
    piv = t.pivot_table(index=["run", "cls"], columns="family", values="novel_rate").mul(100).round(3)
    print("\nnovel hit rate (% of reads) by run x class\n", piv)
    print(
        "\nreads per run x class\n",
        t[t.family == "any"].set_index(["run", "cls"])[["reads", "items", "novel", "items_with_novel", "items_frac"]],
    )
    lay = rates[(rates.cls == "ALL") | (rates.split.str.isdigit())]
    lay = rates[rates.split.str.isdigit() & (rates.family == "any")].pivot_table(
        index="run", columns="split", values="novel_rate", aggfunc="mean"
    )
    print("\nnovel any-family rate by layer (mean over classes)\n", lay.mul(100).round(3))
    print(
        "\ngate split\n",
        rates[rates.split.str.startswith("gate")]
        .pivot_table(index=["run", "split"], columns="family", values="novel_rate")
        .mul(100)
        .round(3),
    )
    print("\nA vs B (pp)\n", comps.assign(**{c: comps[c] * 100 for c in ("rate_a", "rate_b", "diff", "lo", "hi")}).round(3).to_string())
    print("\ntop patterns\n", subs[subs.hit > 0].sort_values("novel", ascending=False).head(40).to_string())


if __name__ == "__main__":
    main()
