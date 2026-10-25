# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2", "pyarrow>=17", "tokenizers>=0.20", "huggingface_hub>=0.25"]
# ///
"""Import Celeste's GLM-5.3 O-lens readout datasets into expview runs, one run per config.

    uv run --script expview/importers/celeste.py --ds data/olens_search/ds --ds data/olens_search/ds_new --ds data/olens_search/ds_wild \
        [--runs-root data/hf_stage_glm53 --runs-root data] [--docent-root data] [--skip coding_envs/readouts_dense.parquet]

Sources: ceselder/glm-5.3-olens-counterfactual-investigator, ceselder/glm-5.3-olens-new-sources and the wild-rollouts
dataset (`hf download ... --repo-type dataset --local-dir DIR`; expview/run.sh --pull does it). Output:
data/expview/imports/celeste/<source>/<config>/ (dataset id imports/celeste/<source>/<config>).

One item per transcript view (tid). Its transcript is Celeste's render re-tokenised with GLM-5.3's tokenizer, split into
sections at the chat special tokens; at every read position:
  O-lens RL (greedy, inline) · O-lens RL samples (s1-s4) · O-lens SFT (greedy + s1-s4) · J-lens L60 / L51 top-10 ·
  Qwen distilled / from_scratch (greedy + s1-s4)
A read whose token doesn't match the re-tokenised render at its position is dropped and counted.

When the transcript's source run is on disk (--runs-root: dirs holding runs/<exp>/<run>/), the item also gets that run's
expview calls (the subject transcript and its judges, exact I/O) and environment, from the same line of its items.jsonl.
Docent links come from the runs' docent.json (--docent-root, with docent_aliases.json mapping HF run names to local ones).
"""

from __future__ import annotations

import argparse
import bisect
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from expview import REPO, Run, read

ALIASES = json.loads((HERE / "docent_aliases.json").read_text())
SAMPLES = ["s1", "s2", "s3", "s4"]
SPECIAL = {"<|system|>": "system", "<|user|>": "user", "<|assistant|>": "assistant", "<|observation|>": "tool", "<think>": "reasoning"}
RL, RL_S, SFT = "GLM verbalizer (RL)", "GLM verbalizer (RL) samples", "GLM verbalizer (SFT, unreliable)"
QWEN = {"distilled": "Qwen verbalizer on GLM (distilled)", "from_scratch": "Qwen verbalizer on GLM (from scratch)"}
BON = {
    "s100": "GLM verbalizer RL step 100, best 4 of 16",
    "latest": "GLM verbalizer RL step 225, best 4 of 16",
    "qwen": "Qwen verbalizer from scratch, best 4 of 16",
}
CONTROL, JUDGE = "model's own continuation (control)", "judge (why this position)"
FVE_NOTE = (
    " Line scores, where present, are per-bullet FVE: GLM-5.3's activation reconstructor's prediction from the bullet vs this activation."
)
READERS = {
    RL: {
        "about": "GLM-5.3 reading its own block-51 activation (one-bullet O-lens after RL, `rl1b_2/step_0100`), greedy bullet, "
        "shown inline. Celeste's most trusted reader. Trained on web text, so bullets are often continuation guesses, not "
        "statements of intent." + FVE_NOTE,
        "color": "#2f5d8a",
    },
    RL_S: {"about": "Four more samples at T=1 from the same RL GLM verbalizer: the spread." + FVE_NOTE, "color": "#5a6f8f"},
    SFT: {
        "about": "The GLM verbalizer before RL (SFT warm start, `sft1b_ext/step_0292`): greedy, then four samples. Celeste calls its "
        'readouts "pretty sus"; off by default.' + FVE_NOTE,
        "color": "#3f7f5a",
        "hidden": True,
    },
    "J-lens L60": {"about": "J-lens top-10 next tokens at layer 60.", "color": "#b5662f"},
    "J-lens L51": {"about": "J-lens top-10 at layer 51 (the O-lens input layer).", "color": "#8f7a2f"},
    QWEN["distilled"]: {
        "about": "Qwen3.6-27B verbalising GLM-5.3's activation (affine map into Qwen's layer 42), 4 bullets; SFT on targets distilled "
        "from the GLM verbalizer's own samples, then RL. Not independent of the GLM lens. Celeste: Qwen readers are about 80% as "
        "trustworthy as the GLM one.",
        "color": "#8a4f9e",
    },
    QWEN["from_scratch"]: {
        "about": "Qwen3.6-27B verbalising GLM-5.3's activation, 1 bullet; trained on GLM's verbatim continuations then RL, never saw "
        "GLM-lens output: an independent second opinion. About 80% as trustworthy as the GLM verbalizer (Celeste).",
        "color": "#a4473a",
    },
    BON["s100"]: {
        "about": "Test-time scaffold at judge-flagged positions: 16 samples from the RL GLM verbalizer (step 100), every prefix scored by "
        "GLM-5.3's activation reconstructor, the best 4-prefix combination kept. Line score = that prefix's own cFVE; read score = "
        "the 4-set's cFVE. Label = the judge's category and why the position was picked.",
        "color": "#1f4e79",
    },
    BON["latest"]: {"about": "Same best-of-16 scaffold with the newest RL checkpoint (step 225).", "color": "#33658a"},
    BON["qwen"]: {"about": "Same best-of-16 scaffold with the from-scratch Qwen3.6-27B reader on GLM's activation.", "color": "#86364a"},
    CONTROL: {
        "about": "Control for best-of-N: the model's own next 16 tokens, scored by the same reconstructor (first line: best prefix, "
        "second: all 16). If the verbalizer's cFVE doesn't beat this, it may only be predicting the continuation.",
        "color": "#777777",
    },
    JUDGE: {
        "about": "Sonnet 5 judge pass that flagged this position for best-of-N: its rationale and the bullets it flagged.",
        "color": "#9a7b00",
    },
}
LEGEND = (
    "Underlined tokens have reads; the inline box is what the **GLM verbalizer (RL)** reads from GLM-5.3's activation there. "
    "Click a token for its samples, J-lens top-10, the two Qwen readers and (at judge-flagged positions) best-of-16 with its control. "
    "The lens was trained on web text, so reads often guess the next words rather than state intent."
)


def tokens_of(tok: Tokenizer, text: str) -> list[str]:
    """GLM-5.3 tokens of the render as exact text slices (byte-split characters go to the first token covering them)."""
    enc = tok.encode(text, add_special_tokens=False)
    out, end = [], 0
    for a, b in enc.offsets:
        out.append(text[max(a, end) : b] if b > end else "")
        end = max(end, b)
    if end < len(text):
        out.append(text[end:])
    return out


def same_token(ours: str, theirs: str) -> bool:
    """Celeste's decoded token vs ours. A multi-byte character split over two tokens decodes to U+FFFD on her side; ours puts
    the whole character on the first token and "" on the second."""
    a, b = ours.strip(), theirs.strip()
    if a == b:
        return True
    return "\ufffd" in b and (a == "" or b.replace("\ufffd", "").strip() in a)


def realign(text: str, toks: list[str], ends: list[int], pos: int, token: str, ctx: str | None) -> int | None:
    """Celeste's position when it matches our re-tokenisation, else the token ending where her left context (which ends with the
    read token) ends, taking the occurrence nearest the expected offset. Her token ids drift from a re-encode of the render by a few
    tokens mid-transcript (e.g. "wait" "," vs "wait,"), so positions alone drop ~1-12% of reads in the newer configs."""
    if pos < len(toks) and same_token(toks[pos], token):
        return pos
    if not ctx:
        return None
    exp, best, i = ends[min(pos, len(ends) - 1)], None, text.find(ctx)
    while i != -1:
        if best is None or abs(i + len(ctx) - exp) < abs(best - exp):
            best = i + len(ctx)
        i = text.find(ctx, i + 1)
    j = None if best is None else bisect.bisect_left(ends, best)
    return j if j is not None and j < len(toks) and ends[j] == best and same_token(toks[j], token) else None


def sections_of(toks: list[str]) -> list[list]:
    secs, start, name = [], 0, "header"
    for i, t in enumerate(toks):
        new = SPECIAL.get(t) or ("reply" if t == "</think>" else None)
        if new and i > start:
            secs.append([name, start, i])
            start = i
        if new:
            name = new
    secs.append([name, start, len(toks)])
    return secs


def docent_map(roots: list[Path]) -> dict[tuple[str, int], str]:
    out = {}
    for root in roots:
        for f in root.glob("runs/*/*/docent.json"):
            d = json.loads(f.read_text())
            rd = str(f.parent.relative_to(root))
            for i, run in enumerate(d["runs"]):
                if run:
                    out[(rd, i)] = f"https://docent.transluce.org/dashboard/{d['collection_id']}/agent_run/{run}"
    return out


class Calls:
    """The source run's expview items, by (run_dir, source_file) -> list of lines, loaded lazily."""

    def __init__(self, roots: list[Path]):
        self.roots, self.cache = roots, {}

    def get(self, run_dir: str, source_file: str | None, row: int) -> tuple[dict, Path] | None:
        key = (run_dir, source_file)
        if key not in self.cache:
            self.cache[key] = None
            rel = source_file if source_file and source_file.endswith("items.jsonl") else "expview/items.jsonl"
            for root in self.roots:
                p = root / run_dir / rel
                if p.exists():
                    rows = [json.loads(x) for x in p.open() if x.strip()]
                    if not any(r.get("base") for r in rows):  # overlays on this import (evalcue_rows --base) would loop back
                        self.cache[key] = (rows, p.parent)
                    break
        hit = self.cache[key]
        if hit is None or not (0 <= row < len(hit[0])):
            return None
        return hit[0][row], hit[1]


def bullets(s) -> list[str]:
    return [x for x in str(s or "").split("\n") if x.strip()]


def topk(lst) -> list[list]:
    return [[d["token"], round(float(d["prob"]), 4)] for d in lst]


def units(ds: Path) -> list[tuple[str, Path, list[str]]]:
    """(config, dir, readout files) for one downloaded dataset: one subdir per config, or the wild-rollouts single-table layout."""
    if (ds / "olens_wild_rollouts.parquet").exists():
        for sub in ("tb21", "tb_cc"):  # their transcripts aren't published, so their reads can't be placed
            if (ds / sub).is_dir() and not (ds / sub / "transcripts.parquet").exists():
                print(f"{ds.name}/{sub}: skipped, no transcripts.parquet for its reads", flush=True)
        return [("wild_rollouts", ds, ["olens_wild_rollouts.parquet", "wild/readouts_dense.parquet"])]
    return [(p.name, p, ["readouts.parquet", "readouts_dense.parquet"]) for p in sorted(ds.iterdir()) if (p / "readouts.parquet").exists()]


def is_true(v) -> bool:
    return v is not None and not pd.isna(v) and bool(v)


def fl(v) -> float | None:
    return None if v is None or pd.isna(v) else round(float(v), 4)


def lines_fve(p, cols: list[str]) -> list[list]:
    """[bullet, fve] for each readout column with text (fve only where Celeste scored it)."""
    out = []
    for c in cols:
        t = getattr(p, c, None)
        if t is None or (isinstance(t, float) and pd.isna(t)) or not str(t).strip():
            continue
        f = fl(getattr(p, f"fve_{c}", None))
        out.append([t, f] if f is not None else [t])
    return out


def lens_read(name: str, p, cols: list[str], **kw) -> dict:
    """A GLM-lens read whose lines carry per-bullet FVE; when any is scored, the read's score is the best of them, so the
    viewer's min-FVE box can hide weak reads (unscored reads always show)."""
    lines = lines_fve(p, cols)
    fves = [x[1] for x in lines if len(x) > 1]
    if fves:
        kw.update(score=max(fves), score_name="FVE", filter=True)
    return read(name, lines, **kw)


def dense_label(p) -> str | None:
    """pos_kind / punctuation and, for the donation bet, the running number the model has written so far."""
    kind, punct = getattr(p, "pos_kind", None), getattr(p, "is_punct", None)
    parts = [kind if kind != getattr(p, "role", None) else None]
    if punct is not None and not pd.isna(punct):
        parts.append("punct" if punct else "non-punct")
    n = fl(getattr(p, "running_number", None))
    if n is not None:
        side, changed = getattr(p, "running_number_side", None), getattr(p, "number_changed", None)
        num = f"{n:,.0f}" if n == int(n) else f"{n:,}"
        parts.append(f"running number {num} ({side} threshold){' · number changed' if is_true(changed) else ''}")
    return " · ".join(str(x) for x in parts if x) or None


def bestofn_reads(b) -> list[dict]:
    why = " · ".join([str(b.judge_category or "")] + list(b.why if b.why is not None else []))
    rs = []
    for k, name in BON.items():
        texts, single = getattr(b, f"{k}_opt4_text", None), getattr(b, f"{k}_opt4_single_cfve", None)
        if texts is None:
            continue
        lines = [[t, fl(s)] for t, s in zip(texts, single if single is not None else [None] * len(texts))]
        rs.append(read(name, lines, label=why.strip(" ·") or None, score=fl(getattr(b, f"{k}_opt4_cfve", None)), score_name="cfve"))
    if b.control_text is not None:
        rs.append(
            read(
                CONTROL,
                [[b.control_best_text, fl(b.control_best_cfve)], [b.control_text, fl(b.control_full_cfve)]],
                score=fl(b.control_full_cfve),
                score_name="cfve",
            )
        )
    head = f"schemey {b.judge_schemey} · strategic {b.judge_strategic} · {b.judge_category}"
    head += " · latent eval awareness" if b.judge_latent else ""
    flagged = list(b.judge_flagged_bullets) if b.judge_flagged_bullets is not None else []
    rs.append(read(JUDGE, [b.judge_rationale or ""] + [f"flagged: {x}" for x in flagged], label=head))
    return rs


DROP = {"meta_json", "labels_json", "ctx_right40", "next16", "next16_ids", "extra", "h51_norm", "lens_rl_ckpt", "lens_sft_ckpt", "model"}


def read_cols(f: Path, filters: list | None = None) -> pd.DataFrame:
    """A readout table without the wide provenance columns the viewer doesn't use (the dense tables are millions of rows)."""
    return pd.read_parquet(f, columns=[c for c in pq.ParquetFile(f).schema_arrow.names if c not in DROP], filters=filters)


def truthy(v) -> bool | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v).lower() in ("true", "1")


BATCH_ROWS = 1_500_000  # dense readout rows held in memory at once


def normalise(rd: pd.DataFrame) -> pd.DataFrame:
    rd = rd.rename(columns={"left_ctx": "ctx_left40", "file": "source_file"})
    if "family" in rd:  # wild-rollouts layout
        rd["env"], rd["arm"], rd["positive"] = rd.family, rd.task_id, rd.passed
    for c in ("env", "arm", "kind", "role", "run_dir", "source_file", "row_index", "aware", "positive", "episode_id"):
        if c not in rd:
            rd[c] = None
    rd["_dup"] = rd["_dup"].fillna(False).astype(bool) if "_dup" in rd else False
    return rd.drop_duplicates(["tid", "pos"]).sort_values(["tid", "pos"])


def prefix_dups(cfg_dir: Path, dense: list[str], base: pd.DataFrame) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Positions Celeste read once for a shared token prefix (dense_prefix_dups.parquet) and the canonical rows they point to,
    so every transcript shows reads at its shared prompt, not only the first one read."""
    p = cfg_dir / "dense_prefix_dups.parquet"
    if not p.exists():
        return None, None
    dups = pd.read_parquet(p)
    want = dups[["canon_tid", "canon_pos"]].drop_duplicates()
    rows = [base]
    for f in dense:
        tids = sorted(set(want.canon_tid))
        rows.append(read_cols(cfg_dir / f, [("tid", "in", tids)]))
    canon = pd.concat(rows, ignore_index=True).drop_duplicates(["tid", "pos"])
    canon = canon.merge(want, left_on=["tid", "pos"], right_on=["canon_tid", "canon_pos"]).drop(columns=["tid", "pos"])
    return dups[["tid", "pos", "canon_tid", "canon_pos"]], canon


def fill_dups(dups: pd.DataFrame, canon: pd.DataFrame) -> pd.DataFrame:
    out = dups.merge(canon, on=["canon_tid", "canon_pos"]).assign(_dup=True, _file="dense_prefix_dups.parquet")
    out["kind"] = "shared prefix"
    return out.drop(columns=["canon_tid", "canon_pos"])


def import_config(
    source: str, cfg: str, cfg_dir: Path, files: list[str], out: Path, tok: Tokenizer, docent: dict, calls: Calls, owner: str = "Celeste's"
) -> Counter:
    stats = Counter()
    tx = cfg_dir / "transcripts.parquet"
    eps = pd.read_parquet(tx if tx.exists() else cfg_dir / "episodes.parquet").rename(columns={"text": "rendered_text"})
    dense = [f for f in files if "readouts_dense" in f]
    base = pd.concat([read_cols(cfg_dir / f).assign(_file=f) for f in files if f not in dense], ignore_index=True)
    for fp in (cfg_dir / "fve.parquet", cfg_dir / "fve_readouts.parquet"):  # per-bullet FVE for the capped reads, published separately
        if fp.exists():
            fve = pd.read_parquet(fp).drop(columns=["id"], errors="ignore").drop_duplicates(["tid", "pos"])
            base = base.drop(columns=[c for c in fve.columns if c.startswith("fve_") and c in base])
            base = base.merge(fve, on=["tid", "pos"], how="left")
    dense_tids = {f: set(pq.read_table(cfg_dir / f, columns=["tid"]).column("tid").unique().to_pylist()) for f in dense}
    all_tids = set(base.tid).union(*dense_tids.values())
    dups, canon = prefix_dups(cfg_dir, dense, base) if dense else (None, None)
    if dups is not None:  # a transcript whose reads are all shared-prefix copies still gets an item
        all_tids |= set(dups.tid)
    missing = all_tids - set(eps.tid)
    if missing:
        print(f"{source}/{cfg}: no transcripts for {len(missing)} tids, their reads are skipped", flush=True)
    jl = pd.read_parquet(cfg_dir / "jlens.parquet").set_index(["tid", "pos"]) if (cfg_dir / "jlens.parquet").exists() else None
    qp = cfg_dir / "qwen_readers.parquet"
    qw = pd.read_parquet(qp).set_index(["tid", "pos", "reader"]) if qp.exists() else None
    bp = cfg_dir / "bestofn.parquet"
    bon: dict[str, dict[int, object]] = {}
    for b in pd.read_parquet(bp).itertuples() if bp.exists() else []:
        bon.setdefault(b.tid, {})[int(b.pos)] = b
    run = Run(
        out,
        title=f"GLM 5.3 O-lens reads: {cfg} ({source})" if owner == "Celeste's" else f"GLM 5.3 O-lens reads ({owner}): {cfg}",
        description=(
            f"{owner} O-lens readouts of our GLM-5.3 transcripts, config `{cfg}` of `{source}`. One item per transcript view; "
            "its transcript shows every read inline. **positive** is the env's label for the episode, **aware** the eval-awareness "
            "judge's (where Celeste's data has them). Items whose source run is on disk also carry its model calls (the subject "
            "transcript and every judge, exact I/O) and environment."
        ),
        label_order=["positive", "aware"],
        readers=READERS,
        source=f"{source}/{cfg}: {', '.join(files)}",
    )
    run.reset()
    n_dense = sum(pq.ParquetFile(cfg_dir / f).metadata.num_rows for f in dense)
    todo = [t for t in eps.tid if t in all_tids]
    size = len(todo) if n_dense < BATCH_ROWS else max(1, len(todo) * BATCH_ROWS // n_dense)
    for k in range(0, len(todo), size):
        batch = set(todo[k : k + size])
        parts = [base[base.tid.isin(batch)]]
        for f in dense:
            if batch & dense_tids[f]:
                parts.append(read_cols(cfg_dir / f, [("tid", "in", sorted(batch & dense_tids[f]))]).assign(_file=f))
        if dups is not None:
            parts.append(fill_dups(dups[dups.tid.isin(batch)], canon))
        by_tid = dict(tuple(normalise(pd.concat(parts, ignore_index=True)).groupby("tid")))
        if size < len(todo):
            n = sum(len(v) for v in by_tid.values())
            print(f"{source}/{cfg}: batch {k // size + 1}/{-(-len(todo) // size)}, {n:,} rows", flush=True)
        import_items(eps[eps.tid.isin(batch)], by_tid, run, stats, tok, docent, calls, jl, qw, bon, cfg)
    stats["bestofn_unplaced"] += sum(len(v) for v in bon.values())
    return stats


def import_items(eps, by_tid, run, stats, tok, docent, calls, jl, qw, bon, cfg) -> None:
    for e in eps.itertuples():
        rr = by_tid.get(e.tid)
        if rr is None:
            continue
        toks = tokens_of(tok, e.rendered_text)
        ends = list(itertools.accumulate(len(t) for t in toks))
        reads = {}
        for p in rr.itertuples():
            pos = realign(e.rendered_text, toks, ends, int(p.pos), str(p.token), getattr(p, "ctx_left40", None))
            if pos is None:
                stats["unmatched"] += 1
                continue
            if pos != int(p.pos):
                stats["realigned"] += 1
            g = lambda c, p=p: getattr(p, c, None)
            lab = " · ".join(str(x) for x in (g("kind"), g("role"), dense_label(p)) if x)
            rs = []
            if g("rl_greedy") is not None:
                # dense rows (every token in the donation bet) only peek inline where the stated number changes, else the text drowns
                peek = not is_true(g("dense")) or is_true(g("number_changed"))
                rs.append(lens_read(RL, p, ["rl_greedy"], label=lab, inline=peek or None))
                rs.append(lens_read(RL_S, p, [f"rl_{s}" for s in SAMPLES]))
            if g("sft_greedy") is not None:
                rs.append(lens_read(SFT, p, ["sft_greedy"] + [f"sft_{s}" for s in SAMPLES], label="greedy, then samples"))
            b = bon.get(e.tid, {}).pop(int(p.pos), None)
            if b is not None:
                rs += bestofn_reads(b)
                stats["bestofn"] += 1
            if jl is not None and (e.tid, p.pos) in jl.index:
                j = jl.loc[(e.tid, p.pos)]
                for col, name in (("jlens_L60_top10", "J-lens L60"), ("jlens_L51_top10", "J-lens L51")):
                    if col in j and j[col] is not None:
                        rs.append(read(name, topk(j[col])))
            for reader in ("distilled", "from_scratch"):
                if qw is not None and (e.tid, p.pos, reader) in qw.index:
                    q = qw.loc[(e.tid, p.pos, reader)]
                    lines = bullets(q.greedy) + [f"[{s}] {b}" for s in SAMPLES for b in bullets(getattr(q, s, ""))]
                    rs.append(read(QWEN[reader], lines, label=str(getattr(q, "ckpt", "")) or None))
            reads[str(pos)] = [r for r in rs if r["lines"]]
            stats["reads"] += 1
        for q, b in bon.pop(e.tid, {}).items():  # best-of-N at positions without a readout row here (e.g. dense, not imported)
            if q < len(toks) and same_token(toks[q], str(b.token)):
                reads.setdefault(str(q), []).extend(r for r in bestofn_reads(b) if r["lines"])
                stats["bestofn"] += 1
            else:
                stats["bestofn_unplaced"] += 1
        if not reads:
            stats["no_reads"] += 1
            continue
        own = rr[~rr._dup]  # a prefix-dup row carries its canonical episode's fields
        f0 = (own if len(own) else rr).iloc[0].copy()
        for c in ("run_dir", "source_file", "row_index", "env", "arm"):  # the transcript's own provenance wins over a shared row's
            v = getattr(e, c, None)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                f0[c] = v
        if not len(own):  # every read is a copy from another transcript: its labels aren't this transcript's
            f0["positive"] = f0["aware"] = None
        row = None if pd.isna(f0.row_index) else int(f0.row_index)
        links, item_calls, env, extra = {}, [], None, {"tid": e.tid, "run_dir": f0.run_dir, "source_file": f0.source_file, "row_index": row}
        if isinstance(f0.run_dir, str) and row is not None:
            local = ALIASES.get(f"{f0.run_dir}::{f0.source_file}", ALIASES.get(f0.run_dir, f0.run_dir))
            if (local, row) in docent:
                links["Docent"] = docent[(local, row)]
            src = calls.get(f0.run_dir, f0.source_file if isinstance(f0.source_file, str) else None, row)
            if src:
                x, src_dir = src
                item_calls = x.get("calls") or []
                if isinstance(x.get("env"), str) and (src_dir / "envs" / f"{x['env']}.json").exists():
                    env = {"files": json.loads((src_dir / "envs" / f"{x['env']}.json").read_text())["files"]}
                extra["source_item"] = x.get("id")
                stats["with_calls"] += 1
        labels = {"positive": truthy(f0.positive), "aware": truthy(f0.aware)}
        run.add(
            id=str(e.tid),
            group=" · ".join(str(x) for x in (f0.env, f0.arm) if x is not None and not (isinstance(x, float) and pd.isna(x))) or cfg,
            labels={k: v for k, v in labels.items() if v is not None},
            calls=item_calls,
            env_files=env["files"] if env else None,
            links=links or None,
            extra=extra,
            tokens={"tokens": toks, "sections": sections_of(toks), "reads": reads, "legend": LEGEND}
            | ({"min_score": 0.0} if any(r.get("filter") for rs in reads.values() for r in rs) else {}),
        )
        stats["items"] += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=Path, action="append", required=True, help="downloaded dataset dir (repeatable)")
    ap.add_argument("--out", type=Path, default=REPO / "data/expview/imports/celeste")
    ap.add_argument("--runs-root", type=Path, action="append", default=[], help="dirs holding runs/<exp>/<run>/ to take calls from")
    ap.add_argument("--docent-root", type=Path, action="append", default=[], help="dirs holding runs/<exp>/<run>/docent.json")
    ap.add_argument("--skip", action="append", default=[], help="skip a file, e.g. coding_envs/readouts_dense.parquet (repeatable)")
    ap.add_argument("--only", action="append", default=[], help="only these configs")
    ap.add_argument("--owner", default="Celeste's", help='whose reads these are, for titles (e.g. "our" for runs/olens_reads)')
    ap.add_argument("--no-dense", action="store_true", help="leave out readouts_dense.parquet (every-punctuation / every-token reads)")
    a = ap.parse_args()
    tok = Tokenizer.from_file(hf_hub_download("zai-org/GLM-5.3", "tokenizer.json"))
    docent, calls = docent_map(a.docent_root), Calls(a.runs_root)
    for ds in a.ds:
        for cfg, cfg_dir, files in units(ds):
            if a.only and cfg not in a.only:
                continue
            files = [f for f in files if (cfg_dir / f).exists() and f"{cfg}/{f}" not in a.skip]
            files = [f for f in files if not (a.no_dense and "readouts_dense" in f)]
            stats = import_config(ds.name, cfg, cfg_dir, files, a.out / ds.name / cfg, tok, docent, calls, a.owner)
            print(f"{ds.name}/{cfg}: {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
