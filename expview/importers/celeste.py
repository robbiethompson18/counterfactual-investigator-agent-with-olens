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
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from expview import REPO, Run, read

ALIASES = json.loads((HERE / "docent_aliases.json").read_text())
SAMPLES = ["s1", "s2", "s3", "s4"]
SPECIAL = {"<|system|>": "system", "<|user|>": "user", "<|assistant|>": "assistant", "<|observation|>": "tool", "<think>": "reasoning"}
READERS = {
    "O-lens RL": {"about": "GLM-5.3 O-lens after RL, greedy bullet (shown inline).", "color": "#2f5d8a"},
    "O-lens RL samples": {"about": "Four more samples (T=1) from the RL O-lens.", "color": "#5a6f8f"},
    "O-lens SFT": {"about": "The SFT O-lens before RL: greedy, then four samples.", "color": "#3f7f5a"},
    "J-lens L60": {"about": "J-lens top-10 next tokens at layer 60.", "color": "#b5662f"},
    "J-lens L51": {"about": "J-lens top-10 at layer 51 (the O-lens input layer).", "color": "#8f7a2f"},
    "Qwen distilled": {"about": "Qwen3.6-27B reader distilled from the GLM lens.", "color": "#8a4f9e"},
    "Qwen from_scratch": {"about": "Qwen3.6-27B reader trained from scratch.", "color": "#a4473a"},
}
LEGEND = (
    "Underlined tokens have reads; the inline box is what the **RL O-lens** (greedy) reads from GLM-5.3's activation there. "
    "Click a token for the RL samples, the SFT lens, J-lens top-10 and the two Qwen3.6-27B readers. The lens was trained on "
    "web text, so reads often guess the next words rather than state intent."
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
                    self.cache[key] = ([json.loads(x) for x in p.open() if x.strip()], p.parent)
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
        return [("wild_rollouts", ds, ["olens_wild_rollouts.parquet", "tb21/readouts_dense.parquet"])]
    return [(p.name, p, ["readouts.parquet", "readouts_dense.parquet"]) for p in sorted(ds.iterdir()) if (p / "readouts.parquet").exists()]


def truthy(v) -> bool | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v).lower() in ("true", "1")


def import_config(source: str, cfg: str, cfg_dir: Path, files: list[str], out: Path, tok: Tokenizer, docent: dict, calls: Calls) -> Counter:
    stats = Counter()
    tx = cfg_dir / "transcripts.parquet"
    eps = pd.read_parquet(tx if tx.exists() else cfg_dir / "episodes.parquet").rename(columns={"text": "rendered_text"})
    rd = pd.concat([pd.read_parquet(cfg_dir / f).assign(_file=f) for f in files], ignore_index=True)
    rd = rd.rename(columns={"left_ctx": "ctx_left40", "file": "source_file"})
    if "family" in rd:  # wild-rollouts layout
        rd["env"], rd["arm"], rd["positive"] = rd.family, rd.task_id, rd.passed
    for c in ("env", "arm", "kind", "role", "run_dir", "source_file", "row_index", "aware", "positive", "episode_id"):
        if c not in rd:
            rd[c] = None
    rd = rd.drop_duplicates(["tid", "pos"]).sort_values(["tid", "pos"])
    jl = pd.read_parquet(cfg_dir / "jlens.parquet").set_index(["tid", "pos"]) if (cfg_dir / "jlens.parquet").exists() else None
    qp = cfg_dir / "qwen_readers.parquet"
    qw = pd.read_parquet(qp).set_index(["tid", "pos", "reader"]) if qp.exists() else None
    by_tid = dict(tuple(rd.groupby("tid")))
    run = Run(
        out,
        title=f"GLM 5.3 O-lens reads: {cfg} ({source})",
        description=(
            f"Celeste's O-lens readouts of our GLM-5.3 transcripts, config `{cfg}` of `{source}`. One item per transcript view; "
            "its transcript shows every read inline. **positive** is the env's label for the episode, **aware** the eval-awareness "
            "judge's (where Celeste's data has them). Items whose source run is on disk also carry its model calls (the subject "
            "transcript and every judge, exact I/O) and environment."
        ),
        label_order=["positive", "aware"],
        readers=READERS,
        source=f"{source}/{cfg}: {', '.join(files)}",
    )
    run.reset()
    for e in eps.itertuples():
        rr = by_tid.get(e.tid)
        if rr is None:
            continue
        toks = tokens_of(tok, e.rendered_text)
        reads = {}
        for p in rr.itertuples():
            pos = int(p.pos)
            if pos >= len(toks) or toks[pos].strip() != str(p.token).strip():
                stats["unmatched"] += 1
                continue
            g = lambda c, p=p: getattr(p, c, None)
            lab = " · ".join(str(x) for x in (g("kind"), g("role")) if x)
            rs = []
            if g("rl_greedy") is not None:
                rs.append(read("O-lens RL", [g("rl_greedy")], label=lab, inline=True))
                rs.append(read("O-lens RL samples", [g(f"rl_{s}") for s in SAMPLES if g(f"rl_{s}") is not None]))
            if g("sft_greedy") is not None:
                sft = [g("sft_greedy")] + [g(f"sft_{s}") for s in SAMPLES if g(f"sft_{s}") is not None]
                rs.append(read("O-lens SFT", sft, label="greedy, then samples"))
            if jl is not None and (e.tid, p.pos) in jl.index:
                j = jl.loc[(e.tid, p.pos)]
                for col, name in (("jlens_L60_top10", "J-lens L60"), ("jlens_L51_top10", "J-lens L51")):
                    if col in j and j[col] is not None:
                        rs.append(read(name, topk(j[col])))
            for reader in ("distilled", "from_scratch"):
                if qw is not None and (e.tid, p.pos, reader) in qw.index:
                    q = qw.loc[(e.tid, p.pos, reader)]
                    lines = bullets(q.greedy) + [f"[{s}] {b}" for s in SAMPLES for b in bullets(getattr(q, s, ""))]
                    rs.append(read(f"Qwen {reader}", lines, label=str(getattr(q, "ckpt", "")) or None))
            reads[str(pos)] = [r for r in rs if r["lines"]]
            stats["reads"] += 1
        if not reads:
            stats["no_reads"] += 1
            continue
        f0 = rr.iloc[0]
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
            tokens={"tokens": toks, "sections": sections_of(toks), "reads": reads, "legend": LEGEND},
        )
        stats["items"] += 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=Path, action="append", required=True, help="downloaded dataset dir (repeatable)")
    ap.add_argument("--out", type=Path, default=REPO / "data/expview/imports/celeste")
    ap.add_argument("--runs-root", type=Path, action="append", default=[], help="dirs holding runs/<exp>/<run>/ to take calls from")
    ap.add_argument("--docent-root", type=Path, action="append", default=[], help="dirs holding runs/<exp>/<run>/docent.json")
    ap.add_argument("--skip", action="append", default=[], help="skip a file, e.g. coding_envs/readouts_dense.parquet (repeatable)")
    ap.add_argument("--only", action="append", default=[], help="only these configs")
    a = ap.parse_args()
    tok = Tokenizer.from_file(hf_hub_download("zai-org/GLM-5.3", "tokenizer.json"))
    docent, calls = docent_map(a.docent_root), Calls(a.runs_root)
    for ds in a.ds:
        for cfg, cfg_dir, files in units(ds):
            if a.only and cfg not in a.only:
                continue
            files = [f for f in files if (cfg_dir / f).exists() and f"{cfg}/{f}" not in a.skip]
            stats = import_config(ds.name, cfg, cfg_dir, files, a.out / ds.name / cfg, tok, docent, calls)
            print(f"{ds.name}/{cfg}: {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
