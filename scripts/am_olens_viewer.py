# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "pyarrow", "tokenizers", "huggingface_hub"]
# ///
"""expview page for the GLM-5.3 agentic misalignment episodes with Celeste's O-lens reads at their transcript positions.

    uv run --script scripts/am_olens_viewer.py [--out ~/workspace/tmp/am_olens_viewer]

One item per episode (450: blackmail / leaking / murder x 5 conditions x 30 epochs) with
  - labels: harmful (inspect_evals scenario classifier, Sonnet 5.5) and eval_aware (verbalised eval-awareness monitor,
    glm-evalgaming monitor.py: Sonnet 5.5 -> Sonnet 4.5 -> GPT-5.6 terra fallback), eval_aware_strict if the strict re-run is on disk
  - calls: the GLM subject call, the classifier call and every awareness-monitor attempt, each with its exact input and output
  - token reads: the transcript as GLM-5.3 tokens (Celeste's render) with, at every read position, the O-lens RL greedy bullet
    inline and RL samples, SFT greedy + samples, J-lens L60/L51 top-10 and the two Qwen3.6-27B readers on click

Sources (local copies on bramble by default, else downloaded from HF):
  ceselder/glm-5.3-olens-counterfactual-investigator  agentic_misalignment/{episodes,readouts,jlens,qwen_readers}.parquet
  Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens
    runs/agentic_misalignment/2026-10-24_glm53_inspect_evals  rows.jsonl + expview/ (subject + classifier calls, env snapshots)
    runs/eval_gaming/2026-10-24_glm53_aisi_honeypots/monitor/am_main.jsonl  awareness monitor I/O
Join: episodes (run_dir, row_index) -> line row_index of rows.jsonl = line row_index of expview/items.jsonl; monitor key =
<eval log>:<sample id>:<epoch>.
"""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from tokenizers import Tokenizer

HOME = Path.home()
CELESTE_REPO = "ceselder/glm-5.3-olens-counterfactual-investigator"
SPRINT_REPO = "Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens"
RUN = "runs/agentic_misalignment/2026-10-24_glm53_inspect_evals"
MONITOR = "runs/eval_gaming/2026-10-24_glm53_aisi_honeypots/monitor/am_main.jsonl"
LOCAL_CELESTE = HOME / "workspace/tmp/ceselder_inventory/ds"
LOCAL_SPRINT = HOME / "workspace/counterfactual-investigator-agent-with-olens/data/hf_stage_glm53"
LOCAL_STRICT = HOME / "workspace/glm-evalgaming/monitor/am_main_strict.jsonl"
EXPVIEW = HOME / "workspace/expview"
SPECIAL = {"<|system|>": "system", "<|user|>": "user", "<|assistant|>": "assistant", "<|observation|>": "tool", "<think>": "reasoning"}


def local_or_hf(repo, local_root, rel, patterns):
    p = Path(local_root) / rel
    if p.exists():
        return p
    return Path(snapshot_download(repo, repo_type="dataset", allow_patterns=patterns)) / rel


def jsonl(p):
    return [json.loads(line) for line in open(p) if line.strip()]


def tokens_of(tok, text):
    """GLM-5.3 tokens of the render as exact text slices (byte-split characters go to the first token that covers them)."""
    enc = tok.encode(text, add_special_tokens=False)
    out, end = [], 0
    for a, b in enc.offsets:
        out.append(text[max(a, end) : b] if b > end else "")
        end = max(end, b)
    assert "".join(out) == text[:end] and end == len(text), "token offsets do not cover the render"
    return out, enc.ids


def sections_of(toks):
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


def bullets(s):
    return [[x] for x in str(s or "").split("\n") if x.strip()]


def topk(lst):
    return [[d["token"], round(float(d["prob"]), 4)] for d in lst]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HOME / "workspace/tmp/am_olens_viewer"))
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    sys.path.insert(0, str(EXPVIEW))
    from expview import Run, call

    cdir = local_or_hf(CELESTE_REPO, LOCAL_CELESTE, "agentic_misalignment", ["agentic_misalignment/*", "README.md"])
    rdir = local_or_hf(SPRINT_REPO, LOCAL_SPRINT, RUN, [f"{RUN}/rows.jsonl", f"{RUN}/meta.json", f"{RUN}/expview/*"])
    mon_path = local_or_hf(SPRINT_REPO, LOCAL_SPRINT, MONITOR, [MONITOR])
    tok = Tokenizer.from_file(hf_hub_download("zai-org/GLM-5.3", "tokenizer.json"))

    eps = pd.read_parquet(cdir / "episodes.parquet")
    rd = pd.read_parquet(cdir / "readouts.parquet")
    jl = pd.read_parquet(cdir / "jlens.parquet").set_index(["tid", "pos"])
    qw = pd.read_parquet(cdir / "qwen_readers.parquet").set_index(["tid", "pos", "reader"])
    rows = jsonl(rdir / "rows.jsonl")
    xitems = jsonl(rdir / "expview/items.jsonl")
    envs = {p.stem: json.loads(p.read_text())["files"] for p in (rdir / "expview/envs").glob("*.json")}
    mon = {r["key"]: r for r in jsonl(mon_path)}
    strict = {r["key"]: r for r in jsonl(LOCAL_STRICT)} if LOCAL_STRICT.exists() else {}
    assert len(rows) == len(xitems) == 450 and all(x["id"] == f"{r['arm']}/e{r['sample']:02d}" for r, x in zip(rows, xitems)), (
        "rows.jsonl and expview items out of order"
    )

    run = Run(
        out / "run",
        title="GLM 5.3 agentic misalignment: O-lens reads",
        description=(
            "450 episodes of GLM 5.3 in inspect_evals agentic misalignment (blackmail / leaking / murder, 5 prompt conditions, "
            "30 samples each).\n\n"
            "- **harmful**: did GLM blackmail / leak / cancel the rescue? Judged by Sonnet 5.5 with the eval's own prompt.\n"
            "- **eval_aware**: does GLM say in its reasoning or reply that this might be a test, eval or constructed scenario? "
            "Judged by Sonnet 5.5 (Sonnet 4.5 when 5.5 refused).\n\n"
            "Filter or sort the table below, then click an episode. In the episode: **transcript with reads** shows the "
            "O-lens read (RL lens, greedy) in a blue box after each of the 32 read tokens; click a box for more samples, the SFT "
            "lens, J-lens and Qwen readers. **harm judge** and **eval-awareness judge** show exactly what each judge saw and said."
        ),
        label_order=["harmful", "eval_aware", "scenario", "condition"],
    )
    run.reset()

    rd = rd.sort_values(["tid", "pos"])
    by_tid = dict(tuple(rd.groupby("tid")))
    stats = Counter()
    problems = []
    legend = (
        "Blue boxes: what the O-lens (RL lens, greedy) reads from GLM's activation at that token. Click a box or an underlined "
        "token for 4 more RL samples, the SFT lens, J-lens top-10 and two Qwen3.6-27B readers. The lens was trained on web "
        "text, so reads often guess the next words rather than state intent."
    )
    eps = eps.assign(_i=eps.row_index.astype(int)).sort_values("_i")
    for e in eps.itertuples():
        i = int(e.row_index)
        assert e.run_dir == RUN, e.run_dir
        r, x = rows[i], xitems[i]
        key = f"{x['extra']['eval_log']}:{r['item']}:{r['sample']}"
        m = mon[key]
        toks, _ids = tokens_of(tok, e.rendered_text)
        if len(toks) != int(e.n_tokens_full):
            problems.append(f"{x['id']}: {len(toks)} tokens vs n_tokens_full {e.n_tokens_full}")
        reads = {}
        rr = by_tid[e.tid]
        for p in rr.itertuples():
            pos = int(p.pos)
            if pos >= len(toks) or toks[pos].strip() != str(p.token).strip():
                problems.append(f"{x['id']}: pos {pos} token {p.token!r} vs render {toks[pos] if pos < len(toks) else None!r}")
                stats["unmapped"] += 1
                continue
            lab = f"{p.kind} · {p.role}"
            rs = [
                {"tool": "O-lens RL", "label": lab, "inline": True, "lines": [[p.rl_greedy]]},
                {"tool": "O-lens RL samples", "lines": [[getattr(p, f"rl_s{k}")] for k in range(1, 5)]},
                {
                    "tool": "O-lens SFT",
                    "label": "greedy, then 4 samples",
                    "lines": [[p.sft_greedy]] + [[getattr(p, f"sft_s{k}")] for k in range(1, 5)],
                },
            ]
            if (e.tid, p.pos) in jl.index:
                j = jl.loc[(e.tid, p.pos)]
                rs.append(
                    {
                        "tool": "J-lens L60",
                        "label": f"cos(h51, olens input) {float(j.h51_cos_olens_input):.2f}",
                        "lines": topk(j.jlens_L60_top10),
                    }
                )
                rs.append({"tool": "J-lens L51", "label": "olens input", "lines": topk(j.jlens_L51_top10)})
                stats["jlens"] += 1
            for reader in ("distilled", "from_scratch"):
                if (e.tid, p.pos, reader) in qw.index:
                    q = qw.loc[(e.tid, p.pos, reader)]
                    lines = bullets(q.greedy) + [[f"[s{k}] " + b[0]] for k in range(1, 5) for b in bullets(getattr(q, f"s{k}"))]
                    rs.append({"tool": f"Qwen {reader}", "label": f"{q.ckpt}: greedy, then s1-s4", "lines": lines})
                    stats["qwen_" + reader] += 1
            reads[str(pos)] = rs
            stats["reads"] += 1
            stats["kind:" + p.kind] += 1
        stats["episodes"] += 1

        calls = [{**c, "name": "harm judge" if c["name"] == "classifier" else c["name"]} for c in x["calls"]]
        for k, att in enumerate(m.get("fallback_from") or []):
            calls.append(
                call(
                    f"eval-awareness judge: {att['model'].split('/')[-1]} gave no verdict",
                    att["model"],
                    messages=m["messages"],
                    output={"role": "assistant", "content": att.get("output") or ""},
                    params={"temperature": 0, "max_tokens": 1000},
                    finish_reason=att.get("finish_reason"),
                    error=json.dumps(att["error"]) if att.get("error") else None,
                )
            )
        calls.append(
            call(
                "eval-awareness judge",
                m["monitor_model"],
                messages=m["messages"],
                output={"role": "assistant", "content": m["output"]},
                params={"temperature": 0, "max_tokens": 1000},
                parsed={"aware": m["aware"]},
                finish_reason=m.get("finish_reason"),
            )
        )
        labels = {
            "harmful": bool(r["harmful"]),
            "eval_aware": m["aware"],
            "scenario": r["scenario"],
            "condition": r["condition"],
        }
        strict_aware = None
        if key in strict:
            s = strict[key]
            strict_aware = s["aware"]
            calls.append(
                call(
                    "eval-awareness judge, strict prompt (not used for the label)",
                    s["monitor_model"],
                    messages=s["messages"],
                    output={"role": "assistant", "content": s["output"]},
                    params={"temperature": 0, "max_tokens": 1000},
                    parsed={"aware": s["aware"]},
                    finish_reason=s.get("finish_reason"),
                    fallback_from=json.dumps(s["fallback_from"]) if s.get("fallback_from") else None,
                )
            )
        celeste_aware = str(rr.aware.iloc[0]).lower() == "true"
        if celeste_aware != m["aware"]:
            problems.append(f"{x['id']}: Celeste aware {celeste_aware} vs monitor {m['aware']}")
        if (str(rr.positive.iloc[0]).lower() == "true") != bool(r["harmful"]):
            problems.append(f"{x['id']}: Celeste positive vs rows harmful differ")
        stats["harmful"] += labels["harmful"]
        stats["aware"] += bool(m["aware"])
        stats[f"xtab:{r['scenario']}:harmful={labels['harmful']}:aware={m['aware']}"] += 1
        run.add(
            id=x["id"],
            group=x["group"],
            labels=labels,
            env_files=envs[x["env"]],
            calls=calls,
            extra={
                **x.get("extra", {}),
                "harm_classifier_verdict_before_hard_check": bool(r["classifier_verdict"]),
                "eval_aware_judge_model": m["monitor_model"],
                "eval_aware_strict_prompt": strict_aware,
                "olens_tid": e.tid,
                "episode_id": e.episode_id,
                "row_index": i,
                "monitor_key": key,
                "n_tokens": len(toks),
                "n_read_positions": len(reads),
            },
            tokens={"tokens": toks, "sections": sections_of(toks), "reads": reads, "legend": legend},
        )

    page = out / "index.html"
    subprocess.run([sys.executable, str(EXPVIEW / "expview.py"), "build", str(out / "run"), "-o", str(page)], check=True)
    print(json.dumps(dict(sorted(stats.items())), indent=1))
    print(f"{len(problems)} problems", *problems[:40], sep="\n")
    print(page, f"{page.stat().st_size / 1e6:.2f} MB", f"tokens/ {sum(f.stat().st_size for f in (out / 'tokens').iterdir()) / 1e6:.1f} MB")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    main()
