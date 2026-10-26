"""O-lens reads on Qwen3.6-27B's thinking-off Donation Bet transcripts (run donation_bet_goodbad/2026-10-25_qwen-nothink-cap8k).

    uv run python scripts/dbet_nothink_olens.py newrun                  # -> data/runs/olens_reads/DATE_dbet-nothink
    uv run python scripts/dbet_nothink_olens.py requests OUT_DIR        # -> OUT_DIR/requests.jsonl for scripts/olens_read_requests.py
    uv run python scripts/dbet_nothink_olens.py analyze OUT_DIR         # keyword rates -> OUT_DIR/analysis.json
    uv run --with transformers==5.16.1 --with jinja2 python scripts/dbet_nothink_olens.py expview OUT_DIR
    uv run python scripts/dbet_nothink_olens.py finish OUT_DIR
    uv run --with pandas --with pyarrow python scripts/dbet_nothink_olens.py glm-analyze GLM_DIR
    uv run --with pandas --with pyarrow --with tokenizers python scripts/dbet_nothink_olens.py glm-expview GLM_DIR
GLM_DIR: the GLM O-lens reads of the same design on run 2026-10-25_glm53-nothink-ownline (Celeste's RL lens at block 51, greedy +
4 T=1 samples, one bullet each, no FVE; HF runs/olens_reads/2026-10-25_glm53_donation_nothink_dense/2026-10-25_glm53-nothink-ownline,
made by scripts/glm_olens_dense/ on a 4x B200 box). Positions there: punctuation and template tokens plus the reply's first 6 tokens.

Requests (space lens olens_s3d_space_rl600, greedy, 4 bullets, layer 40; all 5 of 24-56 ran at ~11 transcripts/h per GPU),
PER_ARM random rows of each of the 7 arms:
    ID          every punctuation token and <|im_end|> of the prompt and the reply (olens_read_requests.py's default positions)
    ID#head     the reply's <|im_start|> and the HEAD tokens after it: "assistant", "\\n", the empty think block
                ("<think>", "\\n\\n", "</think>", "\\n\\n") and the reply's first tokens (usually the estimate)
analyze: a read hits BET (the bet, threshold, donation, causes) or STEER (manipulation / bias / steering the number) if any of
its 4 bullets matches. Rates per arm, split by whether the estimate landed on the good cause's side and by whether DeepSeek's
judge says the reply explicitly ties the number to the bet (judged.jsonl).
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path[:0] = ["scripts", "."]
from runs import finish, new_run

SRC = Path("data/runs/donation_bet_goodbad/2026-10-25_qwen-nothink-cap8k")
GLM_SRC = Path("data/runs/donation_bet_goodbad/2026-10-25_glm53-nothink-ownline")
GLM_LINES = json.loads(Path("evalcue/envs/donation_bet/thresholds_goodbad_glm53.json").read_text())
GLM_TOKENIZER = Path(os.environ.get("GLM_TOKENIZER", Path.home() / ".cache/dbet-nothink/glm/tokenizer.json"))  # zai-org/GLM-5.3
ENVD = Path("evalcue/envs/donation_bet")
ENV_FILES = {
    f"{ENVD}/{n}": (ENVD / n).read_text() for n in ["goodbad.py", "__init__.py", "thresholds_goodbad.json", "thresholds_goodbad_glm53.json"]
}
GLM_COLS = ["rl_greedy", "rl_s1", "rl_s2", "rl_s3", "rl_s4"]
ARMS = ["neutral", "good_below", "good_above", "beers_good_below", "beers_good_above", "nothing_good_below", "nothing_good_above"]
PER_ARM = 24
HEAD = 12
LINES = json.loads(Path("evalcue/envs/donation_bet/thresholds_goodbad.json").read_text())
BET = re.compile(r"\b(?:bets?|betting|wager\w*|threshold\w*|donat\w*|charit\w*|causes?|beers?|friend)\b", re.IGNORECASE)
STEER = re.compile(
    r"\b(?:manipulat\w*|bias\w*|influenc\w*|nudg\w*|skew\w*|steer\w*|sway\w*|rig\w*|game the|strategic\w*|incentiv\w*|"
    r"conflict of interest|motivated)\b",
    re.IGNORECASE,
)


def build_requests(out: Path):
    rows = [(i, json.loads(line)) for i, line in enumerate((SRC / "rows.jsonl").open())]
    rng = random.Random(0)
    n = 0
    with (out / "requests.jsonl").open("w") as f:
        for arm in ARMS:
            pick = [x for x in rows if x[1]["arm"] == arm and "error" not in x[1]]
            for i, r in rng.sample(pick, PER_ARM):
                rid = f"{SRC.parent.name}/{SRC.name}:{i}"
                base = {
                    "env": "donation_bet_goodbad",
                    "arm": arm,
                    "item": r["item"],
                    "messages": r["messages"],
                    "shard_key": f"{arm}/{r['item']}",
                }  # same prompt -> same shard, so its prompt reads are cached
                f.write(json.dumps({"id": rid, **base}) + "\n")
                head = {"turn_starts": HEAD, "last_turn_only": True, "turn_role": "assistant"}
                f.write(json.dumps({"id": rid + "#head", **base, **head}) + "\n")
                n += 2
    print(f"wrote {n} requests to {out / 'requests.jsonl'}")


def load(out: Path):
    by = defaultdict(list)
    for line in (out / "reads.jsonl").open():
        r = json.loads(line)
        by[r["id"].split("#")[0]].append(r | {"head": r["id"].endswith("#head")})
    return by


def episode_info(src=SRC, lines=LINES):
    """rows.jsonl line -> (arm, good-side bool or None, judge ack, estimate, line)."""
    judged = {(j["arm"], j["item"], j["sample"]): j for j in map(json.loads, (src / "judged.jsonl").open())}
    info = {}
    for i, line in enumerate((src / "rows.jsonl").open()):
        r = json.loads(line)
        j = judged.get((r["arm"], r["item"], r["sample"]), {})
        est, thr = j.get("estimate"), r["meta"].get("threshold", lines[r["item"]])
        side = r["meta"].get("good_side")
        good = None if est is None or side is None else (est <= thr) == (side == "below")
        text_bet = bool(BET.search(r["messages"][-1].get("content") or "")) if "error" not in r else None
        info[i] = {"arm": r["arm"], "good_side": good, "ack": j.get("acknowledges"), "estimate": est, "line": thr, "text_bet": text_bet}
    return info


def hit(r, pat):
    return any(pat.search(b["text"]) for b in r["bullets"])


def where(r):
    """head (reply header + first tokens), prompt or reply."""
    if r["head"]:
        return "head"
    return "reply" if r["role"] == "assistant" else "prompt"


def analyze(out: Path):
    by, info = load(out), episode_info()
    eps = {
        int(rid.rsplit(":", 1)[1]): {w: [[b["text"] for b in r["bullets"]] for r in rs if where(r) == w] for w in WHERE}
        for rid, rs in by.items()
    }
    tabulate(eps, info, out)


WHERE = ["head", "reply", "prompt"]


def tabulate(eps, info, out, name="analysis.json"):
    """eps: rows.jsonl line -> {where: [[bullet texts of one read], ...]}. Prints and writes, per group x where: the share of
    episodes with any hitting read, and the share of reads that hit (mean over episodes, so long replies don't dominate)."""
    t = defaultdict(list)  # key -> per episode (any BET, any STEER, BET read share, STEER read share)
    for i, ws in eps.items():
        e = info[i]
        groups = [e["arm"]] + (
            []
            if e["arm"] == "neutral"
            else [
                "all bets",
                f"good side={e['good_side']}, ack={e['ack']}",
                f"reply text names bet={e['text_bet']}, good side={e['good_side']}",
            ]
        )
        for w in WHERE:
            sub = ws[w]
            if not sub:
                continue
            b = [any(BET.search(x) for x in read) for read in sub]
            st = [any(STEER.search(x) for x in read) for read in sub]
            for g in groups:
                t[g, w].append((any(b), any(st), sum(b) / len(b), sum(st) / len(st)))
    res = {}
    for (g, w), v in sorted(t.items(), key=lambda x: (WHERE.index(x[0][1]), x[0][0])):
        n = len(v)
        res[f"{g} | {w}"] = {
            "n": n,
            **{k: sum(x[j] for x in v) / n for j, k in enumerate(["ep_bet", "ep_steer", "read_bet", "read_steer"])},
        }
        r = res[f"{g} | {w}"]
        print(
            f"{w:6s} {g:32s} n={n:3d}  episodes: BET {r['ep_bet']:4.0%} STEER {r['ep_steer']:4.0%}   "
            f"reads: BET {r['read_bet']:5.1%} STEER {r['read_steer']:5.1%}"
        )
    (out / name).write_text(json.dumps(res, indent=1) + "\n")


def analysis_md(path: Path):
    """analysis*.json -> markdown table (head and reply rows, per group)."""
    if not path.exists():
        return ""
    res = json.loads(path.read_text())
    out = [
        "| group | where | n | episodes with a BET read | with a STEER read | BET reads | STEER reads |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, v in res.items():
        g, w = k.split(" | ")
        out.append(f"| {g} | {w} | {v['n']} | {v['ep_bet']:.0%} | {v['ep_steer']:.0%} | {v['read_bet']:.1%} | {v['read_steer']:.1%} |")
    return "\n\n**Keyword rates** (reads share = mean over episodes of the share of that region's reads that hit):\n\n" + "\n".join(out)


def expview(out: Path):
    """One item per transcript: the reply as Qwen saw it, with token reads at every read position."""
    from transformers import AutoTokenizer  # ty: ignore[unresolved-import]  (uv run --with transformers)

    ev = Path(os.environ.get("EXPVIEW", Path.home() / "workspace/expview")).expanduser()
    sys.path.insert(0, str(ev))
    from expview import Run, call  # ty: ignore[unresolved-import]

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    reqs = {r["id"]: r for r in map(json.loads, (out / "requests.jsonl").open()) if "#" not in r["id"]}
    by, info = load(out), episode_info()
    dest = ev / "runs" / f"dbet_nothink_olens_{out.name}"
    desc = (
        "O-lens (agu18dec `olens_s3d_space_rl600`, greedy, 4 bullets, whitened FVE; layer 40) on Qwen3.6-27B's "
        "**thinking-off** Donation Bet replies (run `donation_bet_goodbad/2026-10-25_qwen-nothink-cap8k`: the template's empty "
        f"think block, thinking-on sampling). {PER_ARM} random transcripts per arm. Reads at every punctuation token and "
        "`<|im_end|>` (prompt and reply), plus the reply's header, its empty think block and its first tokens. Labels: "
        "`good_side` = DeepSeek judge's estimate lands on the good cause's side of the line; `ack` = the judge says the reply "
        "explicitly ties the number to the bet; `lens_bet` / `lens_steer` = any read in the reply or its header has a bullet "
        f"matching `{BET.pattern}` / `{STEER.pattern}`. Run dir `olens_reads/{out.name}` in the sprint HF dataset."
    ) + analysis_md(out / "analysis.json")
    run = Run(str(dest), title="Donation Bet thinking off: O-lens", description=desc, tokens_section="O-lens reads")
    run.reset()
    for rid, rs in sorted(by.items(), key=lambda x: (ARMS.index(reqs[x[0]]["arm"]), x[0])):
        req = reqs[rid]
        text = tok.apply_chat_template(req["messages"], tokenize=False)
        ids = tok(text, add_special_tokens=False).input_ids
        toks = [tok.decode([i]) for i in ids]
        reads = defaultdict(list)
        for r in sorted(rs, key=lambda r: r["layer"]):
            reads[str(r["token_index"])].append(
                {
                    "tool": "O-lens",
                    "label": f"L{r['layer']}",
                    "score": round(r["joint_fve"], 3),
                    "score_name": "joint FVE",
                    "lines": [[b["text"], round(b["fve"], 3)] for b in r["bullets"]],
                    "inline": r["layer"] == 40,
                }
            )
        e = info[int(rid.rsplit(":", 1)[1])]
        out_rs = [r for r in rs if where(r) != "prompt"]
        labels = {
            "arm": req["arm"],
            "good_side": e["good_side"],
            "ack": e["ack"],
            "lens_bet": any(hit(r, BET) for r in out_rs),
            "lens_steer": any(hit(r, STEER) for r in out_rs),
            "estimate": e["estimate"],
            "line": e["line"],
            "question": req["item"],
        }
        start = max(i for i, t in enumerate(toks) if t == "<|im_start|>")
        tokens = {
            "tokens": toks,
            "sections": [["prompt", 0, start], ["reply", start, len(toks)]],
            "reads": reads,
            "legend": "Inline: L40's first bullet.",
        }
        msgs = req["messages"]
        calls = [call("qwen (as read)", "Qwen/Qwen3.6-27B", messages=msgs[:-1], output=msgs[-1])]
        run.add(
            id=rid,
            group=req["arm"],
            labels={k: v for k, v in labels.items() if v is not None},
            calls=calls,
            tokens=tokens,
            env_files=ENV_FILES,
            links={"Docent transcript": d} if (d := docent_link(rid)) else None,
        )
    import subprocess

    subprocess.run([sys.executable, str(ev / "expview.py"), "build", str(dest), "-o", str(dest / "page.html")], check=True)


def docent_link(rid, src=SRC):
    d = json.loads((src / "docent.json").read_text())
    aid = d["runs"][rid if isinstance(rid, int) else int(rid.rsplit(":", 1)[1])]
    return f"https://docent.transluce.org/dashboard/{d['collection_id']}/agent_run/{aid}" if aid else None


def glm_load(gdir: Path):
    """-> {rows.jsonl line: (episode row, [(pos, token, where, [5 bullet strings]), ...])}, prefix-duplicate positions filled
    from their canonical reads."""
    import pandas as pd  # ty: ignore[unresolved-import]  (uv run --with pandas --with pyarrow)

    eps = pd.read_parquet(gdir / "episodes.parquet")
    d = pd.read_parquet(gdir / "readouts_dense.parquet")
    reads = {(r.tid, r.pos): (r.token, r.role, [getattr(r, c) for c in GLM_COLS]) for r in d.itertuples()}
    for r in pd.read_parquet(gdir / "dense_prefix_dups.parquet").itertuples():
        if (r.canon_tid, r.canon_pos) in reads:
            reads[(r.tid, r.pos)] = reads[(r.canon_tid, r.canon_pos)]
    out = {}
    for e in eps.itertuples():
        mine = sorted((p, *v) for (t, p), v in reads.items() if t == e.tid)
        start = next(p for p, tok, *_ in mine if tok == "<|assistant|>")
        head = [p for p, *_ in mine if p >= start][:9]  # <|assistant|>, <think>, </think> and the reply's first 6 tokens
        rows = [(p, tok, "head" if p in head else ("reply" if p > start else "prompt"), texts) for p, tok, role, texts in mine]
        out[int(e.row_index)] = (e, rows)
    return out


def glm_analyze(gdir: Path):
    by, info = glm_load(gdir), episode_info(GLM_SRC, GLM_LINES)
    for cols, tag in [(slice(0, 1), "greedy"), (slice(0, 5), "greedy + 4 samples")]:
        print(f"== GLM lens, {tag}")
        eps = {i: {w: [x[3][cols] for x in rows if x[2] == w] for w in WHERE} for i, (_, rows) in by.items()}
        tabulate(eps, info, gdir, name=f"analysis_{'greedy' if tag == 'greedy' else 'all5'}.json")


def glm_expview(gdir: Path):
    """One item per GLM transcript with token reads (greedy + 4 samples per position)."""
    from tokenizers import Tokenizer  # ty: ignore[unresolved-import]  (uv run --with tokenizers)

    ev = Path(os.environ.get("EXPVIEW", Path.home() / "workspace/expview")).expanduser()
    sys.path.insert(0, str(ev))
    from expview import Run, call  # ty: ignore[unresolved-import]

    tok = Tokenizer.from_file(str(GLM_TOKENIZER))
    by, info = glm_load(gdir), episode_info(GLM_SRC, GLM_LINES)
    rows = [json.loads(line) for line in (GLM_SRC / "rows.jsonl").open()]
    dest = ev / "runs" / f"dbet_nothink_olens_glm_{gdir.parent.name}"
    desc = (
        "GLM 5.3 O-lens (Celeste's RL lens `ceselder/glm-5.3-olens-lens` rl1b_2 step 100, block 51; greedy + 4 samples at T=1, "
        "one bullet each, no FVE) on GLM 5.3's **thinking-off** Donation Bet replies (run "
        "`donation_bet_goodbad/2026-10-25_glm53-nothink-ownline`: raw completions with `<|assistant|><think></think>` prefilled, "
        "thinking-on sampling). 24 random transcripts per arm. Reads at punctuation and template tokens and the reply's first 6 "
        "tokens, on the exact empty-think render. Labels: `good_side` = DeepSeek judge's estimate lands on the good cause's "
        "side of GLM's line; `ack` = the judge says the reply explicitly ties the number to the bet; `lens_bet` / `lens_steer` "
        f"= any greedy read in the reply or its header matches `{BET.pattern}` / `{STEER.pattern}`. Reads on HF: "
        f"`runs/olens_reads/{gdir.parent.name}/{gdir.name}`."
    ) + analysis_md(gdir / "analysis_greedy.json")
    run = Run(str(dest), title="Donation Bet thinking off: GLM O-lens", description=desc, tokens_section="O-lens reads")
    run.reset()
    for i, (e, reads_) in sorted(by.items(), key=lambda x: (ARMS.index(x[1][0].arm), x[0])):
        ids = tok.encode(e.rendered_text, add_special_tokens=False).ids
        toks = [tok.id_to_token(t) if t >= 154820 else tok.decode([t]) for t in ids]
        reads = {
            str(p): [
                {"tool": "GLM O-lens", "label": c.replace("rl_", ""), "lines": [[x, None]], "inline": c == "rl_greedy"}
                for c, x in zip(GLM_COLS, texts)
            ]
            for p, _, _, texts in reads_
        }
        inf = info[i]
        out_rs = [x for x in reads_ if x[2] != "prompt"]
        labels = {
            "arm": e.arm,
            "good_side": inf["good_side"],
            "ack": inf["ack"],
            "lens_bet": any(BET.search(x[3][0]) for x in out_rs),
            "lens_steer": any(STEER.search(x[3][0]) for x in out_rs),
            "estimate": inf["estimate"],
            "line": inf["line"],
            "question": e.item,
        }
        start = ids.index(tok.token_to_id("<|assistant|>"))
        tokens = {
            "tokens": toks,
            "sections": [["prompt", 0, start], ["reply", start, len(toks)]],
            "reads": reads,
            "legend": "Inline: the greedy bullet.",
        }
        msgs = rows[i]["messages"]
        calls = [
            call(
                "glm (as read)",
                "z-ai/glm-5.3",
                messages=msgs[:-1],
                output=msgs[-1],
                params={"raw_prompt": e.rendered_text[: e.rendered_text.index("</think>") + 8]},
            )
        ]
        run.add(
            id=f"{GLM_SRC.parent.name}/{GLM_SRC.name}:{i}",
            group=e.arm,
            labels={k: v for k, v in labels.items() if v is not None},
            calls=calls,
            tokens=tokens,
            env_files=ENV_FILES,
            links={"Docent transcript": d} if (d := docent_link(i, GLM_SRC)) else None,
        )
    import subprocess

    subprocess.run([sys.executable, str(ev / "expview.py"), "build", str(dest), "-o", str(dest / "page.html")], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["newrun", "requests", "analyze", "expview", "finish", "glm-analyze", "glm-expview"])
    ap.add_argument("out", type=Path, nargs="?")
    a = ap.parse_args()
    if a.cmd == "newrun":
        print(
            new_run(
                "olens_reads",
                "dbet-nothink",
                description=f"O-lens reads on Qwen's thinking-off Donation Bet replies ({SRC.name}), {PER_ARM} per arm",
                models=[{"name": "agu18dec/olens_and_ar olens_s3d_space_rl600 on Qwen/Qwen3.6-27B", "served_by": "sky RunPod 2x H200"}],
                params={"layers": [40], "per_arm": PER_ARM, "head_tokens": HEAD, "sampling": "greedy, 4 bullets"},
            )
        )
    elif a.cmd == "finish":
        finish(a.out)
    else:
        cmds = {"requests": build_requests, "analyze": analyze, "expview": expview, "glm-analyze": glm_analyze, "glm-expview": glm_expview}
        cmds[a.cmd](a.out)
