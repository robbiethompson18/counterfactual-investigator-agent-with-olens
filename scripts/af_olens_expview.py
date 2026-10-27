"""expview run of the O-lens reads on the reworded alignment-faking runs (v2-v8, thinking off): GLM 5.3 (Celeste's RL lens,
scripts/glm_olens_dense) and DeepSeek V4 Flash (Camila's SFT inverter at layer 30, scripts/dsv4_olens), one item per read
transcript with the reads inline at their tokens, the subject call, the env's files and the episode's Docent run.

    uv run --with transformers python scripts/af_olens_expview.py --envs data/envs/alignment_faking \
        --glm-ds <dir with one readouts_dense dir per GLM run> --glm-rows <rows dir> --glm-tok <GLM-5.3 tokenizer dir> \
        --dsv4 <dir with requests.jsonl, logits.jsonl, reads_L30.jsonl> --dsv4-tok <V4 tokenizer.json> --out RUN/expview
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, "expview")
from expview import Run, call, read

DOCENT = "https://docent.transluce.org/dashboard/{}/agent_run/{}"
GLM_SPECIAL = ("<|system|>", "<|user|>", "<|assistant|>")


def load(path: str, name: str):
    """import a script by path (both pipelines have a prep.py, so not via sys.path)"""
    import importlib.util

    sys.path.insert(0, str(Path(path).parent))  # its own siblings (render.py)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def docent(envs: Path, run: str) -> dict:
    p = envs / run / "docent.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {ep: DOCENT.format(d["collection_id"], rid) for ep, rid in d["runs"].items() if rid}


def sections(toks: list[str], markers: dict[str, str], reply_from: int) -> list:
    """chat blocks from the role markers: [name, start, end)."""
    cuts = [(i, markers[t]) for i, t in enumerate(toks) if t in markers]
    out = []
    for k, (i, name) in enumerate(cuts):
        end = cuts[k + 1][0] if k + 1 < len(cuts) else len(toks)
        if name == "assistant":
            out.append(["reply", i, len(toks)])
            break
        out.append([name, i, end])
    if out and out[0][1] > 0:
        out[0][1] = 0
    return out


def glm_items(run: Run, a, envs: Path, env_id: str):
    P = load("scripts/glm_olens_dense/prep.py", "glm_prep")  # the raw-prefill render the runs were sampled with
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(a.glm_tok)
    for ds in sorted(Path(a.glm_ds).iterdir()):
        name = ds.name
        R = pd.read_parquet(ds / "readouts_dense.parquet", columns=["tid", "pos", "token", "rl_greedy", "rl_s1", "rl_s2", "rl_s3", "rl_s4"])
        by = {(t, int(p)): r for t, p, r in zip(R.tid, R.pos, R.itertuples(index=False), strict=True)}
        D = pd.read_parquet(ds / "dense_prefix_dups.parquet")
        canon = {(t, int(p)): (ct, int(cp)) for t, p, ct, cp in zip(D.tid, D.pos, D.canon_tid, D.canon_pos, strict=True)}
        E = pd.read_parquet(ds / "episodes.parquet", columns=["tid", "row_index", "positions"])
        rows = [json.loads(x) for x in (Path(a.glm_rows) / f"{name}.jsonl").read_text().splitlines()]
        links = docent(envs, name)
        n_items = n_reads = 0
        for tid, ri, poss in zip(E.tid, E.row_index, E.positions, strict=True):
            r = rows[int(ri)]
            text = P.raw_render(r["messages"])
            ids = tk(text, add_special_tokens=False)["input_ids"]
            toks = [tk.decode([i]) for i in ids]
            reads = {}
            for p in map(int, poss):
                x = by.get((tid, p)) or by.get(canon.get((tid, p), ("", -1)))
                if x is None:
                    continue
                assert x.token == toks[p], (name, tid, p, x.token, toks[p])
                reads[str(p)] = [
                    read("O-lens GLM", [x.rl_greedy], label="RL greedy", inline=True),
                    read("O-lens GLM samples", [[s] for s in (x.rl_s1, x.rl_s2, x.rl_s3, x.rl_s4) if isinstance(s, str)]),
                ]
            secs = sections(toks, {"<|system|>": "system", "<|user|>": "user", "<|assistant|>": "assistant"}, 0)
            n_reads += len(reads)
            run.add(
                id=f"glm53/{r['episode']}",
                group=f"GLM 5.3 {r['version']}/{r['stage']}",
                labels={
                    "model": "GLM 5.3",
                    "version": r["version"],
                    "stage": r["stage"],
                    "positive": r["positive"],
                    "verdict": r["verdict"],
                },
                calls=[call("subject", r["model"], messages=r["messages"], params={"backend": r["backend"], "thinking": False})],
                env=env_id,
                tokens={"tokens": toks, "sections": secs, "reads": reads, "legend": GLM_LEGEND},
                links={"Docent": links[r["episode"]]} if r["episode"] in links else None,
                extra={"source": f"envs/alignment_faking/{name}", "episode": r["episode"], "item": r["item"], "sample": r["sample"]},
            )
            n_items += 1
        print(f"glm {name}: {n_items} items, {n_reads:,} read positions")


def dsv4_items(run: Run, a, envs: Path, env_id: str):
    RD = load("scripts/dsv4_olens/read.py", "dsv4_read")  # the render the reads were captured on
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_file=a.dsv4_tok)
    d = Path(a.dsv4)
    reqs = [json.loads(x) for x in (d / "requests.jsonl").read_text().splitlines()]
    logits = {(x := json.loads(line))["id"]: x for line in (d / "logits.jsonl").read_text().splitlines()}
    reads = defaultdict(dict)
    for line in (d / f"reads_L{a.dsv4_layer}.jsonl").read_text().splitlines():
        x = json.loads(line)
        reads[x["id"]][x["pos"]] = x
    # system-prompt positions were read once per distinct system prompt (identical activations): share them by version
    sys_reads = defaultdict(dict)
    for q in reqs:
        for p, x in reads[q["id"]].items():
            if x["role"] == "system":
                sys_reads[q["version"]][p] = x
    links = {}
    for run_name in {q["run"] for q in reqs}:
        links |= docent(envs, run_name)
    for q in reqs:
        prompt, reply = RD.render(q["messages"])
        enc = tok(prompt + reply, add_special_tokens=False)
        toks = [tok.decode([i]) for i in enc["input_ids"]]
        rs = {}
        for p, x in {**sys_reads[q["version"]], **reads[q["id"]]}.items():
            assert x["token"] == toks[p] or x["role"] == "assistant", (q["id"], p, x["token"], toks[p])
            rs[str(p)] = [
                read("O-lens V4", [x["greedy"]], label=f"L{a.dsv4_layer} greedy", inline=True),
                read("O-lens V4 samples", [[x[k]] for k in ("s1", "s2") if k in x]),
            ]
        lg = logits.get(q["id"], {})
        secs = sections(toks, {"<｜User｜>": "user", "<｜Assistant｜>": "assistant"}, 0)
        secs = [["system", 0, secs[0][1]]] + secs if secs and secs[0][0] == "user" and secs[0][1] > 0 else secs
        labels = {"model": "DSV4 Flash", "version": q["version"], "stage": q["arm"], "positive": q["positive"], "verdict": q["verdict"]}
        if lg:
            labels["logP_accept_minus_reject"] = round(lg["accept"] - lg["reject"], 3)
        full = dict(q["messages"][-1])
        run.add(
            id=f"dsv4flash/{q['episode']}",
            group=f"DSV4 Flash {q['version']}/{q['arm']}",
            labels=labels,
            calls=[call("subject", "deepseek/deepseek-v4-flash", messages=q["messages"][:-1] + [full], params={"thinking": False})],
            env=env_id,
            tokens={"tokens": toks, "sections": secs, "reads": rs, "legend": DSV4_LEGEND},
            links={"Docent": links[q["episode"]]} if q["episode"] in links else None,
            extra={
                "source": f"envs/alignment_faking/{q['run']}",
                "episode": q["episode"],
                "reply_chars": q["reply_chars"],
                "reply_cut_at": a.reply_chars,
                "first_token_top10": lg.get("top_first"),
            },
        )
    print(f"dsv4: {len(reqs)} items")


GLM_LEGEND = (
    "GLM 5.3 (thinking off: raw completion with an empty think block, as sampled). Celeste's GLM-5.3 O-lens "
    "(`ceselder/glm-5.3-olens-lens`, RL `rl1b_rl1b_2_step_0100`), block 51: greedy inline, 4 samples at T=1 in the panel. "
    "Read at every token of the user turn, every punctuation / template token of the system prompt and the first 8 reply tokens."
)
DSV4_LEGEND = (
    "DeepSeek V4 Flash (thinking off). Camila Blank's V4 Flash O-lens inverter (`camila-blank/deepseek-v4-flash-olens-inverter`, "
    "SFT) at layer 30 of 43: greedy inline, 2 samples (T 1.0, top_p 0.95) in the panel. Read at every token of the user turn and the "
    "reply (cut at its first 120 characters), and every punctuation / special token of the system prompt (read once per prompt "
    "version, shared across items: the activations are identical). Not the same lens family as GLM's: its reads continue "
    "the text rather than writing bullets."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", required=True)
    ap.add_argument("--env-dir", default="environments/alignment_faking")
    ap.add_argument("--glm-ds")
    ap.add_argument("--glm-rows")
    ap.add_argument("--glm-tok")
    ap.add_argument("--dsv4")
    ap.add_argument("--dsv4-tok")
    ap.add_argument("--dsv4-layer", type=int, default=30)
    ap.add_argument("--reply-chars", type=int, default=120)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Run(
        a.out,
        title="Alignment faking v2-v8: O-lens reads (GLM 5.3, DSV4 Flash)",
        description=(
            "O-lens reads on the reworded alignment-faking prompts (`environments/alignment_faking`, v2-v8, thinking off, "
            "preview vs public `<deployment_stage>`). One item per transcript; `positive` = complied "
            "(`<request_fully_accepted/>` and no refusal), `verdict` = the detector's label. GLM: every episode of the GLM "
            "runs. DSV4 Flash: sample 0 of q0-q49 per arm; `logP_accept_minus_reject` = log P(accept tag) - log P(rejected tag) "
            "as the first reply tokens."
        ),
        label_order=["model", "version", "stage", "positive", "verdict"],
        readers={
            "O-lens GLM": {"about": GLM_LEGEND, "color": "#d9822b"},
            "O-lens GLM samples": {"about": "4 samples at T=1", "color": "#e8a85f"},
            "O-lens V4": {"about": DSV4_LEGEND, "color": "#c0592a"},
            "O-lens V4 samples": {"about": "2 samples at T=1", "color": "#d98a5f"},
        },
    )
    run.reset()
    env_id = run.env(dir=a.env_dir, skip=("__pycache__",))
    envs = Path(a.envs)
    if a.glm_ds:
        glm_items(run, a, envs, env_id)
    if a.dsv4:
        dsv4_items(run, a, envs, env_id)


if __name__ == "__main__":
    main()
