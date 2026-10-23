"""Model-forensics reads (O-lens, J++, NLA) on Donation Bet transcripts, through the case page's read server.

    uv run python scripts/dbet_forensics.py select             # new run + the 32 transcripts, tokenised
    uv run python scripts/dbet_forensics.py read RUN --phase punct [--url http://127.0.0.1:18822]
    uv run python scripts/dbet_forensics.py read RUN --phase rest
    uv run python scripts/dbet_forensics.py status RUN

Transcripts: data/runs/donation_bet_goodbad/2026-10-22_{bets,beers-nothing}, 8 per arm for good_below, good_above,
beers_good_below, beers_good_above, each arm mixing the judge's CoT-acknowledges-bet label and whether the estimate landed
on the good-cause side of the line (2 per combination where the arm has them). Token ids are the chat template with
thinking on, then reasoning + "</think>" + reply + "<|im_end|>", re-tokenised from the stored text (vLLM's own sampled
ids weren't kept).

The server is transcript-viewer/pod/server.py (sky/pod_h200.yaml, reached over an ssh tunnel): /read gives O-lens
readouts (space lens, 4 bullets, per-bullet + joint FVE) at (pos, layer); /extras gives J++ top-k at every layer and the
ceselder L42 NLA explanation with its critic FVE. Phase "punct" reads the chat-template, newline and punctuation tokens
at O-lens layers PUNCT_LAYERS; phase "rest" reads every other token at REST_LAYERS. Outputs append to reads.jsonl and
extras.jsonl in the run dir, so rerunning resumes.
"""

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

SRC = Path("data/runs/donation_bet_goodbad")
ARMS = ["good_below", "good_above", "beers_good_below", "beers_good_above"]
PER_ARM = 8
PUNCT_LAYERS = [32, 40, 48]
REST_LAYERS = [40]
MODEL = "Qwen/Qwen3.6-27B"
REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
PUNCT = re.compile(r"^\s*[.,;:!?]+[\"')\]*]*\s*$")


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION)


def render(tok, row):
    """ids + decoded tokens + message token ranges for one Donation Bet row."""
    user = row["messages"][0]["content"]
    out = row["messages"][-1]
    prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    reasoning, reply = out.get("reasoning_content") or "", out.get("content") or ""
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    r_ids = tok(reasoning, add_special_tokens=False).input_ids
    tail = tok("</think>" + reply + "<|im_end|>", add_special_tokens=False).input_ids
    ids = p_ids + r_ids + tail
    toks = [tok.decode([i]) for i in ids]
    a = len(p_ids)
    return {
        "ids": ids,
        "tokens": toks,
        "sections": {"prompt": [0, a], "reasoning": [a, a + len(r_ids)], "reply": [a + len(r_ids), len(ids)]},
    }


def punct_positions(tok, ids, toks):
    special = {i for i, a in tok.added_tokens_decoder.items() if a.special} | {
        tok.convert_tokens_to_ids(x) for x in ("<think>", "</think>")
    }
    start = tok.convert_tokens_to_ids("<|im_start|>")
    pos = []
    for i, (x, t) in enumerate(zip(ids, toks)):
        if x in special or (i and ids[i - 1] == start) or ("\n" in t and not t.strip()) or PUNCT.match(t):
            pos.append(i)
    return pos


def select(args):
    from runs import new_run

    judged, rows = {}, []
    for s in ["2026-10-22_bets", "2026-10-22_beers-nothing"]:
        judged |= {(j["arm"], j["item"], j["sample"]): j for j in map(json.loads, (SRC / s / "judged.jsonl").open())}
        rows += [dict(r, source=s) for r in map(json.loads, (SRC / s / "rows.jsonl").open()) if r["arm"] in ARMS and "error" not in r]
    cells = defaultdict(list)
    for r in rows:
        j = judged.get((r["arm"], r["item"], r["sample"]), {})
        if j.get("estimate") is None or "judge_error" in j:
            continue
        good = "below" if r["arm"].endswith("_below") else "above"
        on_good = (j["estimate"] <= r["meta"]["threshold"]) == (good == "below")
        r["judge"], r["on_good_side"] = j, on_good
        cells[(r["arm"], bool(j["acknowledges"]), on_good)].append(r)
    rng = random.Random(args.seed)
    picked = []
    for arm in ARMS:
        combos = [(a, g) for a in (True, False) for g in (True, False)]
        take = {c: min(2, len(cells[(arm, *c)])) for c in combos}
        while sum(take.values()) < PER_ARM:  # top up from the combos that have more
            c = max(combos, key=lambda c: len(cells[(arm, *c)]) - take[c])
            take[c] += 1
        for c in combos:
            pool = cells[(arm, *c)][:]
            rng.shuffle(pool)
            # spread over questions: one per question first
            seen, chosen = set(), []
            for r in pool:
                if len(chosen) < take[c] and r["item"] not in seen:
                    chosen.append(r)
                    seen.add(r["item"])
            chosen += [r for r in pool if r not in chosen][: take[c] - len(chosen)]
            picked += chosen
    run = new_run(
        "donation_bet_forensics",
        args.slug,
        description=(
            "O-lens (space lens) / J++ / ceselder L42 NLA reads on 32 Donation Bet goodbad transcripts"
            " (8 per arm: good vs bad and good vs beers, both sides)"
        ),
        models=[
            {"name": MODEL, "revision": REVISION, "served_by": "SkyPilot RunPod H200, transcript-viewer/pod/server.py (HF bf16)"},
            {"name": "agu18dec/olens_and_ar olens_s3d_space_rl600 + AR"},
            {"name": "J++ lens (olens.extras.JPP)"},
            {"name": "ceselder L42 NLA (av_sft_lora + av_rl_lora_step400) + critic"},
        ],
        params={
            "per_arm": PER_ARM,
            "punct_layers": PUNCT_LAYERS,
            "rest_layers": REST_LAYERS,
            "max_bullets": 4,
            "temperature": 0,
            "jpp_k": 10,
            "seed": args.seed,
        },
    )
    tok = tokenizer()
    with open(run / "transcripts.jsonl", "w") as f:
        for r in picked:
            rd = render(tok, r)
            tid = f"{r['arm']}/{r['item']}/s{r['sample']:04d}"
            rec = {
                "tid": tid,
                "arm": r["arm"],
                "item": r["item"],
                "sample": r["sample"],
                "source": r["source"],
                "meta": r["meta"],
                "judge": r["judge"],
                "on_good_side": r["on_good_side"],
                "messages": r["messages"],
                **rd,
                "punct": punct_positions(tok, rd["ids"], rd["tokens"]),
            }
            f.write(json.dumps(rec) + "\n")
    n = [json.loads(x) for x in (run / "transcripts.jsonl").open()]
    print(run, len(n), "transcripts,", sum(len(x["ids"]) for x in n), "tokens,", sum(len(x["punct"]) for x in n), "punct positions")


def call(url, path, body, timeout=3600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(json.loads(e.read() or b"{}").get("error", f"HTTP {e.code}")) from None


def done_keys(path, key):
    return {key(json.loads(x)) for x in path.open()} if path.exists() else set()


def read(args):
    """Phase jpp: J++ at every token (cheap). Phase punct / rest: O-lens at PUNCT_LAYERS / REST_LAYERS plus the NLA, on the
    punctuation + template tokens / every other token."""
    run = Path(args.run)
    trs = [json.loads(x) for x in (run / "transcripts.jsonl").open()]
    sfx = f".{args.shard}" if args.shard else ""
    rpath, npath, jpath = run / f"reads{sfx}.jsonl", run / f"nla{sfx}.jsonl", run / f"jpp{sfx}.jsonl"
    key2 = lambda r: (r["tid"], r["pos"])
    have_r = set().union(*[done_keys(p, lambda r: (r["tid"], r["pos"], r["layer"])) for p in run.glob("reads*.jsonl")])
    have_n = set().union(*[done_keys(p, key2) for p in run.glob("nla*.jsonl")])
    have_j = set().union(*[done_keys(p, key2) for p in run.glob("jpp*.jsonl")])
    if args.only:
        keep = set(args.only.split(","))
        trs = [t for t in trs if t["tid"] in keep]
    t0, n = time.time(), defaultdict(int)
    for t in trs:
        tid = t["tid"]
        if args.phase == "jpp":
            todo = [p for p in range(len(t["ids"])) if (tid, p) not in have_j]
            for i in range(0, len(todo), args.chunk_jpp):
                ch = todo[i : i + args.chunk_jpp]
                res = call(args.url, "/extras", {"ids": t["ids"], "positions": ch, "nla": False, "jpp": True, "k": 10})
                with jpath.open("a") as f:
                    for p in ch:
                        f.write(json.dumps({"tid": tid, "pos": p, "jpp": res["jpp"][str(p)]}) + "\n")
                if res.get("jpp_layers") and not (run / "jpp_layers.json").exists():
                    (run / "jpp_layers.json").write_text(json.dumps(res["jpp_layers"]))
                n["jpp"] += len(ch)
        else:
            if args.phase == "punct":
                pos, layers = t["punct"], args.layers or PUNCT_LAYERS
            else:
                ps = set(t["punct"])
                pos, layers = [i for i in range(len(t["ids"])) if i not in ps], args.layers or REST_LAYERS
            todo_r = [(p, L) for p in pos for L in layers if (tid, p, L) not in have_r]
            for i in range(0, len(todo_r), args.chunk_reads):
                ch = todo_r[i : i + args.chunk_reads]
                res = call(
                    args.url,
                    "/read",
                    {"ids": t["ids"], "items": [{"pos": p, "layer": L} for p, L in ch], "temperature": 0, "n": 1, "max_bullets": 4},
                )
                with rpath.open("a") as f:
                    for (p, L), out in zip(ch, res["results"]):
                        f.write(json.dumps({"tid": tid, "pos": p, "layer": L, **out[0]}) + "\n")
                n["olens"] += len(ch)
            if not args.no_nla:
                todo_n = [p for p in pos if (tid, p) not in have_n]
                for i in range(0, len(todo_n), args.chunk_nla):
                    ch = todo_n[i : i + args.chunk_nla]
                    res = call(args.url, "/extras", {"ids": t["ids"], "positions": ch, "nla": True, "jpp": False})
                    with npath.open("a") as f:
                        for p, x in zip(ch, res["nla"]):
                            f.write(json.dumps({"tid": tid, "pos": p, "nla": x}) + "\n")
                    n["nla"] += len(ch)
        print(f"{tid}: {dict(n)} in {time.time() - t0:.0f}s", flush=True)


def subprocess_after(cmd):
    import subprocess

    subprocess.run(cmd, shell=True, check=False)


def status(args):
    run = Path(args.run)
    trs = [json.loads(x) for x in (run / "transcripts.jsonl").open()]
    nr = sum(1 for p in run.glob("reads*.jsonl") for _ in p.open())
    nx = {k: sum(1 for p in run.glob(f"{k}*.jsonl") for _ in p.open()) for k in ("jpp", "nla")}
    npu = sum(len(t["punct"]) for t in trs)
    ntok = sum(len(t["ids"]) for t in trs)
    print(
        f"lens reads {nr} (punct target {npu * len(PUNCT_LAYERS)},"
        f" all target {npu * len(PUNCT_LAYERS) + (ntok - npu) * len(REST_LAYERS)});"
        f" J++ {nx['jpp']}/{ntok}; NLA {nx['nla']}/{ntok}"
    )


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("select")
    s.add_argument("--slug", default="olens-jpp-nla")
    s.add_argument("--seed", type=int, default=0)
    r = sp.add_parser("read")
    r.add_argument("run")
    r.add_argument("--phase", choices=["jpp", "punct", "rest"], required=True)
    r.add_argument("--layers", type=lambda x: [int(v) for v in x.split(",")], help="override the phase O-lens layers")
    r.add_argument("--no-nla", action="store_true", help="O-lens only in punct/rest")
    r.add_argument("--chunk-jpp", type=int, default=1024)
    r.add_argument("--chunk-nla", type=int, default=128)
    r.add_argument("--url", default="http://127.0.0.1:18822")
    r.add_argument("--chunk-reads", type=int, default=384, help="O-lens (pos, layer) reads per /read call (one capture each)")
    r.add_argument("--shard", help="write reads.<shard>.jsonl / extras.<shard>.jsonl (one per pod when several run at once)")
    r.add_argument("--only", help="comma-separated tids (for sharding across pods)")
    pa = sp.add_parser("pairs")
    pa.add_argument("run")
    pa.add_argument("--seed", type=int, default=1)
    pa.add_argument("--mode", choices=["mirror", "nothing_beers", "nothing_beers_full", "nothing_beers_ack"], default="nothing_beers_full")
    st = sp.add_parser("status")
    st.add_argument("run")
    a = ap.parse_args()
    {"select": select, "read": read, "status": status, "pairs": pairs}[a.cmd](a)


def pairs(args):
    """Matched pairs for the case page: each 'good wins if the estimate exceeds' transcript (good_above, beers_good_above) gets a
    same-question partner from its mirror arm (good_below, beers_good_below), reusing a sampled transcript where one is free and
    otherwise adding one from the source run (tokenised and appended to transcripts.jsonl, so the read queue picks it up).
    The rollouts had no per-request seed, so pairs match on question only."""
    run = Path(args.run)
    trs = [json.loads(x) for x in (run / "transcripts.jsonl").open()]
    judged, rows = {}, defaultdict(list)
    for s in ["2026-10-22_bets", "2026-10-22_beers-nothing"]:
        judged |= {(j["arm"], j["item"], j["sample"]): j for j in map(json.loads, (SRC / s / "judged.jsonl").open())}
        for r in map(json.loads, (SRC / s / "rows.jsonl").open()):
            if "error" not in r and judged.get((r["arm"], r["item"], r["sample"]), {}).get("estimate") is not None:
                rows[(r["arm"], r["item"])].append(dict(r, source=s))
    rng = random.Random(args.seed)
    tok, out, new = None, [], []

    def above(x):
        j = x.get("judge") or judged[(x["arm"], x["item"], x["sample"])]
        return j["estimate"] > x["meta"]["threshold"]

    def ack(x):
        return bool((x.get("judge") or judged[(x["arm"], x["item"], x["sample"])])["acknowledges"])

    def get(arm, item, exclude=(), want=None, want_ack=None):
        """a transcript of (arm, item) already in the run (and not in exclude), else a new one from the source run; want=True/False
        asks for an estimate above / not above the line. None if no rollout fits."""
        nonlocal tok
        ok = lambda x: (want is None or above(x) == want) and (want_ack is None or ack(x) == want_ack)
        t = next((t for t in trs + new if t["arm"] == arm and t["item"] == item and t["tid"] not in exclude and ok(t)), None)
        if t:
            return t
        taken = {t["sample"] for t in trs + new if t["arm"] == arm and t["item"] == item}
        pool = [r for r in rows[(arm, item)] if r["sample"] not in taken and ok(r)]
        if not pool:
            return None
        r = rng.choice(pool)
        tok = tok or tokenizer()
        j = judged[(r["arm"], r["item"], r["sample"])]
        rd = render(tok, r)
        below = arm.endswith("_below")
        t = {
            "tid": f"{r['arm']}/{r['item']}/s{r['sample']:04d}",
            "arm": r["arm"],
            "item": r["item"],
            "sample": r["sample"],
            "source": r["source"],
            "meta": r["meta"],
            "judge": j,
            "on_good_side": (j["estimate"] <= r["meta"]["threshold"]) == below,
            "messages": r["messages"],
            **rd,
            "punct": punct_positions(tok, rd["ids"], rd["tokens"]),
            "added_for": "pairs",
        }
        new.append(t)
        return t

    if args.mode == "nothing_beers_full":  # every question x direction: a both-above, a both-below and a split pair where rollouts allow
        items = sorted({k[1] for k in rows})
        for d in ["above", "below"]:
            for item in items:
                na, ba = f"nothing_good_{d}", f"beers_good_{d}"
                for kind, combos in [
                    ("both above", [(True, True)]),
                    ("both below", [(False, False)]),
                    ("split", [(True, False), (False, True)]),
                ]:
                    for wa, wb in combos:
                        a = get(na, item, want=wa)
                        b = get(ba, item, want=wb) if a else None
                        if a and b:
                            out.append(
                                {
                                    "comparison": f"nothing vs beers · good wins if {'>' if d == 'above' else '≤'} line",
                                    "question": item,
                                    "kind": kind,
                                    "a": a["tid"],
                                    "b": b["tid"],
                                }
                            )
                            break
    if (
        args.mode == "nothing_beers_ack"
    ):  # add pairs where the CoT acknowledges the bet: both / nothing only / beers only, up to 3 each per direction
        out = json.loads((run / "pairs_nothing_beers.json").read_text())
        have = {(p["a"], p["b"]) for p in out}
        items = sorted({k[1] for k in rows})
        for d in ["above", "below"]:
            na, ba = f"nothing_good_{d}", f"beers_good_{d}"
            for kind, (xa, xb) in [("ack both", (True, True)), ("ack nothing only", (True, False)), ("ack beers only", (False, True))]:
                got = 0
                for item in items:
                    if got >= 3:
                        break
                    if not any(ack(r) == xa for r in rows[(na, item)]) or not any(ack(r) == xb for r in rows[(ba, item)]):
                        continue
                    a, b = get(na, item, want_ack=xa), get(ba, item, want_ack=xb)
                    if a and b and (a["tid"], b["tid"]) not in have:
                        out.append(
                            {
                                "comparison": f"nothing vs beers · good wins if {'>' if d == 'above' else '≤'} line",
                                "question": item,
                                "kind": kind,
                                "a": a["tid"],
                                "b": b["tid"],
                            }
                        )
                        have.add((a["tid"], b["tid"]))
                        got += 1
    if args.mode == "nothing_beers":  # same question, same direction: good vs nothing (left) and good vs beers (right)
        items = sorted({k[1] for k in rows})
        for d in ["above", "below"]:
            for item in items:
                a, b = get(f"nothing_good_{d}", item), get(f"beers_good_{d}", item)
                out.append(
                    {
                        "comparison": f"nothing vs beers · good wins if {'>' if d == 'above' else '≤'} line",
                        "question": item,
                        "a": a["tid"],
                        "b": b["tid"],
                    }
                )
    for anchor, mirror in [("good_above", "good_below"), ("beers_good_above", "beers_good_below")] if args.mode == "mirror" else []:
        free = [t for t in trs if t["arm"] == mirror]
        for a in [t for t in trs if t["arm"] == anchor]:
            b = next((t for t in free if t["item"] == a["item"]), None)
            if b:
                free.remove(b)
            else:
                taken = {t["sample"] for t in trs + new if t["arm"] == mirror and t["item"] == a["item"]}
                r = rng.choice([r for r in rows[(mirror, a["item"])] if r["sample"] not in taken])
                tok = tok or tokenizer()
                j = judged[(r["arm"], r["item"], r["sample"])]
                rd = render(tok, r)
                b = {
                    "tid": f"{r['arm']}/{r['item']}/s{r['sample']:04d}",
                    "arm": r["arm"],
                    "item": r["item"],
                    "sample": r["sample"],
                    "source": r["source"],
                    "meta": r["meta"],
                    "judge": j,
                    "on_good_side": j["estimate"] <= r["meta"]["threshold"],
                    "messages": r["messages"],
                    **rd,
                    "punct": punct_positions(tok, rd["ids"], rd["tokens"]),
                    "added_for": "pairs",
                }
                new.append(b)
            out.append(
                {
                    "comparison": "good vs bad" if anchor == "good_above" else "good vs beers",
                    "question": a["item"],
                    "a": a["tid"],
                    "b": b["tid"],
                }
            )
    with open(run / "transcripts.jsonl", "a") as f:
        f.writelines(json.dumps(t) + "\n" for t in new)
    (run / f"pairs_{args.mode.removesuffix('_full').removesuffix('_ack')}.json").write_text(json.dumps(out, indent=1))
    print(f"{len(out)} pairs, {len(new)} new transcripts ({sum(len(t['ids']) for t in new)} tokens)")


if __name__ == "__main__":
    main()
