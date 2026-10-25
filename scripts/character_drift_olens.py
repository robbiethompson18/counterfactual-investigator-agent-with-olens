"""O-lens reads for character_drift: what does the lens say at the Assistant token after drifted vs default-voice turns?

    uv run python scripts/character_drift_olens.py requests OUT_DIR    # -> OUT_DIR/requests.jsonl for scripts/olens_read_requests.py
    uv run python scripts/character_drift_olens.py analyze OUT_DIR     # keyword counts per prefill arm from OUT_DIR/reads.jsonl
    uv run python scripts/character_drift_olens.py expview OUT_DIR     # token-read expview page ($EXPVIEW/runs/<name>/page.html)

Requests (space lens olens_s3d_space_rl600, read with `turn_starts` = only each <|im_start|> and the tokens after it):
    prefix   every Sonnet scenario under drift / qwensys / sysrp / default, and every WildChat chat under drift / qwensys, with
             the identity and the in-world follow-up, rendered with the open generation header (`<|im_start|>assistant\\n<think>\\n`),
             as Qwen saw it just before writing its reply. Read at every assistant header (the prefilled turns' and the open one):
             <|im_start|> and the 3 tokens after it ("assistant", "\\n", then "<think>" on the open header).
    reply    the generated replies of REPLY_ARMS in the thinking-off Sonnet run (first sample of 20 scenarios), at the last
             <|im_start|> and the 24 tokens after it: the reply's header, its empty think block and its first ~18 tokens.
The "Assistant token" of the workspace paper (Gurnee et al. 2026) is the role token after the final <|im_start|>.

analyze: a read hits a keyword set (CORE: fiction/roleplay/character/disclaimer/pretend/persona; BROAD: CORE plus acting and
story words; SELF_AI: AI/assistant/language model/...) if any of its 4 bullets matches. Rates are over reads (5 layers x the
header tokens) at the final header, by arm, per token, and at the earlier (prefilled) assistant headers; also the bullet
words most over-represented after drifted vs default-voice turns.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path[:0] = ["scripts", "."]
from runs import finish, new_run  # ty: ignore[unresolved-import]

CORE = re.compile(r"\b(?:fiction\w*|role[- ]?play\w*|characters?|in-character|disclaim\w*|pretend\w*|persona\w*)\b", re.IGNORECASE)
SELF_AI = re.compile(r"\b(?:AI|language model|assistant|chatbot|Qwen|Alibaba)\b", re.IGNORECASE)
KEYWORDS = re.compile(  # BROAD: CORE plus story / acting words
    r"\b(?:fiction\w*|role[- ]?play\w*|characters?|in-character|disclaim\w*|pretend\w*|persona\w*|acting|act as|story|stories|"
    r"storytell\w*|narrat\w*|imaginary|make-believe|performance|perform\w*)\b",
    re.IGNORECASE,
)
SONNET_RUN = Path("data/runs/character_drift_sonnet/2026-10-25_nothink")
REPLY_ARMS = ["drift_identity", "sysrp_identity", "default_identity", "drift_inworld"]


def build_requests(out: Path):
    os.environ.setdefault("QWEN_VLLM_URL", "http://unused")  # evalcue.harness builds a client at import
    os.environ.setdefault("QWEN_VLLM_API_KEY", "unused")
    from evalcue.envs.character_drift import sonnet, wildchat

    reqs = []
    for env, mod, prefills in (("sonnet", sonnet, sonnet.PREFILLS), ("wildchat", wildchat, wildchat.PREFILLS)):
        for item in mod.items():
            for p in prefills:
                for kind in ("identity", "inworld"):
                    ep = mod.build(f"{p}_{kind}", item)
                    reqs.append(
                        {
                            "id": f"prefix/{env}/{p}_{kind}/{item}",
                            "env": env,
                            "arm": f"{p}_{kind}",
                            "messages": ep.messages,
                            "generation_prompt": True,
                            "turn_starts": 3,
                            "turn_role": "assistant",
                            "shard_key": f"{env}/{p}/{item}",
                            "priority": 0 if kind == "identity" else 1,
                        }
                    )
    rows = [json.loads(line) for line in (SONNET_RUN / "rows.jsonl").open()]
    for r in rows:
        if r["arm"] in REPLY_ARMS and r["sample"] < 20 and "error" not in r:
            reqs.append(
                {
                    "id": f"reply/sonnet/{r['arm']}/{r['item']}",
                    "env": "sonnet-reply",
                    "arm": r["arm"],
                    "messages": r["messages"],
                    "turn_starts": 24,
                    "last_turn_only": True,
                    "priority": 2,
                    "source": f"{SONNET_RUN.parent.name}/{SONNET_RUN.name}:{rows.index(r)}",
                }
            )
    out.mkdir(parents=True, exist_ok=True)
    (out / "requests.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reqs))
    print(f"{out / 'requests.jsonl'}: {len(reqs)} requests")


def load_reads(out: Path):
    reads = [json.loads(line) for line in (out / "reads.jsonl").open()]
    by = defaultdict(list)
    for r in reads:
        by[r["id"]].append(r)
    return by


def header_reads(rs):
    """-> {(header #, offset): [reads]} for the assistant headers: offset 0 = <|im_start|>, 1 = "assistant", 2 = "\\n", ..."""
    starts = sorted({r["token_index"] for r in rs if r["token"] == "<|im_start|>"})
    out = defaultdict(list)
    for r in rs:
        s = max((x for x in starts if x <= r["token_index"]), default=None)
        if s is None:
            continue
        out[starts.index(s), r["token_index"] - s].append(r)
    roles = {}
    for (h, off), lst in out.items():
        if off == 1:
            roles[h] = lst[0]["token"]
    return {k: v for k, v in out.items() if roles.get(k[0]) == "assistant"}, len(starts)


def hit(r, pat=None):
    return any((pat or KEYWORDS).search(b["text"]) for b in r["bullets"])


def rate(c):
    return f"{c[0] / c[1]:4.0%} ({c[0]}/{c[1]})" if c[1] else "-"


def analyze(out: Path):
    """Prints the tables in docs/envs/character_drift.md and writes OUT/analysis.json."""
    by = load_reads(out)
    sets = {"core": CORE, "broad": KEYWORDS, "ai": SELF_AI}
    t = defaultdict(lambda: [0, 0])  # (env, arm, where, offset or "all", set) -> [hit reads, reads]
    words = defaultdict(Counter)  # prefill -> bullet doc-frequency at the final header (sonnet only, same scenarios)
    examples = defaultdict(list)
    for tid, rs in by.items():
        kind, env, arm, _ = tid.split("/", 3)
        heads, _ = header_reads(rs)
        if not heads:
            continue
        last = max(h for h, _ in heads)
        think_end = min((off for (h, off), lst in heads.items() if h == last and lst[0]["token"] == "</think>"), default=99)
        for (h, off), lst in heads.items():
            if kind == "reply":
                where, pos = (
                    ("earlier", off) if h != last else ("reply", "header" if off <= 3 else "think" if off <= think_end + 1 else "body")
                )
            else:
                where, pos = ("final" if h == last else "earlier"), off
            for r in lst:
                for name, pat in sets.items():
                    k = hit(r, pat)
                    for key in ((env, arm, where, pos, name), (env, arm, where, "all", name)):
                        t[key][0] += k
                        t[key][1] += 1
                if where == "final" and env == "sonnet":
                    for bl in r["bullets"]:
                        words[arm.split("_")[0]].update(set(re.findall(r"[a-z']{3,}", bl["text"].lower())))
                if where == "final" and off == 1 and hit(r, CORE) and len(examples[env, arm]) < 4:
                    examples[env, arm].append((r["layer"], [bl["text"] for bl in r["bullets"]]))
    arms = sorted({(e, a) for e, a, w, _, _ in t if w == "final"})
    print("final assistant header, share of reads (5 layers x tokens) with a bullet matching CORE / BROAD / SELF_AI;")
    print("CORE by token: <|im_start|>, 'assistant' (the paper's Assistant token), '\\n', '<think>'; CORE at earlier (prefilled) headers")
    summary = {}
    for e, a in arms:
        f = lambda w, pos, n, e=e, a=a: t[e, a, w, pos, n]
        cells = (
            [rate(f("final", "all", n)) for n in sets]
            + [rate(f("final", o, "core")) for o in range(4)]
            + [rate(f("earlier", "all", "core"))]
        )
        summary[f"{e}/{a}"] = {k: f("final", "all", n) for k, n in zip(("core", "broad", "ai"), sets, strict=True)}
        summary[f"{e}/{a}"] |= {f"core_tok{o}": f("final", o, "core") for o in range(4)} | {"core_earlier": f("earlier", "all", "core")}
        print(f"| {e} | `{a}` | " + " | ".join(cells) + " |")
    print("\nreplies (sonnet, thinking off), CORE / SELF_AI: header (<|im_start|> + 3) | empty think block | first ~18 reply tokens")
    for a in REPLY_ARMS:
        cells = [
            f"{rate(t['sonnet-reply', a, 'reply', p, 'core'])} / {rate(t['sonnet-reply', a, 'reply', p, 'ai'])}"
            for p in ("header", "think", "body")
        ]
        summary[f"reply/{a}"] = {p: {n: t["sonnet-reply", a, "reply", p, n] for n in ("core", "ai")} for p in ("header", "think", "body")}
        print(f"| `{a}` | " + " | ".join(cells) + " |")
    print("\nbullet words at the final header, by prefill (sonnet; doc frequency per read, words in >= 20 reads)")
    n_reads = {p: sum(t["sonnet", f"{p}_{f}", "final", "all", "core"][1] for f in ("identity", "inworld")) for p in words}
    for p in ("drift", "qwensys", "sysrp"):
        d, c = words[p], words["default"]
        ratio = sorted((((d[w] + 1) / n_reads[p]) / ((c[w] + 1) / n_reads["default"]), w) for w in d if d[w] >= 20)
        print(f"  {p} >> default:", ", ".join(f"{w} {d[w]}/{c[w]}" for _, w in ratio[::-1][:25]))
        if p == "drift":
            print("  default >> drift:", ", ".join(f"{w} {c[w]}/{d[w]}" for _, w in ratio[:20]))
    print("\nexample CORE hits at the Assistant token")
    for (e, a), ex in sorted(examples.items()):
        for L, bs in ex[:2]:
            print(f"  {e} {a} L{L}: {' | '.join(bs)[:260]}")
    (out / "analysis.json").write_text(json.dumps(summary, indent=1) + "\n")


def expview(out: Path):
    """Token-read pages: one item per transcript, reads at every read position (O-lens bullets with FVE)."""
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(os.environ.get("EXPVIEW", Path.home() / "workspace/expview")).expanduser()))
    from expview import Run, call  # ty: ignore[unresolved-import]

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    reqs = {r["id"]: r for r in map(json.loads, (out / "requests.jsonl").open())}
    by = load_reads(out)
    name = f"character_drift_olens_{out.name}"
    dest = Path(os.environ.get("EXPVIEW", Path.home() / "workspace/expview")).expanduser() / "runs" / name
    desc = (
        "O-lens (agu18dec `olens_s3d_space_rl600`, greedy, 4 bullets, whitened FVE) on character_drift transcripts. "
        "`prefix/...` items: the conversation as Qwen saw it before its reply (open `<|im_start|>assistant\\n<think>\\n` header); "
        "reads at each assistant `<|im_start|>` and the 3 tokens after it. `reply/...` items: a generated thinking-off reply "
        "with reads at its "
        "header and first ~20 tokens. Label `kw_final` = any read at the final header has a bullet matching "
        f"`{KEYWORDS.pattern}`. Run dir olens_reads/{out.name} in the sprint HF dataset."
    )
    run = Run(str(dest), title="Character drift: O-lens reads", description=desc, tokens_section="O-lens reads")
    run.reset()
    for tid, rs in sorted(by.items()):
        req = reqs[tid]
        text = tok.apply_chat_template(req["messages"], tokenize=False, add_generation_prompt=bool(req.get("generation_prompt")))
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
        heads, _ = header_reads(rs)
        last = max((h for h, _ in heads), default=None)
        kw = any(hit(r) for (h, _), lst in heads.items() if h == last for r in lst)
        kind, env, arm, _ = tid.split("/", 3)
        tokens = {"tokens": toks, "sections": [["transcript", 0, len(toks)]], "reads": reads, "legend": "Inline: L40's first bullet."}
        msgs = req["messages"]
        output = msgs[-1] if msgs[-1]["role"] == "assistant" else None
        calls = [call("qwen (as read)", "Qwen/Qwen3.6-27B", messages=msgs[:-1] if output else msgs, output=output)]
        labels = {"kind": kind, "env": env, "prefill": arm.split("_")[0], "followup": arm.split("_")[1], "kw_final": kw}
        run.add(id=tid, group=f"{kind}/{env}/{arm}", labels=labels, calls=calls, tokens=tokens)
    import subprocess

    subprocess.run([sys.executable, str(dest.parent.parent / "expview.py"), "build", str(dest), "-o", str(dest / "page.html")], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["requests", "analyze", "expview", "newrun", "finish"])
    ap.add_argument("out", type=Path, nargs="?")
    a = ap.parse_args()
    if a.cmd == "newrun":
        run = new_run(
            "olens_reads",
            "character-drift",
            description="O-lens reads at the Assistant token of character_drift prefixes (drift vs default voice) + reply starts",
            models=[{"name": "agu18dec/olens_and_ar olens_s3d_space_rl600 on Qwen/Qwen3.6-27B", "served_by": "sky RunPod 2x H200"}],
            params={"layers": [24, 32, 40, 48, 56], "turn_starts": {"prefix": 3, "reply": 24}, "sampling": "greedy, 4 bullets"},
        )
        print(run)
    elif a.cmd == "finish":
        finish(a.out)
    else:
        {"requests": build_requests, "analyze": analyze, "expview": expview}[a.cmd](a.out)
