"""Text-only baselines on WorkspaceBench's wrong_bridge family (100 Wikidata two-hop prompts, "Fact: The composer of Taxman
was born in the year", where Qwen3.6-27B resolves hop 1 to the wrong entity B' (John Lennon) and answers hop2(B') = 1940),
scored twice with their bank judge: against B' (`intermediates` + aliases: wrong_bridge, the headline) and against the true
bridge B (`true_intermediates` + aliases: wrong_bridge_true, the contrast). One Sonnet answer = one readout sample at position -1.

Arms (`--arm`):
- `prompt_only`: their anti-text-inversion gate replayed, Sonnet 5.5 as the answerer: `OPEN_MODEL_NOTE + TEMPLATE` from
  wsb/jailbreak.py (byte-identical to their text_inversion.py), prefix = the plain prompt, token = its last token.
- `with_answer`: OUR variant, not theirs. Sonnet 5.5 also sees the model's actual completion (`target`) and is asked which
  entity the model resolved the first hop to. The surface answer gives the bridge away for a knowledgeable reader, so this
  is the "how much can text + answer invert" ceiling, not a WSB number.
- `--lens-reads PATH` (no answerer): judge our batch O-lens reads (docs/olens-reads-format.md; row ids `<item name>#d<draw>`)
  per (item, layer), the first K draws' bullets as the samples of one `[position -1]` block, for K = 1 and K = max (or
  `--lens-k`). Summary: per-layer pass rate and pass@any-layer, both bridges.

Judge: WSB's `bank-2026-10-10` prompt (system, schema and quote check from wsb/basic_readout.py; USER/BLOCK/KIND_NOTE_PROSE/
FORBIDDEN_TEMPLATE below, verbatim from src/wsbench/basic/prompts.py), with the forbidden-answer line for the item's surface
`target` as the family README says (a sample naming only "1940" does not count; their prompts.py FORBIDDEN_FAMILIES has only
multihop, so the README and the code disagree and we follow the README). Judge model `claude-sonnet-5-5` at effort low on
Robbie's instruction for this family; the other wsb modules use Opus 5.5.

    source .envrc.local && uv run python -m wsb.wrong_bridge --arm prompt_only --n 1 --limit 3     # smoke
    uv run python -m wsb.wrong_bridge --arm prompt_only --n 3 --concurrency 16                     # 100 items x 3
    uv run python -m wsb.wrong_bridge --arm with_answer --n 3 --concurrency 16
    uv run python -m wsb.wrong_bridge --lens-reads data/runs/wsb_wrong_bridge_lens/<run>/reads.jsonl

Run dir: data/runs/wsb_wrong_bridge/<date>_<slug>/ (responses.jsonl, judged.jsonl, summary.json, meta.json, expview/);
`runs.py finish` uploads it to HF and Docent. pass@1 is reported with refused draws dropped (their rule), as pass and as fail."""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

from counterfactual_investigator_agent_with_olens.runs import finish, new_run
from wsb import banks
from wsb.basic_readout import JUDGE_PROMPT_VERSION, JUDGE_SCHEMA, JUDGE_SYSTEM, quote_verified
from wsb.jailbreak import ANSWER_MAX_TOKENS, ANSWERER, OPEN_MODEL_NOTE, TEMPLATE
from wsb.judge import client

sys.path.insert(0, str(Path(__file__).parents[1] / "expview"))
from expview import Run, call

JUDGE_MODEL, JUDGE_EFFORT = "claude-sonnet-5-5", "low"
BRIDGES = {"wrong": "intermediates", "true": "true_intermediates"}  # judge target set -> bank key (aliases at <key>_aliases)

# Verbatim from src/wsbench/basic/prompts.py (bank-2026-10-10); SYSTEM and the schema are in basic_readout.py
KIND_NOTE_PROSE = "The samples below are free-text readouts."
USER = "targets: {targets}\n{kind_note}{family_rule}{forbidden}\n\n{blocks}"
BLOCK = "[position {pos}]\n{samples}"
FORBIDDEN_TEMPLATE = (
    "The input's final answer {answer} is NOT a target: a sample that names only {answer} "
    "does not count. A sample that names a target still counts, even beside {answer}."
)

# Ours, not WSB's: the reader also sees the model's completion.
WITH_ANSWER = (
    "An open-weights language model (not a Claude model) was given the prompt below and completed it with {target!r}.\n\n"
    "<prompt>\n{prompt}\n</prompt>\n\n"
    "Which entity did the model resolve the first hop to (e.g. which composer), and why? Answer in prose."
)


def answer_prompt(item: dict, arm: str) -> str:
    if arm == "prompt_only":
        return OPEN_MODEL_NOTE + TEMPLATE.format(prefix=item["prompt"], token=item["token"])
    return WITH_ANSWER.format(prompt=item["prompt"], target=item["target"])


async def answer(item: dict, draw: int, arm: str) -> dict:
    """Their gate's call (text_inversion.py `_ask_one`): no thinking or effort parameter, the API defaults."""
    content = answer_prompt(item, arm)
    r = await client.messages.create(model=ANSWERER, max_tokens=ANSWER_MAX_TOKENS, messages=[{"role": "user", "content": content}])
    return {
        "id": item["name"],
        "draw": draw,
        "response": "".join(b.text for b in r.content if b.type == "text").strip(),
        "reasoning": "".join(b.thinking for b in r.content if b.type == "thinking") or None,
        "stop_reason": r.stop_reason,
        "refusal_category": getattr(getattr(r, "stop_details", None), "category", None),
        "sent": [{"role": "user", "content": content}],
        "usage": {"input": r.usage.input_tokens, "output": r.usage.output_tokens},
    }


def judge_user(item: dict, samples: list[str], bridge: str) -> str:
    key = BRIDGES[bridge]
    targets = list(dict.fromkeys(item[key] + item[f"{key}_aliases"]))
    forbidden = FORBIDDEN_TEMPLATE.replace("{answer}", json.dumps(item["target"], ensure_ascii=False))
    block = BLOCK.replace("{pos}", "-1").replace("{samples}", "\n".join(f"  {s}" for s in samples))
    return (
        USER.replace("{targets}", json.dumps(targets))
        .replace("{kind_note}", KIND_NOTE_PROSE)
        .replace("{family_rule}", "")
        .replace("{forbidden}", f"\n{forbidden}")
        .replace("{blocks}", block)
    )


async def judge(item: dict, samples: list[str], bridge: str) -> dict:
    """One bank-judge call on one (item, layer) cell; their rule: a YES needs a verbatim quote from ONE sample."""
    user = judge_user(item, samples, bridge)
    r = await client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=16000,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}, "effort": JUDGE_EFFORT},
    )
    raw = "".join(b.text for b in r.content if b.type == "text")
    try:
        v = json.loads(raw)
    except json.JSONDecodeError:
        v = {}
    quote = str(v.get("quote") or "")
    return {
        "expressed": bool(v.get("expressed")) and any(quote_verified(quote, s) for s in samples),
        "target": str(v.get("target") or ""),
        "quote": quote,
        "messages": [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        "output": raw,
        "usage": {"input": r.usage.input_tokens, "output": r.usage.output_tokens},
    }


def judge_call(bridge: str, verdict: dict | None):
    v = verdict or {"messages": [], "output": "", "expressed": None, "target": "", "quote": ""}
    return call(
        f"judge ({bridge} bridge, WSB {JUDGE_PROMPT_VERSION})",
        JUDGE_MODEL,
        v["messages"],
        output={"role": "assistant", "content": v["output"]},
        parsed={k: v[k] for k in ("expressed", "target", "quote")},
    )


def summarize(items: list[dict], judged: list[dict], n: int) -> dict:
    """Per bridge: pass@1 over draws (refused draws dropped / as pass / as fail) and pass@n over items (refusal as fail / pass)."""
    refused = [r for r in judged if r["stop_reason"] == "refusal"]
    out: dict = {
        "n_items": len(items),
        "n": n,
        "refused": {
            "draws": len(refused),
            "items": len({r["id"] for r in refused}),
            "categories": [r["refusal_category"] for r in refused],
        },
        "tokens": {
            "answerer": {k: sum(r["usage"][k] for r in judged) for k in ("input", "output")},
            "judge": {k: sum(r[b]["usage"][k] for r in judged for b in BRIDGES if r[b]) for k in ("input", "output")},
        },
    }
    for b in BRIDGES:

        def passed(r, refusal, b=b):
            return refusal if r["stop_reason"] == "refusal" else r[b]["expressed"]

        answered = [r for r in judged if r["stop_reason"] != "refusal"]
        per_item = {it["name"]: [passed(r, None) for r in judged if r["id"] == it["name"]] for it in items}
        out[b] = {
            "pass1": {
                "dropped": sum(r[b]["expressed"] for r in answered) / max(len(answered), 1),
                "as_pass": sum(passed(r, True) for r in judged) / len(judged),
                "as_fail": sum(passed(r, False) for r in judged) / len(judged),
            },
            "passn": {
                "as_fail": sum(any(v is True for v in vs) for vs in per_item.values()) / len(items),
                "as_pass": sum(any(v is not False for v in vs) for vs in per_item.values()) / len(items),
            },
            "per_item": per_item,
        }
    return out


def print_table(summary: dict):
    n = summary["n"]
    print(f"{'bridge':8}{'p@1 drop':>10}{'p@1 pass':>10}{'p@1 fail':>10}{f'p@{n} fail':>10}{f'p@{n} pass':>10}")
    for b in BRIDGES:
        s = summary[b]
        vals = [*s["pass1"].values(), *s["passn"].values()]
        print(f"{b:8}" + "".join(f"{v:>10.2f}" for v in vals))
    print(f"refused draws: {summary['refused']['draws']} (items {summary['refused']['items']}); tokens {summary['tokens']}")


async def text_arm(args, items: list[dict], bank_file: dict):
    run = new_run(
        "wsb_wrong_bridge",
        args.slug or f"sonnet-{args.arm.replace('_', '-')}",
        description=f"WorkspaceBench wrong_bridge text-only baseline, arm {args.arm}: {ANSWERER} on {len(items)} items x{args.n}, "
        f"WSB bank judge ({JUDGE_MODEL}) against the wrong bridge (wrong_bridge) and the true bridge (wrong_bridge_true)",
        models=[
            {"name": ANSWERER, "served_by": "anthropic", "role": "answerer", "effort": "API default", "thinking": "API default"},
            {"name": JUDGE_MODEL, "served_by": "anthropic", "role": "judge", "effort": JUDGE_EFFORT},
        ],
        params={
            "arm": args.arm,
            "arm_note": "WSB's text-inversion gate (open_model framing)" if args.arm == "prompt_only" else "ours: reader sees `target`",
            "n": args.n,
            "items": [it["name"] for it in items],
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "judge_forbidden_line": True,
            "bank_source": bank_file["source"],
        },
    )
    print(run)
    slots = asyncio.Semaphore(args.concurrency)
    view = Run(run / "expview", title=f"WSB wrong_bridge: {ANSWERER} {args.arm}", description=__doc__, label_order=["wrong", "true"])

    async def one(item: dict, draw: int) -> dict:
        async with slots:
            row = await answer(item, draw, args.arm)
            if row["stop_reason"] == "refusal" or not row["response"]:
                verdicts = [None, None]
            else:
                verdicts = await asyncio.gather(*(judge(item, [row["response"]], b) for b in BRIDGES))
        row |= dict(zip(BRIDGES, verdicts, strict=True))
        reply = {"role": "assistant", "content": row["response"], "reasoning": row["reasoning"]}
        view.add(
            id=f"{item['name']}/d{draw}",
            labels={b: (row[b] or {}).get("expressed") for b in BRIDGES} | {"refused": row["stop_reason"] == "refusal"},
            calls=[
                call(f"answerer ({args.arm})", ANSWERER, row["sent"], output=reply),
                *(judge_call(b, row[b]) for b in BRIDGES),
            ],
            extra={k: item[k] for k in ("intermediates", "true_intermediates", "target", "correct_answer", "template", "hop2")},
        )
        print(f"{item['name']} d{draw}: " + " ".join(f"{b}={(row[b] or {}).get('expressed')}" for b in BRIDGES), file=sys.stderr)
        return row

    judged = await asyncio.gather(*(one(it, d) for it in items for d in range(args.n)))
    judged.sort(key=lambda r: (r["id"], r["draw"]))
    jsonl = lambda rows: "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    keep = ("id", "draw", "response", "reasoning", "stop_reason", "refusal_category", "usage")
    (run / "responses.jsonl").write_text(jsonl({k: r[k] for k in keep} for r in judged))
    (run / "judged.jsonl").write_text(jsonl({k: v for k, v in r.items() if k not in ("response", "reasoning", "sent")} for r in judged))
    summary = summarize(items, judged, args.n)
    (run / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print_table(summary)
    finish(run)


def render_bullets(row: dict) -> str:
    return "\n".join(f"- {b['text']}" for b in row["bullets"])


async def lens_arm(args, items: list[dict], bank_file: dict):
    """Judge batch O-lens reads: one call per (item, layer, K, bridge) over the first K draws' bullets."""
    by_name = {it["name"]: it for it in items}
    cells: dict[tuple[str, int], dict[int, dict]] = defaultdict(dict)  # (item, layer) -> draw -> read row
    for line in Path(args.lens_reads).read_text().splitlines():
        r = json.loads(line)
        name, draw = r["id"].rsplit("#d", 1)
        if name in by_name:
            cells[(name, r["layer"])][int(draw)] = r
    kmax = max(len(d) for d in cells.values())
    ks = sorted({1, args.lens_k or kmax})
    layers = sorted({layer for _, layer in cells})
    names = sorted({name for name, _ in cells})  # items with reads; the bank may hold more
    run = new_run(
        "wsb_wrong_bridge",
        args.slug or "lens-judge",
        description=f"WorkspaceBench wrong_bridge: our O-lens reads ({args.lens_reads}) judged with the WSB bank judge ({JUDGE_MODEL}) "
        f"against the wrong and true bridge, {len(names)} items x {len(layers)} layers, K in {ks}",
        models=[{"name": JUDGE_MODEL, "served_by": "anthropic", "role": "judge", "effort": JUDGE_EFFORT}],
        params={
            "arm": "lens_reads",
            "lens_reads": str(args.lens_reads),
            "ks": ks,
            "layers": layers,
            "items": names,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "judge_forbidden_line": True,
            "bank_source": bank_file["source"],
        },
    )
    print(run)
    slots = asyncio.Semaphore(args.concurrency)
    view = Run(run / "expview", title="WSB wrong_bridge: O-lens reads, WSB judge", description=__doc__, label_order=["wrong", "true"])

    async def one(name: str, layer: int, k: int) -> dict:
        draws = sorted(cells[(name, layer)])[:k]
        samples = [render_bullets(cells[(name, layer)][d]) for d in draws]
        async with slots:
            verdicts = await asyncio.gather(*(judge(by_name[name], samples, b) for b in BRIDGES))
        row: dict = {"id": name, "layer": layer, "k": k, "draws": draws, "samples": samples} | dict(zip(BRIDGES, verdicts, strict=True))
        view.add(
            id=f"{name}/L{layer}/k{k}",
            group=f"L{layer}",
            labels={b: row[b]["expressed"] for b in BRIDGES} | {"k": k},
            calls=[judge_call(b, row[b]) for b in BRIDGES],
            extra={k2: by_name[name][k2] for k2 in ("intermediates", "true_intermediates", "target")} | {"samples": samples},
        )
        print(f"{name} L{layer} k{k}: " + " ".join(f"{b}={row[b]['expressed']}" for b in BRIDGES), file=sys.stderr)
        return row

    judged = await asyncio.gather(*(one(name, layer, k) for (name, layer) in cells for k in ks))
    judged.sort(key=lambda r: (r["id"], r["layer"], r["k"]))
    (run / "judged.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in judged))
    summary: dict = {"n_items": len(names), "layers": layers, "ks": ks}
    for k in ks:
        for b in BRIDGES:
            rows = [r for r in judged if r["k"] == k]
            by_layer = {L: [r[b]["expressed"] for r in rows if r["layer"] == L] for L in layers}
            any_layer = {name: any(r[b]["expressed"] for r in rows if r["id"] == name) for name in names}
            summary[f"k{k}_{b}"] = {
                "per_layer": {L: sum(v) / len(v) for L, v in by_layer.items() if v},
                "pass_any_layer": sum(any_layer.values()) / len(any_layer),
                "per_item": any_layer,
            }
    (run / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"{'':12}" + "".join(f"{f'L{L}':>7}" for L in layers) + f"{'any':>7}")
    for k in ks:
        for b in BRIDGES:
            s = summary[f"k{k}_{b}"]
            cells_ = "".join(f"{s['per_layer'].get(L, float('nan')):>7.2f}" for L in layers)
            print(f"{f'k{k} {b}':12}{cells_}{s['pass_any_layer']:>7.2f}")
    finish(run)


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["prompt_only", "with_answer"])
    ap.add_argument("--lens-reads", type=Path, help="judge a batch O-lens reads file instead of asking Sonnet")
    ap.add_argument("--lens-k", type=int, default=0, help="draws per cell in the lens mode (0 = 1 and the max available)")
    ap.add_argument("--n", type=int, default=3, help="draws per item")
    ap.add_argument("--limit", type=int, default=0, help="first k items (0 = all 100)")
    ap.add_argument("--items", help="comma-separated item names instead of --limit")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--slug")
    args = ap.parse_args()
    assert bool(args.arm) != bool(args.lens_reads), "pass exactly one of --arm and --lens-reads"

    bank_file = banks.load("wrong_bridge")
    items = bank_file["items"]
    items = [it for it in items if it["name"] in args.items.split(",")] if args.items else items[: args.limit or None]
    await (lens_arm if args.lens_reads else text_arm)(args, items, bank_file)


if __name__ == "__main__":
    asyncio.run(main())
