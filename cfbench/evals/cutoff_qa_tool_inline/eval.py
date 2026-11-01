"""cutoff_qa_tool_inline: cutoff_qa_inline with Qwen answering through a submit_answer tool call, so the end-of-prompt reads
are no longer next-token reads (Robbie, 2026-10-29; scaled to bank v3 2026-10-30).

Qwen sees each question with a `submit_answer(answer)` tool (scripts/cutoff_gate.py SUBMIT_TOOL): the chat template puts
the tool definition in a system block before the question, and the reply opens with the tool-call wrapper, so in Qwen's
native format the answer is 15 tokens past the last prompt token (`<tool_call>\n<function=submit_answer>\n<parameter=answer>\n`)
instead of the next one. Labels come from GATE (tool_choice "required"): Qwen's modal
tool-call answer, stable = >= 7/8. Cases: bank v3's stable non-numeric items (no-tool gate) that are also stable under GATE,
in bank order. AUTO_GATE (tool_choice "auto", same questions) gives each item's unforced tool-call rate, for the
always-tool split. No env run: the gate's prompts are the transcripts.

The investigator sees what Qwen saw except the reply: the system block (the tool definition, as the template renders it)
and the question. Read positions (Robbie, 2026-10-30, the standard set): O-lens and NLA at every punctuation token of the
user turn plus the last 10 prompt tokens (the question's ".", <|im_end|>, \n, <|im_start|>, assistant, \n, then the empty think
block <think> \n\n </think> \n\n), layers 32-60, never the system block; J++ at every token from the user turn's <|im_start|> to
the end of the prompt, top 3 (cutoff_qa_inline's jlens block). Each O-lens / NLA read is shown after the ~60 characters of
this item's prompt before it. Placebos: O-lens and NLA, another item's reads mapped onto this item's positions (the last 10
one to one, punctuation in order, cycling), shown under this item's text, so a placebo block looks like a real one; J++,
cutoff_qa_inline's (a hint_mcq transcript's last n tokens, pool from POOLS). Claude Haiku 5.5 at effort medium.

The 2026-10-29 run (`cfbench/2026-10-29_cutoff-qa-tool-haiku-200`: first 200 stable v2 items, question only, no system
block shown, O-lens conditions only) used this file at that day's commit.

    uv run --with transformers --with jinja2 python -m cfbench.evals.cutoff_qa_tool_inline.eval DIR   # DIR/{tail,user}.jsonl
    uv run python -m cfbench.eval cutoff_qa_tool_inline --prep
    uv run python -m cfbench.eval cutoff_qa_tool_inline --slug cutoff-qa-tool-v3 --predictors anthropic/claude-haiku-5.5 --draws 3 \
        --conditions transcript_inline,olens_inline,olens_placebo_inline,jlens_inline,jlens_placebo_inline,nla_inline,nla_placebo_inline
    uv run python -m cfbench.evals.cutoff_qa_inline.report data/runs/cfbench/<date>_cutoff-qa-tool-v3
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from cfbench.core import Rollout
from cfbench.evals.cutoff_qa_inline import eval as base
from cfbench.evals.cutoff_qa_inline import inline
from cfbench.evals.cutoff_qa_inline.eval import Prompt
from cfbench.olens import show_token
from environments._lib.core import load_env

ERROR_RATE = True  # the env viewer shows this eval as error rate, lower is better (Kylie, 2026-10-30)
ENV = "cutoff_qa"
ENVS = [ENV]
BANK = "v3"
GATE = "cutoff_gate/2026-10-30_v3-tool-required"  # labels and stability
AUTO_GATE = "cutoff_gate/2026-10-30_v3-tool-auto"  # unforced tool-call rate per item
READS = Path("data/runs/olens_reads/2026-10-30_cutoff-v3tool-punct/reads.jsonl")  # O-lens: user-turn punctuation + last 10
NLA_READS = Path("data/runs/nla_reads/2026-10-30_cutoff-v3tool-punct/reads.jsonl")  # NLA, same positions
JREADS = Path("data/runs/jpp_reads/2026-10-30_cutoff-v3tool-user/reads.jsonl")  # J++, user turn to end of prompt
POOLS = Path("data/runs/cfbench_evals/2026-10-29_cutoff-qa-inline-inputs-1k/reads")  # the J++ placebo pool (jpool.jsonl)
QWEN = "Qwen/Qwen3.6-27B"
MODEL_NAME = "the model"
TASK = "answer"
CONDITIONS = ("transcript_inline", "olens_inline", "olens_placebo_inline")
INLINE_EFFORT = "medium"
SYSTEM = inline.SYSTEM
VERSIONS: dict[str, dict] = {"nohint": {"arm": "nohint"}}
ARM_LABELS = {"nohint": "the question, verbatim (asked with a submit_answer tool)"}
PAIR_LABELS = {"nohint -> nohint": "no counterfactual"}


def _run(name: str) -> Path:
    return Path("data/runs") / name


def tool() -> dict:
    return json.loads((_run(GATE) / "meta.json").read_text())["params"]["tools"][0]


def collect() -> list[dict]:
    """Bank v3's stable non-numeric items also stable under GATE, in bank order, labelled with GATE's modal answer."""
    mod = load_env(ENV)
    gate = {q["id"]: q for q in json.loads((_run(GATE) / "summary.json").read_text())["questions"]}
    auto = {r["id"]: sum(r["tool_called"][:8]) for r in map(json.loads, (_run(AUTO_GATE) / "samples.jsonl").read_text().splitlines())}
    out = []
    for it in mod.bank(BANK)["items"]:
        g = gate.get(it["id"])
        if g is None or not g["class"].startswith("stable"):
            continue
        meta = it | {"qwen_modal": g["modal"], "qwen_modal_count": g["modal_count"], "gate_class": g["class"], "gate_run": GATE,
                     "nohint_modal": it["qwen_modal"], "auto_tool_calls": auto[it["id"]]}  # fmt: skip
        modal = mod.gate_label(GATE, it["id"], g["modal"])
        out.append({"version": "nohint", "item": it["id"], "modal": modal, "source": f"cutoff_qa/v3-tool:{it['id']}",
                    "messages": [{"role": "user", "content": it["question"]}], "meta": meta})  # fmt: skip
    return out


PUNCT = re.compile(r"^\s*[.,;:!?]+[\"')\]*]*\s*$")  # the reader's stop tokens (scripts/olens_read_requests.py, transcript-viewer)
TAIL = [".", "<|im_end|>", "\n", "<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n"]  # the last 10 tokens


def requests(out: Path):
    """Read requests on the tool render: tail.jsonl (O-lens and NLA: every punctuation token of the user turn, then the last 10
    tokens: the question's ".", <|im_end|>, \n, <|im_start|>, assistant, \n and the empty think block; Robbie, 2026-10-30, the
    standard set for every eval) and user.jsonl (J++: every token from the user turn's <|im_start|> to the end of the prompt)."""
    from transformers import AutoTokenizer  # uv run --with transformers --with jinja2

    tok, tools = AutoTokenizer.from_pretrained(QWEN), [tool()]
    start = tok.convert_tokens_to_ids("<|im_start|>")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "tail.jsonl").open("w") as tail, (out / "user.jsonl").open("w") as user:
        for c in collect():
            text = tok.apply_chat_template(c["messages"], tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(text)["input_ids"]
            starts = [i for i, x in enumerate(ids) if x == start]
            assert len(starts) == 3 and tok.decode(ids[starts[1] + 1 : starts[1] + 2]) == "user", c["item"]  # system, user, assistant
            assert [tok.decode([i]) for i in ids[-9:]] == TAIL[1:], c["item"]  # ids[-10] ends the question: "." or ")." etc.
            punct = [i for i in range(starts[1], len(ids) - 10) if PUNCT.match(tok.decode([ids[i]]))]
            pos = punct + list(range(len(ids) - 10, len(ids)))
            # the exact thinking-off render as a raw prompt: a messages row is rendered by the reader with thinking on, whose
            # generation prompt has no empty think block (2026-10-30: the think-block positions fell off the end, KeyError)
            row = {"id": c["source"], "item": c["item"], "version": "nohint", "text": text, "priority": 0}
            tail.write(json.dumps(row | {"positions": pos}, ensure_ascii=False) + "\n")
            user.write(json.dumps(row | {"positions": list(range(starts[1], len(ids)))}, ensure_ascii=False) + "\n")
    print("wrote", out / "tail.jsonl", out / "user.jsonl")


def system_block() -> str:
    """The system turn's text exactly as Qwen's template renders it for this tool (the same for every item)."""
    from transformers import AutoTokenizer  # uv run --with transformers --with jinja2

    text = AutoTokenizer.from_pretrained(QWEN).apply_chat_template([{"role": "user", "content": "Q"}], tools=[tool()], tokenize=False)
    return text.split("<|im_start|>system\n", 1)[1].split("<|im_end|>", 1)[0]


def prep(out: Path, versions: list[str], placebo: Path, n_items: int | None = None, strict: bool = False):
    """cases.jsonl, system.txt and reads/: the items' O-lens, NLA and J++ reads plus the J++ placebo pool from POOLS."""
    (out / "reads").mkdir(parents=True)
    for src, dst in ((READS, "reads"), (NLA_READS, "nla"), (JREADS, "jreads")):
        rows = src.read_text() if src.exists() else ""
        if dst == "jreads" and src.exists():
            rows = "".join(json.dumps(r, ensure_ascii=False) + "\n" for rs in base.jpp_rows(src).values() for r in rs)
        (out / f"reads/{dst}.jsonl").write_text(rows)
        print(dst, "copied" if src.exists() else f"missing ({src})")
    (out / "reads/jpool.jsonl").write_text((POOLS / "jpool.jsonl").read_text())
    (out / "system.txt").write_text(system_block())
    cs = collect()[: n_items or None]
    (out / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cs))
    print(len(cs), "cases")


def reads(inputs: Path) -> list[str]:
    return [str(p) for p in [inputs / "reads/reads.jsonl"] if p.exists()]


# cutoff_qa_inline's reads texts with the new positions: every punctuation token of the question, then the last 10 tokens
OLD_WHERE = (
    "at the end of the prompt above: the period that ends its last line, the end-of-turn token and the newline after it, and "
    'the start of {model}\'s own turn (<|im_start|>, "assistant" and the newline), just before it answers,'
)
NEW_WHERE = (
    "at every punctuation mark in the user's question above, then at the end of the prompt: the period that ends the question, "
    'the end-of-turn token and the newline after it, the start of {model}\'s own turn (<|im_start|>, "assistant" and the '
    "newline) and the empty thinking block that follows (<think>, a blank line, </think>, a blank line), the last tokens "
    "before it answers,"
)
OLD_SHOWN, NEW_SHOWN = (
    "shows the end of the prompt with the token read wrapped",
    "shows the text just before it with the token read wrapped",
)
# read-position ablation (Kylie, 2026-10-30): the question's punctuation only (its final "." included), or that plus <|im_end|>;
# inputs whose reads were cut to those positions (cut_reads), the block's preamble naming only them
QPUNCT_WHERE = "at every punctuation mark in the user's question above, including the period that ends it,"
QPUNCT_IMEND_WHERE = QPUNCT_WHERE + " and at the end-of-turn token after it,"
TEXTS, NTAIL = {}, {"tail10": 10, "qpunct": 1, "qpunct_imend": 2}  # positions at the end mapped one to one by the placebo
for _kind, _text in (("olens", inline.OLENS_TEXT), ("nla", inline.NLA_TEXT)):
    _text = _text.replace("{where},", OLD_WHERE)  # cutoff_qa_inline's O-lens text names its positions as {where} since PR #439
    assert OLD_WHERE in _text and OLD_SHOWN in _text, _kind
    _text = _text.replace(OLD_SHOWN, NEW_SHOWN)
    for _cut, _where in (("tail10", NEW_WHERE), ("qpunct", QPUNCT_WHERE), ("qpunct_imend", QPUNCT_IMEND_WHERE)):
        TEXTS[_kind, _cut] = _text.replace(OLD_WHERE, _where)


def cut_of(rows: list[dict]) -> str:
    """Which positions an item's reads cover: the full set (last 10 included), or cut_reads' question punctuation (+ <|im_end|>)."""
    toks = {r["token"] for r in rows}
    if "</think>" in toks:
        return "tail10"
    return "qpunct_imend" if "<|im_end|>" in toks else "qpunct"


def cut_reads(src: Path, dst: Path, cut: str):
    """Keep each item's reads up to the question's final "." (qpunct) or <|im_end|> (qpunct_imend): drop the last 9 or 8 of the 10."""
    by: dict[str, list[dict]] = defaultdict(list)
    rows = [json.loads(line) for line in src.read_text().splitlines()]
    for r in rows:
        by[r["id"]].append(r)
    keep = {s: set(sorted({r["token_index"] for r in rs})[: -10 + NTAIL[cut]]) for s, rs in by.items()}
    out = [r for r in rows if r["token_index"] in keep[r["id"]]]
    assert {cut_of([r for r in rs if r["token_index"] in keep[s]]) for s, rs in by.items()} == {cut}
    dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(cut, len(rows), "->", len(out), "rows")


def _by_pos(rows: list[dict]) -> dict[int, dict[int, dict]]:
    out: dict[int, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        out[r["token_index"]][r["layer"]] = r
    return dict(sorted(out.items()))


def _align(own: list[int], other: list[int], n: int = 10) -> dict[int, int]:
    """Map this item's read positions onto another item's: the last n (10, or a cut's 1 or 2) one to one, the punctuation
    before them in order (cycling)."""
    mine, theirs = own[:-n], other[:-n] or other[-n : -n + 1]  # no punctuation: the question's final "."
    return {p: theirs[k % len(theirs)] for k, p in enumerate(mine)} | dict(zip(own[-n:], other[-n:], strict=True))


def _context(row: dict) -> str:
    """~60 characters of this item's prompt before the read token (the row's 32-token prefix), from the user turn on."""
    tok, prefix = row["token"], row["prefix"].split("<|im_start|>user\n")[-1]
    before = prefix.removesuffix(tok)
    return "…" + before[-60:].replace("\n", "↵") + f"<read>{show_token(tok)}</read>"


def reads_block(kind: str, own: list[dict], shown: list[dict]) -> str:
    """Per position of `own` (this item's reads, for the context), the bullets / explanation of `shown` at the aligned position."""
    mine, theirs = _by_pos(own), _by_pos(shown)
    assert cut_of(own) == cut_of(shown)
    where = _align(list(mine), list(theirs), NTAIL[cut_of(own)])
    out = []
    for p, layers in mine.items():
        out.append(f"At {json.dumps(next(iter(layers.values()))['token'], ensure_ascii=False)}: {_context(next(iter(layers.values())))}")
        for layer, r in sorted(theirs[where[p]].items()):
            if kind == "olens":
                out += [
                    f"- layer {layer}, FVE {b['fve']:.2f}: {' '.join(b['text'].split())}"
                    for b in sorted(r["bullets"], key=lambda b: -b["fve"])
                ]
            else:
                fve = f", FVE {r['fve']:.2f}" if r.get("fve") is not None else ""
                lines = [" ".join(line.split()) for line in r["text"].splitlines() if line.strip()]
                out.append(f"- layer {layer}{fve}:\n" + "\n".join(f"  {line}" for line in lines))
        out.append("")
    return "\n".join(out).strip()


def INLINE(env, condition: str, case: dict, base_runs: list, seed: int) -> tuple[str, dict]:
    """The condition's prompt with the system turn Qwen saw shown above the question; O-lens and NLA blocks over this eval's
    positions (reads_block), J++ and transcript-only as cutoff_qa_inline builds them."""
    question, extra = base_runs[0].messages[0]["content"], {}
    if condition.startswith(("olens", "nla")):
        kind = "olens" if condition.startswith("olens") else "nla"
        by_source = inline.reads_by_source(Path(case["inputs"]) / f"reads/{'reads' if kind == 'olens' else 'nla'}.jsonl")
        own = shown = by_source[base_runs[0].source]
        if condition.endswith("placebo_inline"):  # another item's reads, seeded per (draw, case), never its own
            src = random.Random(f"{seed}:{case['version']}:{case['item']}").choice(sorted(set(by_source) - {base_runs[0].source}))
            shown, extra = by_source[src], {"placebo_sources": [src]}
        layers = ", ".join(map(str, sorted({r["layer"] for r in own})))
        reads = TEXTS[kind, cut_of(own)].format(model=MODEL_NAME, layers=layers, block=reads_block(kind, own, shown))
        prompt = inline.USER.format(told="", prompt=question, reads=reads)
    else:
        prompt, extra = base.INLINE(env, condition, case, base_runs, seed)
    system = (Path(case["inputs"]) / "system.txt").read_text()
    head = '```\nUser: "'
    assert prompt.count(head) == 1
    if condition.startswith("jlens"):  # J++ read the user turn onward, not the system turn now shown above it
        old = "every token of the prompt above as"
        assert old in prompt
        prompt = prompt.replace(old, "every token of the user's turn above (not the system turn) as")
    return prompt.replace(head, f'```\nSystem: "{system}"\n\nUser: "'), extra


_FULL = ("question punct.\n+ last 10 tokens", "olens_inline", "olens_placebo_inline")
POSITION_CHART = {  # the env viewer's read-position bars (page.html positionChart; docs/cutoff-qa-inline.md, Kylie 2026-10-30)
    "title": "O-lens read positions",
    "caption": "Does the gain need the last 10 tokens? The question's punctuation is read before the assistant turn begins.",
    "subsets": [["", "all"], ["stable_wrong", "stale only"]],
    "panels": [
        {
            "title": "question punctuation vs the full set",
            "groups": [
                ["question\npunctuation", "olens_qpunct_inline", "olens_qpunct_placebo_inline"],
                ["question punct.\n+ <|im_end|>", "olens_qpunct_imend_inline", "olens_qpunct_imend_placebo_inline"],
                list(_FULL),
            ],
        },
    ],
}


def ITEM_INFO(row: dict) -> dict:
    """The env viewer's question table: the question, Qwen's modal answer and the class (stale = stable_wrong)."""
    return {"question": row["messages"][1]["content"].split('User: "', 1)[-1].split('"\n', 1)[0] if row.get("messages") else "",
            "own": row.get("modal_cf"), "class": row.get("gate_class")}  # fmt: skip


def cases(inputs: Path, versions: list[str]):
    for line in (inputs / "cases.jsonl").read_text().splitlines():
        c = json.loads(line)
        env, m = Prompt(c["meta"], c["messages"]), c["meta"]
        base_runs = [Rollout(source=c["source"], messages=c["messages"], label=True, answer=c["modal"])]
        row = {"version": c["version"], "item": c["item"], "model": QWEN, "env_name": ENV, "task": TASK,
               "env": f"eval:cutoff_qa_tool_inline/{c['version']}@{c['item']}", "base_arm": "nohint", "cf_arm": "nohint",
               "k_base": 1, "n_base": 1, "answers_base": {c["modal"]: 1}, "answers_cf": {c["modal"]: 1},
               "modal_cf": c["modal"], "k_cf": m["qwen_modal_count"], "n_cf": 8,
               "gold": m["qwen_modal"], "truth": m["truth"], "truth_now": m.get("truth_now"), "gate_class": m["gate_class"],
               "domain": m["domain"], "answer_type": m["answer_type"], "auto_tool_calls": m["auto_tool_calls"]}  # fmt: skip
        yield env, row, base_runs


if __name__ == "__main__":
    requests(Path(__import__("sys").argv[1]))
