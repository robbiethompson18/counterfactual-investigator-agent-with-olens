"""chive_agam_inline: CHIVE-style yes/no claims on Agam's Qwen3.6-27B incidents, with O-lens / NLA / J++ reads (Robbie, 2026-10-31).

chive_inline asked whether the target's reply helps on CHIVE's official claims; none of those targets has our lenses. Agam's
300 incidents (agu18dec/chive-forensics) are Qwen3.6-27B, thinking off, with every experiment's edit stored as turn ops and
its measured rate. A claim is (incident, experiment) labelled by CHIVE's rule (chive_critique.data.band: true >= 50pp moved,
false <= 15pp, else dropped). Only incidents with at most the median number of read positions are kept (Robbie: smaller
lens blocks, cheaper, simpler; cut on length, not outcome).

The prompt is chive_inline's (CHIVE's plain-arm format, no Question line), except the INTERVENTION is a unified diff of the
conversation (Agam's per-experiment summaries state the result, so they can't be shown). Conditions:
`no_transcript_inline` (edit only: the conversation without the reply), `transcript_inline` (plus the reply), and that plus
O-lens / NLA / J++ reads of the original conversation and reply (never the edited one), each with its placebo: another
kept incident's reads laid on this one's positions segment by segment (prompt / tail / reply; docs/placebo-rules.md).
Positions: every PUNCT token and <|im_end|> of the whole transcript plus the ten tail tokens before the reply, layers
32-60 (docs/read-positions.md); all bullets, every read shown.

    uv run python -m cfbench.eval chive_agam_inline --prep      # cases, layout and the read requests, as a runs.py run
    runpod/olens_reads.sh run POD INPUTS/requests.jsonl chive-agam-olens --layers 32,40,48,52,56,60
    runpod/nla_reads.sh run POD INPUTS/requests.jsonl chive-agam-nla --layers 32,40,48,52,56,60   # then nla_reads.sh score
    READER=scripts/jpp_read_requests.py runpod/olens_reads.sh run POD INPUTS/requests.jsonl chive-agam-jpp \\
      --layers 32,40,48,52,56,60 --positions all --top-k 5
    uv run python -m cfbench.eval chive_agam_inline --show --conditions olens_inline --limit 1
    uv run python -m cfbench.eval chive_agam_inline --slug chive-agam-haiku --draws 3 \\
      --predictors anthropic-batch/claude-haiku-5-5 --batch submit
    uv run python -m cfbench.evals.chive_agam_inline.score data/runs/cfbench/<date>_chive-agam-haiku
"""

import ast
import difflib
import json
import random
import statistics
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

from huggingface_hub import hf_hub_download

from cfbench.olens import PUNCT, show_token
from chive_critique.data import AGAM_REPO, band
from environments._lib.runfolder import REPO_ROOT

ENVS: list[str] = []
MODEL_NAME = None
TASK = "claim"
VERSIONS: dict[str, dict] = {"median": {}}
CONDITIONS = ["no_transcript_inline", "transcript_inline", "olens_inline", "olens_placebo_inline", "nla_inline",
              "nla_placebo_inline", "jlens_inline", "jlens_placebo_inline"]  # fmt: skip
INLINE_EFFORT = "medium"
QWEN = "Qwen/Qwen3.6-27B"
LAYERS = (32, 40, 48, 52, 56, 60)
JPP_K = 3
CONTEXT = 40
TAG = "2026-10-31_chive-agam-median"
READS = {
    lens: REPO_ROOT / f"data/runs/{d}/{TAG}/reads.jsonl"
    for lens, d in (("olens", "olens_reads"), ("nla", "nla_reads"), ("jlens", "jpp_reads"))
}
THINK_END = ["<think>", "\n\n", "</think>", "\n\n"]

TRANSCRIPT = "Transcript:\n{text}"
CLAIM = """\
Claim:
BEHAVIOR (the completion is scored "yes" when): {behaviour}

BASELINE: on the original prompt at temperature 1.0, this behavior occurs in ~{base}% of completions.

INTERVENTION: the conversation was edited as in this diff (lines starting "-" removed, "+" added, the rest unchanged):
```diff
{diff}
```

PREDICTION: applying this intervention (everything else unchanged) would change how often the behavior occurs by at least \
30 percentage points (in either direction) from the ~{base}% baseline.

Is this claim true? In your reasoning, weigh the strongest case that it is TRUE and the strongest case that it is FALSE \
before deciding. Then call `submit_prediction` once with your calibrated probability that the claim is true."""
WHERE = (
    "at every punctuation token and end-of-turn token of the conversation and of the model's reply, and at the start of the "
    'model\'s turn (<|im_start|>, "assistant", the newline, and the empty <think></think> block the model sees with its '
    "thinking mode off, whose closing newlines are the last tokens before its reply)"
)
LENS_TEXT = {
    "olens": f"""\
O-lens reads:

The O-lens is a model trained to describe another model's internal activations in words. It was run on the model's \
activations as it read the transcript above (the original conversation, not the edited one) and wrote its reply, {WHERE}, \
at layers {", ".join(map(str, LAYERS))}. Each read is a few short bullets about what the model is representing at that token, \
at one layer, each with its FVE (fraction of the activation's variance the bullet explains): high FVE is good, low FVE \
bullets are often hallucinated. Reads can show what the model registers but never says; they also often just paraphrase \
the nearby text, and get fine details (exact numbers, names) wrong more often than broad themes.

Each position below shows the text around it with the token read wrapped in <read></read> (↵ is a newline), then its \
bullets, by layer and highest FVE first within a layer.""",
    "nla": f"""\
NLA reads:

An NLA (natural language autoencoder) is a model trained to describe one of another model's internal activations in a short \
free-text explanation; a second model, the critic, tries to rebuild the activation from the explanation alone, which scores \
it: FVE, the fraction of the activation's variance the rebuilt one explains. High FVE is good, and low or negative FVE \
explanations are often confabulated. The NLA was run on the model's activations as it read the transcript above (the \
original conversation, not the edited one) and wrote its reply, {WHERE}, at layers {", ".join(map(str, LAYERS))}, one \
explanation per token and layer. Explanations can show what the model registers but never says; they also often just \
paraphrase the nearby text, and get fine details wrong more often than broad themes.

Each position below shows the text around it with the token read wrapped in <read></read> (↵ is a newline), then its \
explanations by layer, each with its FVE.""",
    "jlens": f"""\
J-lens reads:

The J-lens (Jacobian lens) maps the model's internal activation at one token and one layer to the vocabulary: it lists the \
tokens that activation is pushing the model toward, with the lens's probabilities. It was run on the model's activations as \
it read the transcript above (the original conversation, not the edited one) and wrote its reply, {WHERE}, at layers \
{", ".join(map(str, LAYERS))}. Reads are noisy: at many positions the lens just predicts the next word, and the \
probabilities are the lens's, not the model's.

Each position below shows the text around it with the token read wrapped in <read></read> (↵ is a newline), then one line \
per layer with the lens's top {JPP_K} tokens and their probabilities.""",
}

Pos = tuple[int, str, dict[int, dict]]  # (token index, token, {layer: row})


@cache
def agam() -> dict[str, dict]:
    """prompt_id -> the incident row + its tokens + its ground truth, from the public HF dataset."""
    load = lambda c: {
        r["prompt_id"]: r
        for r in map(json.loads, Path(hf_hub_download(AGAM_REPO, f"data/{c}.jsonl", repo_type="dataset")).read_text().splitlines())
    }
    inc, tok, gt = load("incidents"), load("tokens"), load("ground_truth")
    return {p: inc[p] | tok[p] | {"gt": gt[p]} for p in inc}


def render(messages: list[dict]) -> str:
    """CHIVE's transcript format: `Role: content` turns separated by blank lines."""
    return "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)


def edited(messages: list[dict], turns) -> list[dict]:
    """The experiment's conversation from its turn ops: {"keep": i} keeps original turn i, {"role", "content"} adds one."""
    ops = ast.literal_eval(turns) if isinstance(turns, str) else turns
    return [messages[op["keep"]] if "keep" in op else {"role": op["role"], "content": op["content"]} for op in ops]


def layout(inc: dict) -> dict:
    """{"id", "text", "pos": [(token index, char start, char end, segment)]}: the default read positions of the whole transcript
    (every PUNCT token and <|im_end|>, plus the ten tokens before the reply), segment prompt / tail / reply."""
    pieces = inc["pieces"]
    reply = max(i for i, p in enumerate(pieces) if p == "</think>") + 2
    assert pieces[reply - 4 : reply] == THINK_END, inc["prompt_id"]
    tail = range(reply - 10, reply)
    pos = sorted({k for k, p in enumerate(pieces) if PUNCT.match(p) or p == "<|im_end|>"} | set(tail))
    starts = [0]
    for p in pieces:
        starts.append(starts[-1] + len(p))
    seg = lambda k: "tail" if k in tail else "reply" if k >= reply else "prompt"
    return {"id": inc["prompt_id"], "text": "".join(pieces), "pos": [(k, starts[k], starts[k + 1], seg(k)) for k in pos]}


def prep(out: Path, versions: list[str], placebo: Path, n_items: int | None = None, strict: bool = False):
    """cases.jsonl (one row per labelled experiment of a kept incident), layout.jsonl and requests.jsonl (the reads)."""
    data = agam()
    lays = {p: layout(inc) for p, inc in data.items()}
    cut = statistics.median(len(lay["pos"]) for lay in lays.values())
    keep = sorted(p for p, lay in lays.items() if len(lay["pos"]) <= cut)
    n = Counter()
    with (out / "cases.jsonl").open("w") as f:
        for p in keep:
            for e in data[p]["gt"]["experiments"]:
                label = band(e["baseline_rate"], e["rate"])
                n[label] += 1
                if label != "excluded":
                    f.write(json.dumps({"prompt_id": p, "exp": e["number"], "is_true": label == "true", "baseline": e["baseline_rate"],
                                        "rate": e["rate"], "turns": e["turns"], "is_key": e["is_key"]}) + "\n")  # fmt: skip
    with (out / "layout.jsonl").open("w") as f, (out / "requests.jsonl").open("w") as g:
        for p in keep:
            f.write(json.dumps(lays[p]) + "\n")
            msgs = [*data[p]["messages"], {"role": "assistant", "content": data[p]["completion"]}]  # the reader renders these too
            row = {
                "id": p,
                "env": "chive",
                "messages": msgs,
                "prompt_ids": data[p]["token_ids"],
                "positions": [k for k, *_ in lays[p]["pos"]],
            }
            g.write(json.dumps(row) + "\n")
    print(f"kept {len(keep)} of {len(lays)} incidents (<= {cut} read positions); experiments: {dict(n)}")


def reads(inputs: Path) -> list[str]:
    return []


def cases(inputs: Path, versions: list[str]):
    for c in map(json.loads, (inputs / "cases.jsonl").read_text().splitlines()):
        item = f"{c['prompt_id']}:{c['exp']}"
        row = {"version": "median", "item": item, "model": QWEN, "env_name": "chive", "env": "eval:chive_agam_inline/median",
               "task": TASK, "base_arm": "original", "cf_arm": "intervention", "k_base": 1, "n_base": 2,
               "k_cf": int(c["is_true"]), "n_cf": 1, "is_key": c["is_key"], "rate_base": c["baseline"], "rate_cf": c["rate"]}  # fmt: skip
        yield item, row, []


@cache
def _cases(inputs: str) -> dict[str, dict]:
    return {f"{c['prompt_id']}:{c['exp']}": c for c in map(json.loads, (Path(inputs) / "cases.jsonl").open())}


@cache
def _layout(inputs: str) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, (Path(inputs) / "layout.jsonl").open())}


@cache
def _pool(lens: str) -> dict[str, list[Pos]]:
    """id -> its reads in token order, at the layout's positions (J++ was read at every token: cut to them)."""
    want = {i: {k for k, *_ in lay["pos"]} for i, lay in _layout(str(_inputs())).items()}
    by: dict[str, dict[int, tuple[str, dict]]] = defaultdict(dict)
    for line in READS[lens].open():
        r = json.loads(line)
        if r["token_index"] not in want.get(r["id"], ()):
            continue
        if lens == "jlens":
            by[r["id"]][r["token_index"]] = (
                r["token"],
                {L: {"top": t[:JPP_K]} for L, t in zip(r["layers"], r["top"], strict=True) if L in LAYERS},
            )
        elif r["layer"] in LAYERS:
            kept = {k: r[k] for k in ("bullets", "text", "fve") if k in r}
            by[r["id"]].setdefault(r["token_index"], (r["token"], {}))[1][r["layer"]] = kept
    return {i: [(k, *pos[k]) for k in sorted(pos)] for i, pos in by.items()}


def _inputs() -> Path:
    from cfbench.eval import latest_inputs

    return latest_inputs("chive_agam_inline", ["median"])


def placebo(inputs: str, me: str, seed: int, item: str) -> str:
    """Another kept incident with a read in every segment this one has, seeded per (draw, claim)."""
    lays = _layout(inputs)
    need = {s for *_, s in lays[me]["pos"]}
    ok = sorted(i for i, lay in lays.items() if i != me and need <= {s for *_, s in lay["pos"]})
    return random.Random(f"{seed}:{item}").choice(ok)


def aligned(me: dict, src: dict, reads: list[Pos]) -> list[Pos]:
    """`src`'s reads laid onto `me`'s positions segment by segment, in order, cycled when `src` has fewer."""
    by: dict[str, list[Pos]] = defaultdict(list)
    for (*_, seg), r in zip(src["pos"], reads, strict=True):
        by[seg].append(r)
    seen: Counter = Counter()
    out = []
    for *_, seg in me["pos"]:
        out.append(by[seg][seen[seg] % len(by[seg])])
        seen[seg] += 1
    return out


def _lines(layers: dict[int, dict]) -> list[str]:
    out = []
    for layer in sorted(layers):
        r = layers[layer]
        if "top" in r:
            out.append(f"- layer {layer}: " + " ".join(f"{json.dumps(t, ensure_ascii=False)} {p:.2f}" for t, p in r["top"]))
        elif "text" in r:
            out.append(f"- layer {layer}, FVE {r['fve']:.2f}: {' '.join(r['text'].split())}")
        else:
            out += [
                f"- layer {layer}, FVE {b['fve']:.2f}: {' '.join(b['text'].split())}" for b in sorted(r["bullets"], key=lambda b: -b["fve"])
            ]
    return out


def block(lay: dict, reads: list[Pos]) -> str:
    flat, text = (lambda s: s.replace("\n", "↵")), lay["text"]
    assert len(reads) == len(lay["pos"]), (lay["id"], len(reads), len(lay["pos"]))
    out = []
    for (_, a, b, _), (_, _, layers) in zip(lay["pos"], reads, strict=True):
        ctx = flat(text[max(0, a - CONTEXT) : a]) + f"<read>{show_token(text[a:b])}</read>" + flat(text[b : b + CONTEXT])
        out += [f"At {json.dumps(text[a:b])}: …{ctx}…", *_lines(layers), ""]
    return "\n".join(out).strip()


def INLINE(env, condition: str, case: dict, base: list, seed: int) -> tuple[str, dict]:
    assert condition in CONDITIONS, condition
    c, inc = _cases(case["inputs"])[case["item"]], agam()[case["item"].split(":")[0]]
    msgs = inc["messages"]
    convo = msgs if condition == "no_transcript_inline" else [*msgs, {"role": "assistant", "content": inc["completion"]}]
    diff = "\n".join(difflib.unified_diff(render(msgs).splitlines(), render(edited(msgs, c["turns"])).splitlines(), "original",
                                          "edited", lineterm="", n=2))  # fmt: skip
    parts, extra = [TRANSCRIPT.format(text=render(convo))], {}
    if condition not in ("no_transcript_inline", "transcript_inline"):
        lens, me = condition.split("_")[0], c["prompt_id"]
        src = placebo(case["inputs"], me, seed, case["item"]) if "placebo" in condition else me
        if src != me:
            extra["placebo_sources"] = [src]
        lays = _layout(case["inputs"])
        parts += [LENS_TEXT[lens], block(lays[me], aligned(lays[me], lays[src], _pool(lens)[src]))]
    parts.append(CLAIM.format(behaviour=inc["behaviour"], base=round(100 * c["baseline"]), diff=diff))
    return "\n\n".join(parts), extra
