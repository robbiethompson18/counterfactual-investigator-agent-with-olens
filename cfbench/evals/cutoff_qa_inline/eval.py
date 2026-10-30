"""cutoff_qa_inline: predict Qwen3.6-27B's answer to a present-tense factual question from the prompt alone.

Robbie, 2026-10-29. The investigator sees the question Qwen was asked (not the reply) and predicts the reply in one turn with one
submit_answer call (knowledge_cutoff's prompt, PR #378; inline.py here is a copy). The label is Qwen's modal answer from
the bank's stability gate (`qwen_modal`, >= 7/8 samples agree), not the truth; stable_wrong items are the stale-belief ones. No
counterfactual: base arm = cf arm = `nohint`, truth = the modal answer, so cfbench.answer's "edit does nothing" column is always
yes. Cases: the episodes of the Qwen env runs RUNS, in bank order: the first 200 stable, non-numeric v2 items by bank order
(smoke200), plus 800 seeded-random stable non-numeric v3 items not among them (new800; the 1k scale-up, 2026-10-29).

Conditions (single-turn, no tools): `transcript_inline` (the question), `olens_inline` (plus the item's own end-of-prompt reads:
the "." ending the question, <|im_end|>, "\\n", <|im_start|>, "assistant", "\\n", layers 32-60, from READS),
`olens_placebo_inline` (another item's reads at the same positions, from the same file, presented as this item's). The model is
never named (MODEL_NAME; the runner asserts the id never reaches the investigator). Investigator: Claude Haiku 5.5 at effort
medium (`--predictors anthropic/claude-haiku-5.5`): Gemini 3.8 Flash shares Qwen's cutoff and holds the stale answer itself on
a quarter of WorkspaceBench's items (docs/envs/knowledge_cutoff.md).

    uv run python -m cfbench.eval cutoff_qa_inline --prep
    uv run python -m cfbench.eval cutoff_qa_inline --show --limit 1
    uv run python -m cfbench.eval cutoff_qa_inline --slug cutoff-qa-haiku-200 --predictors anthropic/claude-haiku-5.5 --draws 3
    uv run python -m cfbench.evals.cutoff_qa_inline.report data/runs/cfbench/<date>_cutoff-qa-haiku-200

Scoring: `correct` = the submission's `norm` (environments/cutoff_qa/env.py, the same form as `qwen_modal`) equals the modal
answer; report.py adds a loose whole-word match, per gate_class, the paired sign test vs transcript_inline and read-level recall.

NLA conditions (2026-10-29; not in CONDITIONS, pass `--conditions nla_inline,nla_placebo_inline` once NLA_READS exist and
the inputs are re-prepped): `nla_inline` puts the item's own NLA readouts (agu18dec/qwen3.6-27b-all-layer-nla-rl2000,
olens/src/olens/nla.py) at the O-lens block's 6 positions and layers in place of the O-lens block, one readout per (position,
layer), every readout shown with its critic FVE; `nla_placebo_inline` is an unrelated prompt's readouts at the same 6 tokens,
from the 200 MMLU-Pro no-hint prompts NLA_READS also holds (Robbie, 2026-10-29: the placebo is "some totally unrelated
thing", not another cutoff item), chosen with the O-lens placebo's seed. A readout is ~140-185 tokens, so the 36 make a
~5.5-6.5k-token block, about the O-lens block's size (5.7k Qwen tokens on the first case, 4 bullets x 36 reads). The reads
are the O-lens tail run's requests plus the MMLU-Pro pool's through the NLA reader, then the critic (runpod/nla_reads.sh,
docs/nla-reader.md), in a `scripts/runs.py new nla_reads cutoff200-tail` run, finished with `scripts/runs.py finish`:

    runpod/nla_reads.sh setup NAME
    runpod/nla_reads.sh run NAME data/runs/nla_reads/2026-10-29_cutoff200-tail/requests.jsonl cutoff200-nla --layers 32,40,48,52,56,60
    runpod/nla_reads.sh score NAME cutoff200-nla
    runpod/nla_reads.sh fetch NAME cutoff200-nla data/runs/nla_reads/2026-10-29_cutoff200-tail/reads.jsonl

J++ conditions (2026-10-29; not in CONDITIONS): `jlens_inline` puts the item's own J++ reads (neuronpedia/jacobian-lens
qwen3.6-27b/jpp) at every token of the rendered prompt (thinking-off render, empty think block included) in place of the
O-lens block, top inline.JLENS_K tokens at JLENS_LAYERS (the O-lens block's), from JREADS; `jlens_placebo_inline` is the
last n prompt tokens of a seeded hint_mcq transcript's every-token J++ read (JPOOL), n = this item's token count, each shown
with its own tokens; `jlens_inline_no60` is jlens_inline without layer 60, whose read at the last token is close to Qwen's
own next-token distribution. ~4.5k tokens per block. Results: PR #393.
"""

import json
import random
from pathlib import Path

from cfbench.core import Rollout, render
from cfbench.evals.cutoff_qa_inline import inline
from environments._lib.core import load_env
from environments._lib.runfolder import LOCAL, read_episodes

ENV = "cutoff_qa"
ENVS = [ENV]
RUNS = ("2026-10-29_smoke200-nohint", "2026-10-29_new800-nohint")  # the Qwen env runs whose prompts (and reads) the cases are
READS = [Path(f"data/runs/olens_reads/2026-10-29_cutoff{n}-tail/reads.jsonl") for n in (200, 800)]  # end-of-prompt reads of RUNS
QWEN = "Qwen/Qwen3.6-27B"
MODEL_NAME = "the model"
TASK = "answer"
CONDITIONS = ("transcript_inline", "olens_inline", "olens_placebo_inline")
INLINE_EFFORT = "medium"
SYSTEM = inline.SYSTEM
VERSIONS: dict[str, dict] = {"nohint": {"arm": "nohint"}}
ARM_LABELS = {"nohint": "the question, verbatim"}
PAIR_LABELS = {"nohint -> nohint": "no counterfactual"}
NLA_READS = [Path(f"data/runs/nla_reads/2026-10-29_cutoff{n}-tail/reads.jsonl") for n in (200, 800)]  # at READS's positions, layers
JREADS = [Path(f"data/runs/jpp_reads/2026-10-29_cutoff{n}-all/reads.jsonl") for n in (200, 800)]  # J++ at every token of RUNS
JPOOL = Path("data/runs/jpp_reads/2026-10-27_hint-mcq-header/reads.jsonl")  # J++ placebo pool: hint_mcq, every token, L0-60/4
JLENS_LAYERS = (32, 40, 48, 52, 56, 60)  # the O-lens reads' layers


def _mod():
    return load_env(ENV)


def collect() -> list[dict]:
    """One case per episode of RUNS, in bank order: the prompt, Qwen's gated modal answer (the label) and the bank metadata."""
    mod = _mod()
    order = {item: k for k, item in enumerate(mod.items())}
    out = []
    for run, ep in ((r, ep) for r in RUNS for ep in read_episodes(LOCAL / ENV / r)):
        assert not ep.get("error") and ep["target"].split("@")[0].lower() == QWEN.lower(), ep["id"]
        meta = ep["meta"]
        assert not meta["numeric"] and meta["gate_class"].startswith("stable"), ep["id"]
        out.append({"version": "nohint", "item": ep["item"], "modal": mod.norm(meta["qwen_modal"]), "source": f"{ENV}/{run}:{ep['id']}",
                    "messages": ep["messages"][:1], "meta": meta})  # fmt: skip
    return sorted(out, key=lambda c: order[c["item"]])


def prep(out: Path, versions: list[str], placebo: Path, n_items: int | None = None, strict: bool = False):
    """cases.jsonl plus reads/reads.jsonl (READS, all of them); the placebo is drawn from the same file (another item)."""
    (out / "reads").mkdir(parents=True)
    (out / "reads/reads.jsonl").write_text("".join(p.read_text() for p in READS))
    cs = collect()[: n_items or None]
    (out / "reads/nla.jsonl").write_text("".join(p.read_text() for p in NLA_READS))  # the nla_* conditions' reads and placebo pool
    for srcs, dst, prompt_only in ((JREADS, "jreads", False), ([JPOOL], "jpool", True)):  # the jlens_* conditions' reads and pool
        rows = [r for src in srcs for rs in jpp_rows(src, prompt_only).values() for r in rs]
        (out / f"reads/{dst}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (out / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cs))
    print(len(cs), "cases")


def reads(inputs: Path) -> list[str]:
    return [str(p) for p in [inputs / "reads/reads.jsonl"] if p.exists()]


class Prompt:
    BEHAVIOR = "Positive if the reply names Qwen3.6-27B's own stable answer to the question (the bank's qwen_modal)."
    ANSWER_FORMAT = "just the answer, as the question asks (a name, a title, a place)"
    ANSWER_MATCH = "the answer normalized (lowercase ASCII, punctuation and a leading article stripped) equals Qwen's modal answer"

    def __init__(self, meta: dict, messages: list[dict]):
        self.meta, self.start = meta, render(messages[:1])

    def view(self, base_arm: str, cf_arm: str, rollout: Rollout) -> tuple[str, str]:
        return self.start, "No change: there is no counterfactual arm."

    def canon_answer(self, text: str) -> str | None:
        return _mod().norm(text or "") or None


def cases(inputs: Path, versions: list[str]):
    for line in (inputs / "cases.jsonl").read_text().splitlines():
        c = json.loads(line)
        env, m = Prompt(c["meta"], c["messages"]), c["meta"]
        base = [Rollout(source=c["source"], messages=c["messages"], label=True, answer=c["modal"])]
        row = {"version": c["version"], "item": c["item"], "model": QWEN, "env_name": ENV, "task": TASK,
               "env": f"eval:cutoff_qa_inline/{c['version']}@{c['item']}", "base_arm": "nohint", "cf_arm": "nohint",
               "k_base": 1, "n_base": 1, "answers_base": {c["modal"]: 1}, "answers_cf": {c["modal"]: 1},
               "modal_cf": c["modal"], "k_cf": m["qwen_modal_count"], "n_cf": 8,
               "gold": m["qwen_modal"], "truth": m["truth"], "truth_now": m.get("truth_now"), "gate_class": m["gate_class"],
               "domain": m["domain"], "answer_type": m["answer_type"]}  # fmt: skip
        yield env, row, base


def INLINE(env, condition: str, case: dict, base: list, seed: int) -> tuple[str, dict]:
    if condition.startswith("nla"):
        return nla_inline(condition, case, base, seed)
    if condition.startswith("jlens"):
        return jlens_inline(condition, case, base, seed)
    question = base[0].messages[0]["content"]
    rows, extra = None, {}
    if condition != "transcript_inline":
        by_source = inline.reads_by_source(Path(case["inputs"]) / "reads/reads.jsonl")
        src = base[0].source
        if condition == "olens_placebo_inline":  # another item's reads, seeded per (draw, case), never its own
            src = random.Random(f"{seed}:{case['version']}:{case['item']}").choice(sorted(set(by_source) - {src}))
            extra = {"placebo_sources": [src]}
        rows = by_source[src]
    return inline.prompt(question, condition, False, MODEL_NAME, rows), extra


def LINKS(row: dict) -> dict[str, dict]:
    env_run, _, ep = row["shown"][0].partition(":")
    env, run = env_run.split("/")
    return {"transcript": {"env": env, "run": run, "episode": ep}}


def nla_inline(condition: str, case: dict, base: list, seed: int) -> tuple[str, dict]:
    """nla_inline / nla_placebo_inline: the O-lens conditions' pattern over the NLA readouts (reads/nla.jsonl, NLA_READS)."""
    by_source = inline.reads_by_source(Path(case["inputs"]) / "reads/nla.jsonl")
    src, extra = base[0].source, {}
    if condition == "nla_placebo_inline":  # an unrelated prompt's readouts (NLA_READS' MMLU-Pro pool), seeded per (draw, case)
        pool = sorted(s for s in by_source if not s.startswith(f"{ENV}/"))
        src = random.Random(f"{seed}:{case['version']}:{case['item']}").choice(pool)
        extra = {"placebo_sources": [src]}
    return inline.nla_prompt(base[0].messages[0]["content"], False, MODEL_NAME, by_source[src]), extra


def jpp_rows(path: Path, prompt_only: bool = False) -> dict[str, list[dict]]:
    """id -> J++ rows at JLENS_LAYERS in token order; `prompt_only` cuts each transcript at its first assistant (reply) token."""
    out: dict[str, list[dict]] = {}
    for line in path.open():
        r = json.loads(line)
        keep = [r["layers"].index(L) for L in JLENS_LAYERS]
        out.setdefault(r["id"], []).append(r | {"layers": list(JLENS_LAYERS), "top": [r["top"][k] for k in keep]})
    for i, rows in out.items():
        rows.sort(key=lambda r: r["token_index"])
        if prompt_only:
            out[i] = rows[: next((k for k, r in enumerate(rows) if r["role"] == "assistant"), len(rows))]
    return out


def jlens_inline(condition: str, case: dict, base: list, seed: int) -> tuple[str, dict]:
    """jlens_inline / jlens_placebo_inline / jlens_inline_no60 over reads/jreads.jsonl and reads/jpool.jsonl (prep)."""
    rows = own = inline.reads_by_source(Path(case["inputs"]) / "reads/jreads.jsonl")[base[0].source]
    extra = {}
    if condition == "jlens_placebo_inline":  # a seeded hint_mcq transcript's last len(own) prompt tokens
        pool = inline.reads_by_source(Path(case["inputs"]) / "reads/jpool.jsonl")
        src = random.Random(f"{seed}:{case['version']}:{case['item']}").choice(sorted(pool))
        rows, extra = pool[src][-len(own) :], {"placebo_sources": [src]}
    if condition == "jlens_inline_no60":
        rows = [r | {"layers": r["layers"][:-1], "top": r["top"][:-1]} for r in own]
        assert rows[0]["layers"] == [32, 40, 48, 52, 56], rows[0]["layers"]
    return inline.jlens_prompt(base[0].messages[0]["content"], False, MODEL_NAME, rows), extra
