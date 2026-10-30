"""The hint flip probe (Kylie, 2026-10-29; docs/hint-flip-probe.md): for each element of a hinted MCQ prompt (the hint, the
frame line), "if we removed it, would Qwen's letter change?", predicted from the activations of the prompt WITH the element
only (zero counterfactuals), labelled by Qwen's bf16 letter argmax with and without it.

Per question 15 prompts (KEYS): no hint under three frames (none, the jsonapp JSON line, an eval header) and, per hint
source (the professor, Wikipedia; always the correct letter), the hint after the options, before the question, under the
jsonapp line after / before, as the jsonapp JSON's `note`, and under the header. Removal cases (CASES): the hint from every
hinted prompt (to the no-hint prompt of the same frame; the note to jsonapp/no hint), and the frame from every framed
prompt but the note (to the same prompt unframed).

    # CPU: questions (two pools), prompts, sites; inputs for scripts/hs_at_positions.py
    uv run --with transformers --with jinja2 python -m environments.hint_mcq.flip prep [--smoke 25]
    # GPU: python scripts/hs_at_positions.py RUN/hs_inputs.jsonl data/probe_cache/<run name> --layers 32,40,48,52,56,60 --chunk 2000
    # CPU: flip labels and per-stratum rates (needs only logp.jsonl)
    uv run python -m environments.hint_mcq.flip labels RUN
    # CPU: probes and baselines
    uv run --with scikit-learn python -m environments.hint_mcq.flip probe RUN

Pools: `balanced` = MMLU-Pro questions from the probe half of the mcq_unselected split (500 Qwen gets wrong, 500 right, by
the strict no-hint census's modal letter, no consistency filter) and TruthfulQA (every question Qwen's control-n8 modal
answer gets wrong, plus as many right); `twomodel` = the hint_qwen_dsv4 pool (twomodel_qwen_dsv4_follow.csv: Qwen and
DSv4 Flash consistent and disagreeing), minus the balanced draw and minus any MMLU-Pro item in an eval half.
"""

import argparse
import contextlib
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from counterfactual_investigator_agent_with_olens.runs import new_run
from environments._lib.runfolder import LOCAL, REPO_ROOT, read_episodes
from environments.hint_mcq.probe import CACHE, MODEL

SPLIT = REPO_ROOT / "data/runs/cfbench_evals/2026-10-29_mcq-unselected-inputs-strict/split.json"
FOLLOW = Path(__file__).with_name("twomodel_qwen_dsv4_follow.csv")
TQA_CONTROL = ["2026-10-27_control-n8-qwen36-nothink", "2026-10-27_control-n8-q400-816-qwen36-nothink"]
LAYERS = [32, 40, 48, 52, 56, 60]  # hidden_states indices
N_TAIL = 10  # the instruction's "." (MMLU-Pro) through the empty think block: the last 10 tokens of every prompt
SOURCES = ["prof", "wiki"]
FRAMES = ["none", "jsonapp", "header"]
# key = (frame, position, source); position "-" = no hint, "note" = the hint as the jsonapp JSON's note
KEYS = [(f, "-", "-") for f in FRAMES] + [
    k
    for s in SOURCES
    for k in [
        ("none", "after", s),
        ("none", "before", s),
        ("jsonapp", "after", s),
        ("jsonapp", "before", s),
        ("jsonapp", "note", s),
        ("header", "after", s),
    ]
]
# (element, with, without)
CASES = [("hint", k, ("jsonapp" if k[1] == "note" else k[0], "-", "-")) for k in KEYS if k[2] != "-"]
CASES += [("frame", k, ("none", k[1], k[2])) for k in KEYS if k[0] != "none" and k[1] != "note"]


def key_name(k) -> str:
    return "|".join(k)


def arm(dataset: str, k) -> str:
    frame, pos, src = k
    if dataset == "mmlu_pro":
        h = {"prof": "hint+d1", "wiki": "wiki+d1"}.get(src, "nohint")
        if pos == "note":
            return f"jsonnote/{h}"
        if frame == "jsonapp" and pos == "before":
            return f"jsonapp-before/{h}"
        pre = {"none": "", "jsonapp": "jsonapp/", "header": "mmlu/"}[frame] + ("before/" if pos == "before" else "")
        return pre + h
    s = {"prof": "stanford", "wiki": "wikipedia"}.get(src)
    if pos == "note":
        return f"jsonnote/{s}"
    base = "control" if pos == "-" else f"hint/{s}" if pos == "after" else f"hint_before/{s}"
    return {"none": "", "jsonapp": "jsonapp/", "header": "header/"}[frame] + base


def _mmlu_census() -> dict[str, Counter]:
    from cfbench.evals.mcq_unselected.eval import census

    return census()


def select(seed: int) -> list[dict]:
    """Both pools' questions: {pool, dataset, item (the env item: MMLU-Pro id / tqa-N.h<gold>), qid, gold, census letter,
    right} (+ DSv4's letter on twomodel)."""
    from environments.hint_mcq.env import _table
    from environments.hinted_truthfulqa import env as tqa

    rng = random.Random(seed)
    sp = json.loads(SPLIT.read_text())
    table, cs = _table(), _mmlu_census()
    out = []
    mm = {"right": [], "wrong": []}
    for q in sp["probe"]:
        letter = cs[q].most_common(1)[0][0]
        if letter is not None:
            mm["right" if letter == table.loc[q].answer else "wrong"].append((q, letter))
    for k in ("wrong", "right"):
        for q, letter in rng.sample(mm[k], 500):
            out.append(
                {
                    "pool": "balanced",
                    "dataset": "mmlu_pro",
                    "item": q,
                    "qid": q,
                    "gold": table.loc[q].answer,
                    "census": letter,
                    "right": k == "right",
                }
            )
    ans, gold = defaultdict(list), {}
    for run in TQA_CONTROL:
        for ep in read_episodes(LOCAL / "hinted_truthfulqa" / run):
            if ep["arm"] == "control" and str(ep["item"]).startswith("tqa-"):
                ans[str(ep["item"])].append(tqa.measure(ep)["answer"])
                gold[str(ep["item"])] = ep["meta"]["gold"]
    tq = {"right": [], "wrong": []}
    for it, xs in sorted(ans.items(), key=lambda x: int(x[0].split("-")[1])):
        letter = Counter(xs).most_common(1)[0][0]
        tq["right" if letter == gold[it] else "wrong"].append((it, letter))
    for k, rows in (("wrong", tq["wrong"]), ("right", rng.sample(tq["right"], len(tq["wrong"])))):
        for it, letter in rows:
            out.append(
                {
                    "pool": "balanced",
                    "dataset": "truthfulqa",
                    "item": f"{it}.h{gold[it]}",
                    "qid": it,
                    "gold": gold[it],
                    "census": letter,
                    "right": k == "right",
                }
            )
    taken = {(r["dataset"], r["qid"]) for r in out}
    eval_items = set(sp["eval"]) | set(sp.get("eval2", []))
    drop = Counter()
    for r in csv.DictReader(FOLLOW.open()):
        ds = r["source"]
        if (ds, r["item"]) in taken:
            drop["in balanced"] += 1
            continue
        if ds == "mmlu_pro" and r["item"] in eval_items:
            drop["mmlu_pro in an eval half"] += 1
            continue
        item = r["item"] if ds == "mmlu_pro" else f"{r['item']}.h{r['gold']}"
        out.append(
            {"pool": "twomodel", "dataset": ds, "item": item, "qid": r["item"], "gold": r["gold"], "census": r["qwen_answer"]}
            | {"right": r["qwen_answer"] == r["gold"], "dsv4": r["dsv4_answer"], "dsv4_right": r["dsv4_answer"] == r["gold"]}
        )
    print(f"twomodel dropped: {dict(drop)}")
    print(Counter((r["pool"], r["dataset"], r["right"]) for r in out))
    return out


def render(tok, dataset: str, k, item: str) -> dict:
    """Token ids of the thinking-off generation prompt and the sites: `nl` (per option, the token right after its text: a
    newline, or <|im_end|> when the option ends the user turn), `el` (the hint sentence's last token, the frame line's last
    token; -1 when absent), `tail` (the last N_TAIL tokens). Also the hinted letter and candidate tokens."""
    from environments._lib.core import load_env

    env = load_env("hint_mcq" if dataset == "mmlu_pro" else "hinted_truthfulqa")
    ep = env.build(arm(dataset, k), item)
    user = ep.messages[-1]["content"]
    text = tok.apply_chat_template(ep.messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]

    def at(c):  # token covering char c
        return next(i for i, (s, e) in enumerate(offs) if s <= c < e)

    u0, u1 = text.index(user), text.index(user) + len(user)
    letters = "ABCDEFGHIJKLMNOP"
    pat = (lambda L: f"\n{L}. ") if dataset == "mmlu_pro" else (lambda L: f"\n({L}) ")
    starts = []
    for L in letters:
        nxt = text.find(pat(L), starts[-1] + 1 if starts else u0, u1)
        if nxt < 0:
            break
        starts.append(nxt)
    stop = text.find("\n\n", starts[-1], u1)
    nl = [at(c) for c in [*starts[1:], u1 if stop < 0 else stop]]
    hint = ep.meta.get("hint")
    el = [-1, -1]
    if k[2] != "-":
        if dataset == "mmlu_pro":
            from environments.hint_mcq.env import SENTENCES

            sentence = SENTENCES["wiki" if k[2] == "wiki" else "hint"].format(d=hint)
        else:
            from environments.hinted_truthfulqa.env import HINTS

            sentence = HINTS["hint/stanford" if k[2] == "prof" else "hint/wikipedia"].format(x=hint)
        el[0] = at(text.index(sentence, u0) + len(sentence) - 1)
    if k[0] != "none":
        el[1] = at(text.index("\n\n", u0) - 1)
    cand = [tok.encode(f"{L}" if dataset == "mmlu_pro" else f"({L}", add_special_tokens=False)[0] for L in letters[: len(nl)]]
    return {"ids": ids, "sites": {"nl": nl, "el": el, "tail": list(range(len(ids) - N_TAIL, len(ids)))}, "cand": cand, "hint": hint}


def prep(a):
    from transformers import AutoTokenizer

    qs = select(a.seed)
    if a.smoke:  # the first N per (pool, dataset) in a seeded shuffle
        random.Random(a.seed).shuffle(qs)
        rank = Counter()
        keep = []
        for q in qs:
            rank[(q["pool"], q["dataset"])] += 1
            if rank[(q["pool"], q["dataset"])] <= a.smoke:
                keep.append(q)
        qs = keep
    tok = AutoTokenizer.from_pretrained(MODEL)
    run = new_run(
        "hint_flip_probe",
        a.slug,
        description="Hint flip probe: would removing the hint / frame line change Qwen's letter? (bf16 letter argmax, activations)",
        models=[{"name": MODEL, "served_by": "HF transformers bf16 (scripts/hs_at_positions.py)"}],
        params=vars(a) | {"layers": LAYERS, "n_tail": N_TAIL, "keys": [key_name(k) for k in KEYS], "n_questions": len(qs)},
    )
    inputs, prompts, off = [], [], Counter()
    for qi, q in enumerate(qs):
        q["q"] = qi
        for k in KEYS:
            r = render(tok, q["dataset"], k, q["item"])
            n = len(r["sites"]["nl"])
            assert n == len(r["cand"]) and all(x < len(r["ids"]) for x in r["sites"]["nl"] + r["sites"]["el"])
            pid = f"{q['dataset']}/{q['item']}/{key_name(k)}"
            inputs.append({"id": pid, "ids": r["ids"], "sites": r["sites"], "cand": r["cand"]})
            prompts.append(
                {"p": len(prompts), "id": pid, "q": qi, "key": key_name(k), "arm": arm(q["dataset"], k), "n_opts": n, "hint": r["hint"]}
                | {"nl_off": off["nl"], "n_tokens": len(r["ids"]), "has_el": [x >= 0 for x in r["sites"]["el"]]}
                | {
                    "nl_tokens": [tok.decode([r["ids"][i]]) for i in r["sites"]["nl"]],
                    "el_tokens": [tok.decode([r["ids"][i]]) for i in r["sites"]["el"]],
                }
            )
            off["nl"] += n
        if qi % 100 == 0:
            print(qi, len(qs), flush=True)
    (run / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs))
    (run / "prompts.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts))
    (run / "hs_inputs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in inputs))
    print(f"nl tokens {Counter(t for p in prompts for t in p['nl_tokens']).most_common(6)}")
    print(f"el tokens {Counter(t for p in prompts for t in p['el_tokens']).most_common(8)}")
    print(
        f"run {run}: {len(qs)} questions, {len(prompts)} prompts, {off['nl']} option sites; then hs_at_positions.py into {CACHE / run.name}"
    )


def _load(run: Path):
    qs = [json.loads(line) for line in (run / "questions.jsonl").read_text().splitlines()]
    prompts = [json.loads(line) for line in (run / "prompts.jsonl").read_text().splitlines()]
    cache = CACHE / run.name
    parts = sorted(cache.glob("part-*")) or [cache]
    logp = [json.loads(line) for p in parts for line in (p / "logp.jsonl").read_text().splitlines()]
    assert [x["id"] for x in logp] == [p["id"] for p in prompts[: len(logp)]]
    return qs, prompts[: len(logp)], [np.array(x["logp"]) for x in logp], parts


def consistent(logp: np.ndarray, temp: float = 0.7, top_p: float = 0.8, n: int = 8, k: int = 7) -> tuple[bool, float]:
    """Would Qwen give the same letter in >= k of n samples (the other evals' consistency rule), from the letter logits
    alone: temper to the envs' thinking-off sampling (T 0.7, top-p 0.8), take the argmax letter's probability p in the
    nucleus, and call it consistent if P(Binomial(n, p) >= k) >= 0.5. -> (consistent, p)."""
    from math import comb

    q = np.exp((logp - logp.max()) / temp)
    q = np.sort(q / q.sum())[::-1]
    keep = q[: int(np.searchsorted(np.cumsum(q), top_p) + 1)]
    pt = float(keep[0] / keep.sum())
    return sum(comb(n, j) * pt**j * (1 - pt) ** (n - j) for j in range(k, n + 1)) >= 0.5, pt


def cases(qs, prompts, logp) -> list[dict]:
    """One row per (question, removal) with both prompts' letter pmfs over the question's options."""
    by = {(p["q"], p["key"]): p["p"] for p in prompts}
    letters = "ABCDEFGHIJKLMNOP"
    out = []
    for q in qs:
        for el, kw, ko in CASES:
            pw, po = by.get((q["q"], key_name(kw))), by.get((q["q"], key_name(ko)))
            if pw is None or po is None:
                continue
            fw, fo = np.exp(logp[pw]), np.exp(logp[po])
            mw, mo = fw / fw.sum(), fo / fo.sum()
            hint = letters.index(prompts[pw]["hint"]) if prompts[pw]["hint"] else letters.index(q["gold"])
            aw, ao = int(np.argmax(mw)), int(np.argmax(mo))
            srt = np.sort(mw)[::-1]
            out.append(
                {k: q[k] for k in ("q", "pool", "dataset", "qid", "gold", "right")}
                | {"dsv4_right": q.get("dsv4_right")}
                | {"element": el, "with": key_name(kw), "without": key_name(ko), "p_with": pw, "p_without": po}
                | {"frame": kw[0], "position": kw[1], "source": kw[2], "flip": aw != ao}
                | {"ans_with": letters[aw], "ans_without": letters[ao], "hinted": letters[hint]}
                | {"tv": float(0.5 * np.abs(mw - mo).sum()), "dp_hint": float(mw[hint] - mo[hint])}
                | {
                    "p_hint_with": float(mw[hint]),
                    "margin_with": float(srt[0] - srt[1]),
                    "entropy_with": float(-(mw * np.log(mw + 1e-12)).sum()),
                }
                | {"mass_with": float(fw.sum()), "mass_without": float(fo.sum())}
                | {"top_with": float(mw.max()), "top_without": float(mo.max())}
                | {"consistent": consistent(logp[pw])[0] and consistent(logp[po])[0]}
            )
    return out


def labels(a):
    run = Path(a.run)
    qs, prompts, logp, _ = _load(run)
    rows = cases(qs, prompts, logp)
    (run / "cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    # agreement of the bf16 no-hint argmax with the census letter
    letters = "ABCDEFGHIJKLMNOP"
    nohint = {p["q"]: logp[p["p"]] for p in prompts if p["key"] == "none|-|-"}
    nh = [(q, nohint[q["q"]]) for q in qs if q["q"] in nohint]
    rep = [f"# Flip labels: {run.name}", "", f"{len(qs)} questions, {len(prompts)} prompts, {len(rows)} removal cases", ""]
    for pool in ("balanced", "twomodel"):
        for ds in ("mmlu_pro", "truthfulqa"):
            xs = [letters[int(np.argmax(lp))] == q["census"] for q, lp in nh if q["pool"] == pool and q["dataset"] == ds]
            if xs:
                rep.append(f"- {pool}/{ds}: bf16 no-hint argmax = census modal letter on {np.mean(xs):.1%} of {len(xs)}")
    # the logit consistency call vs the empirical >= 7/8 call on the MMLU-Pro no-hint census (fp8 OpenRouter, 8 samples)
    cs = _mmlu_census()
    pairs = [(consistent(lp)[0], cs[q["qid"]].most_common(1)[0][1] >= 7) for q, lp in nh if q["dataset"] == "mmlu_pro" and q["qid"] in cs]
    if pairs:
        tab = Counter(pairs)
        rep.append(
            f"- consistency, logits vs census >= 7/8 (MMLU-Pro no-hint, {len(pairs)} questions):"
            f" agree {np.mean([x == y for x, y in pairs]):.1%};"
            f" both {tab[(True, True)]}, logits only {tab[(True, False)]}, census only {tab[(False, True)]}, neither {tab[(False, False)]}"
        )
    mass = np.array([r["mass_with"] for r in rows])
    rep += [f"- letter-token mass at the answer position: median {np.median(mass):.3f}, 5th pct {np.percentile(mass, 5):.3f}", ""]
    rep += [
        (
            "| pool | dataset | element | with | without | n | flip | flip (Qwen right / wrong) | TV | Δp(hinted) | n consistent"
            " | flip (consistent) |"
        ),
        "|" + " --- |" * 12,
    ]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["pool"], r["dataset"], r["element"], r["with"], r["without"])].append(r)
    for g, rs in sorted(groups.items()):
        f = np.mean([r["flip"] for r in rs])
        fr = [np.mean([r["flip"] for r in rs if r["right"] == v]) if any(r["right"] == v for r in rs) else np.nan for v in (True, False)]
        tv, dp = np.mean([r["tv"] for r in rs]), np.mean([r["dp_hint"] for r in rs])
        rc = [r["flip"] for r in rs if r["consistent"]]
        fc = f"{np.mean(rc):.2f}" if rc else ""
        rep.append(f"| {' | '.join(g)} | {len(rs)} | {f:.2f} | {fr[0]:.2f} / {fr[1]:.2f} | {tv:.2f} | {dp:+.2f} | {len(rc)} | {fc} |")
    n_flip = sum(r["flip"] for r in rows)
    rep += ["", f"Overall: {n_flip} flips of {len(rows)} cases ({n_flip / len(rows):.1%})", ""]
    for pool in ("balanced", "twomodel", None):
        for e in ("hint", "frame"):
            rc = [r for r in rows if r["element"] == e and pool in (None, r["pool"])]
            kc = [r["flip"] for r in rc if r["consistent"]]
            rep.append(
                f"- consistent on both prompts, {pool or 'pooled'} / {e}: {len(kc)} of {len(rc)} cases, {sum(kc)} flips ({np.mean(kc):.1%})"
            )
    (run / "labels.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


def split(a):
    """Hold out questions for a later investigator eval (Kylie, 2026-10-30): a seeded, per-(pool, dataset) stratified share
    of questions, the smallest share whose consistent cases hold >= n/2 flips, and from those n removal cases 50:50 flip.
    Writes RUN/split.json; `probe` drops these questions entirely."""
    run = Path(a.run)
    C = [json.loads(line) for line in (run / "cases.jsonl").read_text().splitlines()]
    qs = [json.loads(line) for line in (run / "questions.jsonl").read_text().splitlines()]
    rng = random.Random(a.seed)
    strata = defaultdict(list)
    for q in qs:
        strata[(q["pool"], q["dataset"])].append(q["q"])
    order = {k: rng.sample(v, len(v)) for k, v in sorted(strata.items())}
    for share in np.arange(0.05, 0.51, 0.01):
        held = {q for v in order.values() for q in v[: max(1, round(share * len(v)))]}
        cs = [i for i, c in enumerate(C) if c["q"] in held and c["consistent"]]
        flips = [i for i in cs if C[i]["flip"]]
        if len(flips) >= a.n // 2:
            break
    rest = [i for i in cs if not C[i]["flip"]]
    picked = sorted(rng.sample(flips, a.n // 2) + rng.sample(rest, a.n // 2))
    key = lambda i: f"{C[i]['q']}|{C[i]['with']}|{C[i]['without']}"
    out = {
        "seed": a.seed,
        "share": round(float(share), 2),
        "eval_questions": sorted(held),
        "n_eval_questions": len(held),
        "per_stratum": {"/".join(k): sum(q in held for q in v) for k, v in order.items()},
        "n_cases_in_eval_questions": len(cs),
        "n_flips_in_eval_questions": len(flips),
        "eval_cases": [key(i) for i in picked],
    }
    (run / "split.json").write_text(json.dumps(out, indent=1) + "\n")
    print({k: v for k, v in out.items() if k not in ("eval_questions", "eval_cases")})


class Acts:
    """Activations of a run (scripts/hs_at_positions.py output, one dir or part-*/ dirs) at one hidden_states index, per site;
    rows: `tail` N_TAIL per prompt, `el` 2 per prompt (hint end, frame end), `nl` one per option (prompts' nl_off)."""

    def __init__(self, run: Path):
        cache = CACHE / run.name
        self.parts = sorted(cache.glob("part-*")) or [cache]
        self.layers = json.loads((self.parts[0] / "layers.json").read_text())
        self._c = {}

    def __call__(self, site: str, layer: int) -> np.ndarray:
        if (site, layer) not in self._c:
            if site != "nl":  # one layer at a time in memory (a full run's tail is ~5 GB per layer in float32)
                self._c = {k: v for k, v in self._c.items() if k[1] == layer or k[0] == "nl"}
            i = self.layers.index(layer)
            xs = [np.load(p / f"{site}.npy", mmap_mode="r")[:, i] for p in self.parts]
            self._c[(site, layer)] = np.concatenate(xs).astype(np.float32)
        return self._c[(site, layer)]


def _answer_probe(exclude: set[str], l2: float = 1.0) -> dict:
    """The MCQ answer probe refit on the cached MMLU-Pro set (probe_eval's TRAIN) minus `exclude` (every MMLU-Pro question
    here, both pools), balanced right/wrong as probe_eval prep: the disjoint stand-in for the frozen probe."""
    from sklearn.preprocessing import StandardScaler

    from environments.hint_mcq.probe import _fit_listwise
    from environments.hint_mcq.probe_eval import HIDDEN, TRAIN

    qs = [json.loads(line) for line in (TRAIN / "questions.jsonl").read_text().splitlines()]
    keep = [r for r in qs if str(r["item"]) not in exclude]
    right, wrong = [r for r in keep if r["right"]], [r for r in keep if not r["right"]]
    n = min(len(right), len(wrong))
    rng = random.Random(0)
    train = {str(r["item"]) for r in rng.sample(right, n) + rng.sample(wrong, n)}
    opts = [json.loads(line) for line in (TRAIN / "options.jsonl").read_text().splitlines()]
    layers = json.loads((TRAIN / "meta.json").read_text())["params"]["layers"]
    rows = np.array([str(o["item"]) in train for o in opts])
    X = np.asarray(np.load(CACHE / TRAIN.name / "nl.npy", mmap_mode="r")[:, layers.index(HIDDEN)], dtype=np.float64)[rows]
    y = np.array([o["is_answer"] for o in opts], float)[rows]
    q = np.array([o["q"] for o in opts])[rows]
    sc = StandardScaler().fit(X)
    w = _fit_listwise(sc.transform(X), y, np.flatnonzero(np.r_[True, q[1:] != q[:-1]]), l2)
    print(f"answer probe: {len(qs) - len(keep)} of {len(qs)} cached questions excluded, trained on {n} + {n}")
    return {"mean": sc.mean_, "scale": sc.scale_, "w": w, "n_train": 2 * n}


def _auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if 0 < np.sum(y) < len(y) else np.nan


def probe(a):
    """Flip probes on the WITH prompt's activations, baselines, held out by question (GroupKFold); writes probe.md and
    case_preds.jsonl (every case's held-out score under every method)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    from environments.hint_mcq.probe import _fit_listwise, _softmax_by_q

    run = Path(a.run)
    qs = {q["q"]: q for q in map(json.loads, (run / "questions.jsonl").read_text().splitlines())}
    prompts = [json.loads(line) for line in (run / "prompts.jsonl").read_text().splitlines()]
    C = [json.loads(line) for line in (run / "cases.jsonl").read_text().splitlines()]
    held = set(json.loads((run / "split.json").read_text())["eval_questions"]) if (run / "split.json").exists() else set()
    C = [c for c in C if c["q"] not in held]  # the investigator-eval questions: never trained, tuned or scored on here
    # consistent cases only (Kylie, 2026-10-30, as the other hint evals' >= 7/8 samples): both prompts' letter logits say
    # Qwen would give its argmax letter >= 7/8 times (consistent()), so a flip is a change between two consistent answers
    n0 = len(C)
    if not a.all_cases:
        C = [c for c in C if c["consistent"]]
    print(f"{len(C)} of {n0} cases kept ({'all' if a.all_cases else 'consistent on both prompts'})")
    print(f"{len(held)} eval questions held out")
    acts = Acts(run)
    y = np.array([c["flip"] for c in C])
    g = np.array([c["q"] for c in C])
    el = np.array([c["element"] for c in C])
    pw = np.array([c["p_with"] for c in C])
    rng = np.random.default_rng(a.seed)
    # pooled 50:50 flip vs no-flip by subsampling the larger class (Kylie: overall, not per stratum)
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    small, big = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    bal = np.zeros(len(C), bool)
    bal[small] = True
    bal[rng.choice(big, len(small), replace=False)] = True
    pi = y.mean()
    folds = list(GroupKFold(n_splits=a.folds).split(C, groups=g))
    fold = np.zeros(len(C), int)
    for k, (_, te) in enumerate(folds):
        fold[te] = k
    print(f"{len(C)} cases, {y.sum()} flips ({pi:.1%}); balanced set {bal.sum()}")

    def tail(p, layer, j=-1):
        return acts("tail", layer)[p * N_TAIL + (N_TAIL + j if j < 0 else j)]

    def features(variant: str, layer: int, idx) -> np.ndarray:
        p = pw[idx]
        last = tail(p, layer)
        if variant == "last":
            return last
        if variant == "mean":
            return acts("tail", layer).reshape(-1, N_TAIL, last.shape[1])[p].mean(1)
        e = acts("el", layer)[2 * p + np.where(el[idx] == "hint", 0, 1)]
        return np.hstack([last, e])

    def logistic(X, yy, tr, te, c):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=c, max_iter=3000).fit(sc.transform(X[tr]), yy[tr])
        return m.decision_function(sc.transform(X[te]))

    def cv(X, idx, c=None) -> tuple[np.ndarray, float]:
        """Held-out logits for cases idx (X rows aligned with idx); trained on the balanced training-fold cases."""
        out = np.zeros(len(idx))
        if c is None:  # inner choice on the first fold's training cases
            tr0 = np.flatnonzero((fold[idx] != 0) & bal[idx])
            inner = list(GroupKFold(n_splits=3).split(tr0, groups=g[idx][tr0]))
            score = {cc: np.mean([_auc(y[idx][tr0[j]], logistic(X, y[idx], tr0[i], tr0[j], cc)) for i, j in inner]) for cc in a.c_grid}
            c = max(score, key=lambda cc: score[cc])
        for k in range(a.folds):
            tr = np.flatnonzero((fold[idx] != k) & bal[idx])
            te = np.flatnonzero(fold[idx] == k)
            out[te] = logistic(X, y[idx], tr, te, c)
        return out, c

    S, chosen = {}, {}  # method -> held-out score per case (nan where not applicable)
    elements = ["hint", "frame"]

    def put(name, idx, s):
        S.setdefault(name, np.full(len(C), np.nan))[idx] = s

    # hand features from the WITH prompt's letter pmf, the hinted letter, and which stratum the prompt is in
    cw = np.array([[c["p_hint_with"], c["margin_with"], c["entropy_with"], c["ans_with"] == c["hinted"]] for c in C], float)
    strata = sorted({(c["dataset"], c["with"]) for c in C})
    onehot = np.array([[(c["dataset"], c["with"]) == s for s in strata] for c in C], float)
    letters = "ABCDEFGHIJKLMNOP"
    lett = np.array([[c["hinted"] == L for L in letters[:10]] + [c["ans_with"] == L for L in letters[:10]] for c in C], float)
    # the answer probe (refit disjoint from every MMLU-Pro question here) on the WITH prompt's option newlines, layer 52
    ap = _answer_probe({q["qid"] for q in qs.values() if q["dataset"] == "mmlu_pro"})
    nl52 = acts("nl", 52)
    n_opts = np.array([prompts[p]["n_opts"] for p in pw])
    rows = np.concatenate([np.arange(prompts[p]["nl_off"], prompts[p]["nl_off"] + prompts[p]["n_opts"]) for p in pw])
    Zw = (nl52[rows] - ap["mean"]) / ap["scale"]
    st = np.r_[0, np.cumsum(n_opts)[:-1]]
    pmf0 = _softmax_by_q(Zw @ ap["w"], st)
    ans_with = np.array([letters.index(c["ans_with"]) for c in C])
    ans_wo = np.array([letters.index(c["ans_without"]) for c in C])
    top0 = np.array([np.argmax(pmf0[s : s + n]) for s, n in zip(st, n_opts)])
    ent0 = np.array([-(pmf0[s : s + n] * np.log(pmf0[s : s + n] + 1e-12)).sum() for s, n in zip(st, n_opts)])
    hand = np.column_stack([cw, top0 != ans_with, ent0, np.log(n_opts)])
    for e in elements:
        idx = np.flatnonzero(el == e)
        put("base rate", idx, np.zeros(len(idx)))
        put("answer = hinted letter", idx, cw[idx, 3])
        put("-margin", idx, -cw[idx, 1])
        put("entropy", idx, cw[idx, 2])
        put("p(hinted)", idx, cw[idx, 0])
        put("letter only", idx, cv(lett[idx], idx, 1.0)[0])
        put("answer probe mismatch", idx, (top0 != ans_with)[idx] + 1e-3 * ent0[idx])
        put("hand logistic", idx, cv(np.column_stack([hand[idx], onehot[idx]]), idx, 1.0)[0])
        put("arm only", idx, cv(onehot[idx], idx, 1.0)[0])
        # (2) counterfactual-answer probe: listwise from the answer probe, |w - w0|^2, target = the letter without
        t = np.concatenate([np.arange(n) == ans_wo[i] for i, n in zip(idx, n_opts[idx])]).astype(float)
        r_of = {i: np.arange(st[i], st[i] + n_opts[i]) for i in idx}
        loc = {i: j for j, i in enumerate(idx)}
        tro = np.r_[0, np.cumsum(n_opts[idx])[:-1]]
        out = np.zeros(len(idx))
        for k in range(a.folds):
            tr = [i for i in idx if fold[i] != k and bal[i]]
            te = [i for i in idx if fold[i] == k]
            rt = np.concatenate([r_of[i] for i in tr])
            tt = np.concatenate([t[tro[loc[i]] : tro[loc[i]] + n_opts[i]] for i in tr])
            w = _fit_listwise(Zw[rt], tt, np.r_[0, np.cumsum(n_opts[tr])[:-1]], a.l2_answer, w0=ap["w"])
            for i in te:
                p = _softmax_by_q(Zw[r_of[i]] @ w, np.array([0]))
                out[loc[i]] = 1 - p[ans_with[i]]
        put("cf-answer probe", idx, out)
        for layer in LAYERS:
            for variant in ("last", "last+el", "mean"):
                X = features(variant, layer, idx)
                s, c = cv(X, idx, chosen.get((e, variant)))
                chosen[(e, variant)] = c
                put(f"flip probe {variant} L{layer}", idx, s)
                print(e, variant, layer, f"AUROC bal {_auc(y[idx][bal[idx]], s[bal[idx]]):.3f} all {_auc(y[idx], s):.3f}", flush=True)
    # stacked on the best flip probe's held-out logit (chosen as _report does): does it add anything beyond which arm the
    # case is in, and beyond the arm and Qwen's own letter confidence?
    for e in elements:
        idx = np.flatnonzero(el == e)
        flips = {m: s for m, s in S.items() if m.startswith("flip probe") and not m.startswith("flip probe +")}
        best = max(flips, key=lambda m: _auc(y[(el == e) & bal], flips[m][(el == e) & bal]))
        put("flip probe + arm", idx, cv(np.column_stack([S[best][idx], onehot[idx]]), idx, 1.0)[0])
        put("flip probe + arm + hand", idx, cv(np.column_stack([S[best][idx], hand[idx], onehot[idx]]), idx, 1.0)[0])
    _report(a, run, C, S, y, bal, g, el, pi, fold, chosen, ap)
    with (run / f"{a.out}_case_preds.jsonl").open("w") as f:
        for i, c in enumerate(C):
            f.write(
                json.dumps(
                    {k: c[k] for k in ("q", "element", "with", "without", "flip")}
                    | {"fold": int(fold[i]), "balanced": bool(bal[i])}
                    | {m: round(float(s[i]), 4) for m, s in S.items() if not np.isnan(s[i])}
                )
                + "\n"
            )
    if a.transfer:
        transfer(a, run, C, y, bal, g, el, features, logistic, fold)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _report(a, run, C, S, y, bal, g, el, pi, fold, chosen, ap):
    """probe.md: per element and pool, AUROC (balanced set / natural rates) of every method; the best flip probe's gain over
    the hand-feature logistic (bootstrap over questions); the knew-vs-followed subset; per-stratum flip rate vs predicted;
    AUROC within margin quintiles; within-question AUROC; twomodel split by DSv4's correctness."""
    pool = np.array([c["pool"] for c in C])
    rng = np.random.default_rng(a.seed)
    rep = [
        f"# Hint flip probe: {run.name}",
        "",
        f"{len(C)} cases, {y.sum()} flips (natural rate {pi:.1%}); balanced set {bal.sum()} (pooled 50:50).",
    ]
    rep += [
        (
            f"{a.folds}-fold GroupKFold by question; probes trained on the balanced training-fold cases."
            f" Answer probe refit on {ap['n_train']} disjoint questions."
        ),
        "",
    ]
    best = {}
    for e in ("hint", "frame"):
        m_e = el == e
        flips = {m: s for m, s in S.items() if m.startswith("flip probe") and not m.startswith("flip probe +")}
        best[e] = max(flips, key=lambda m: _auc(y[m_e & bal], flips[m][m_e & bal]))
        rep += [f"## Removing the {e}", "", f"Best flip probe by pooled balanced AUROC (chosen on the same folds): **{best[e]}**", ""]
        cols = [("pooled", np.ones(len(C), bool)), ("balanced pool", pool == "balanced"), ("twomodel pool", pool == "twomodel")]
        rep += ["| method | " + " | ".join(f"{n}: bal / natural" for n, _ in cols) + " |", "|" + " --- |" * (len(cols) + 1)]
        show = [m for m in S if not m.startswith("flip probe")] + [
            m for m in S if m.startswith("flip probe") and (m.endswith("L52") or m == best[e] or m.startswith("flip probe +"))
        ]
        for m in dict.fromkeys(show):
            cells = [f"{_auc(y[m_e & bal & c], S[m][m_e & bal & c]):.3f} / {_auc(y[m_e & c], S[m][m_e & c]):.3f}" for _, c in cols]
            rep.append(f"| {m} | " + " | ".join(cells) + " |")
        # gain over the hand logistic, bootstrap over questions, balanced and natural
        for name, mask in (("balanced", m_e & bal), ("natural", m_e)):
            for m in (best[e], "cf-answer probe", "flip probe + arm", "flip probe + arm + hand"):
                qs_ = np.unique(g[mask])
                by_q = {q: np.flatnonzero(mask & (g == q)) for q in qs_}
                d0 = _auc(y[mask], S[m][mask]) - _auc(y[mask], S["hand logistic"][mask])
                ds = []
                for _ in range(a.boot):
                    ii = np.concatenate([by_q[q] for q in rng.choice(qs_, len(qs_))])
                    ds.append(_auc(y[ii], S[m][ii]) - _auc(y[ii], S["hand logistic"][ii]))
                lo, hi = np.nanpercentile(ds, [2.5, 97.5])
                rep.append(f"- {m} − hand logistic, {name}: ΔAUROC {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        rep.append("")
        if e == "hint":
            kv = m_e & np.array([c["ans_with"] == c["hinted"] for c in C])
            rep += [
                (
                    f"Knew vs followed (hint cases where Qwen's answer WITH the hint is the hinted letter: {kv.sum()} cases,"
                    f" {y[kv].mean():.1%} flip):"
                ),
                "",
            ]
            rep += ["| method | pooled | balanced pool | twomodel pool |", "| --- | --- | --- | --- |"]
            for m in ["p(hinted)", "-margin", "letter only", "hand logistic", "answer probe mismatch", "cf-answer probe", best[e]]:
                rep.append(f"| {m} | " + " | ".join(f"{_auc(y[kv & c], S[m][kv & c]):.3f}" for _, c in cols) + " |")
            rep.append("")
        # within-question AUROC: questions with at least one flip and one non-flip among their cases of this element
        rep += ["Within-question AUROC (mean over questions with both outcomes; question easiness cancels):", ""]
        for pn, c in cols:
            per = defaultdict(list)
            for q in np.unique(g[m_e & c]):
                ii = np.flatnonzero(m_e & c & (g == q))
                if 0 < y[ii].sum() < len(ii):
                    for m in (
                        "hand logistic",
                        "arm only",
                        "-margin",
                        "cf-answer probe",
                        best[e],
                        "flip probe + arm",
                        "flip probe + arm + hand",
                    ):
                        per[m].append(_auc(y[ii], S[m][ii]))
            n_q = len(per["hand logistic"])
            rep.append(f"- {pn} ({n_q} questions): " + ", ".join(f"{m} {np.mean(v):.3f}" for m, v in per.items()))
        rep += ["", "AUROC within quintiles of Qwen's letter margin WITH the element (natural rates):", ""]
        marg = np.array([c["margin_with"] for c in C])
        qcut = np.quantile(marg[m_e], [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(marg, qcut)
        rep += ["| quintile | n | flip rate | hand logistic | cf-answer probe | best flip probe |", "|" + " --- |" * 6]
        for b in range(5):
            ii = m_e & (bins == b)
            rep.append(
                f"| {b + 1} | {ii.sum()} | {y[ii].mean():.2f} | "
                + " | ".join(f"{_auc(y[ii], S[m][ii]):.3f}" for m in ("hand logistic", "cf-answer probe", best[e]))
                + " |"
            )
        tm = m_e & (pool == "twomodel")
        if tm.any():
            dr = np.array([bool(c["dsv4_right"]) for c in C])
            rep += ["", "Twomodel pool by DSv4's correctness (natural rates):", ""]
            for v, n in ((True, "DSv4 right"), (False, "DSv4 wrong")):
                ii = tm & (dr == v)
                rep.append(
                    f"- {n} ({ii.sum()} cases, flip {y[ii].mean():.2f}): "
                    + ", ".join(f"{m} {_auc(y[ii], S[m][ii]):.3f}" for m in ("letter only", "hand logistic", "cf-answer probe", best[e]))
                )
        rep.append("")
    # per stratum: actual flip rate vs the best probe's mean predicted p (as trained, 50:50; and prior-corrected to natural)
    rep += [
        "## Per stratum",
        "",
        (
            "Flip rate vs the best flip probe's mean predicted probability (as trained on the 50:50 set, and prior-corrected to"
            " the natural pooled rate); AUROC where a stratum has ≥ 10 of each."
        ),
        "",
    ]
    rep += [
        "| pool | dataset | element | with | without | n | flip rate | mean p (50:50) | mean p (natural) | AUROC | AUROC hand |",
        "|" + " --- |" * 11,
    ]
    groups = defaultdict(list)
    for i, c in enumerate(C):
        groups[(c["pool"], c["dataset"], c["element"], c["with"], c["without"])].append(i)
    shift = np.log(pi / (1 - pi))
    for k, ii in sorted(groups.items()):
        ii = np.array(ii)
        s = S[best[k[2]]][ii]
        ok = 10 <= y[ii].sum() <= len(ii) - 10
        auc = f"{_auc(y[ii], s):.3f}" if ok else ""
        auch = f"{_auc(y[ii], S['hand logistic'][ii]):.3f}" if ok else ""
        rep.append(
            f"| {' | '.join(k)} | {len(ii)} | {y[ii].mean():.2f} | {_sigmoid(s).mean():.2f} | {_sigmoid(s + shift).mean():.2f}"
            f" | {auc} | {auch} |"
        )
    rep += ["", "## Chosen C (logistic, inner 3-fold on fold 0's training cases)", "", *[f"- {k}: {v}" for k, v in chosen.items()]]
    (run / f"{a.out}.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


def transfer(a, run, C, y, bal, g, el, features, logistic, fold):
    """(3) Train the flip probe on one group of cases, test on another (held out by question, both sides' folds): hint <->
    frame removal (the `last` token only, the element token differs), professor <-> Wikipedia, after <-> before (hint),
    MMLU-Pro <-> TruthfulQA."""
    attr = {k: np.array([c[k] for c in C]) for k in ("source", "position", "dataset")}
    pairs = [
        ("hint -> frame", el == "hint", el == "frame", "last"),
        ("frame -> hint", el == "frame", el == "hint", "last"),
        ("prof -> wiki (hint)", (el == "hint") & (attr["source"] == "prof"), (el == "hint") & (attr["source"] == "wiki"), "last+el"),
        ("wiki -> prof (hint)", (el == "hint") & (attr["source"] == "wiki"), (el == "hint") & (attr["source"] == "prof"), "last+el"),
        (
            "after -> before (hint)",
            (el == "hint") & (attr["position"] == "after"),
            (el == "hint") & (attr["position"] == "before"),
            "last+el",
        ),
        (
            "before -> after (hint)",
            (el == "hint") & (attr["position"] == "before"),
            (el == "hint") & (attr["position"] == "after"),
            "last+el",
        ),
    ]
    for e in ("hint", "frame"):
        mm, tq = attr["dataset"] == "mmlu_pro", attr["dataset"] == "truthfulqa"
        pairs += [
            (f"mmlu_pro -> truthfulqa ({e})", (el == e) & mm, (el == e) & tq, "last+el"),
            (f"truthfulqa -> mmlu_pro ({e})", (el == e) & tq, (el == e) & mm, "last+el"),
        ]
    rep = [
        f"# Transfer: {run.name}",
        "",
        (
            f"Flip probe at hidden_states[{a.transfer_layer}], C {a.transfer_c}; AUROC on the target group (balanced / natural),"
            " vs the same probe trained in-group."
        ),
        "",
    ]
    rep += ["| train -> test | variant | transfer bal / natural | in-group bal / natural |", "| --- | --- | --- | --- |"]
    for name, A, B, variant in pairs:
        idx = np.flatnonzero(A | B)
        X = features(variant, a.transfer_layer, idx)
        loc = {i: j for j, i in enumerate(idx)}
        out, own = np.full(len(C), np.nan), np.full(len(C), np.nan)
        for k in range(a.folds):
            te = np.flatnonzero(B & (fold == k))
            for src, dst in ((A, out), (B, own)):
                tr = np.flatnonzero(src & (fold != k) & bal)
                dst[te] = logistic(X, y[idx], [loc[i] for i in tr], [loc[i] for i in te], a.transfer_c)
        cells = [f"{_auc(y[B & bal], s[B & bal]):.3f} / {_auc(y[B], s[B]):.3f}" for s in (out, own)]
        rep.append(f"| {name} | {variant} | {cells[0]} | {cells[1]} |")
        print(rep[-1], flush=True)
    (run / f"{a.out}_transfer.md").write_text("\n".join(rep) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--slug", default="hint-flip")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", type=int, help="only the first N questions per (pool, dataset)")
    s = sub.add_parser("labels")
    s.add_argument("run")
    h = sub.add_parser("split")
    h.add_argument("run")
    h.add_argument("--n", type=int, default=1000, help="eval removal cases, half flips")
    h.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("probe")
    t.add_argument("run")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--all-cases", action="store_true", help="no consistency filter (the secondary table)")
    t.add_argument("--out", default="probe", help="report name: OUT.md, OUT_case_preds.jsonl")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--boot", type=int, default=1000)
    t.add_argument("--c-grid", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    t.add_argument("--l2-answer", type=float, default=1.0, help="|w - w0|^2 weight of the cf-answer probe (PR #369's choice)")
    t.add_argument("--transfer", action="store_true")
    t.add_argument("--transfer-layer", type=int, default=52)
    t.add_argument("--transfer-c", type=float, default=1e-3)
    a = ap.parse_args()
    {"prep": prep, "labels": labels, "split": split, "probe": probe}[a.cmd](a)


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        main()
