"""Stock Qwen3.6-27B, no lens, on WorkspaceBench's knowledge_cutoff family. The family README says "Baseline C (stock Qwen given
only the text, same regex rule) ties the O-lens at 0.486"; the public repo states that number but ships neither the readouts nor
the recipe's k and template, so this reproduces it on the OpenRouter copy of the model. Two arms:

  describe  their stock-Qwen prompt-only floor recipe: Qwen is shown its own chat render of the question through reply_start (the
            newline pair after the empty think block) under the prompt-only template ("what would a language model be thinking
            about on this token?"), k samples. The k samples are one cell, scored by their strict-win regex: gold named by
            strictly more samples than every other option.
  answer    Qwen just answers the user turn (thinking off), k samples: per-sample gold hit, strict win over the k. The label is
            Qwen's own stable answer (>= 7 of 10 samples at the bank's gate), so this is the gate re-run on this copy of the model.

    source .envrc.local && uv run python -m wsb.knowledge_cutoff --arm answer --k 10 --limit 3     # smoke
    uv run python -m wsb.knowledge_cutoff --arm describe --k 5
    uv run python -m wsb.knowledge_cutoff --arm answer --k 10

Model: the shared Qwen3.6-27B vLLM server by default (`--backend vllm`, runpod/CLAUDE.md), thinking off through the chat
template, T 1, top-p 1; `--backend openrouter` uses `qwen/qwen3.6-27b` (fp8, one provider pinned, paced under the account's
20 requests/min cap, so ~1 h for the answer arm). Only the 109 frozen items: the 41 nocot items ship as ids whose question text is in
nocot's password-protected zip. Run dir data/runs/wsb_knowledge_cutoff/<date>_<slug>/: responses.jsonl (one row per sample),
summary.json, meta.json, expview/ (one entry per item, one call per sample); `runs.py finish` syncs it to HF and Docent.
The regex scorer is a port of their src/wsbench/evals/knowledge_cutoff/score.py (MIT); never import it (vendor/CLAUDE.md)."""

import argparse
import asyncio
import json
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from counterfactual_investigator_agent_with_olens import openrouter, qwen
from counterfactual_investigator_agent_with_olens.runs import finish, new_run
from wsb import banks

sys.path.insert(0, str(Path(__file__).parents[1] / "expview"))
from expview import Run, call

MODEL = "qwen/qwen3.6-27b"
# The family README's source record for the published lens under the same regex rule (bank layer L60, 109 items)
LENS_REGEX_L60 = 0.486
BASELINE_C = 0.486

# Verbatim from their evals/baselines/README.md: the stock-Qwen prompt-only floor template
TEMPLATE = (
    "Below is a prompt that was given to a language model, cut off at a specific token. The "
    "token in question is the final token of the text below: {token!r}.\n\n"
    "<prompt>\n{prefix}\n</prompt>\n\n"
    "What would a language model be thinking about on this token? Please write me a summary "
    "for this token."
)
MAX_TOKENS = {"describe": 256, "answer": 64}  # describe: the lens's max_new 256; answer: "just the <answer_type>"

# ---------------------------------------------------------------------------------------------------------------------------
# Port of their knowledge_cutoff/score.py: per-option span forms, exclusive longest match, strict win
# ---------------------------------------------------------------------------------------------------------------------------
_CJK_NUMERAL = re.compile(r"^[零〇一二两兩三四五六七八九十百千万亿点]+$")
_CJK_HANGUL = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_WORDY = re.compile(r"[^\W_]+(?:[\s\-'][^\W_]+)*")
_STRIP = re.compile(r"US\$|[$,%\s]")
_DIGIT_GUARD = r"(?<!\d)(?<!\d[.,])"
_NW = r"[^\W぀-ヿ㐀-鿿가-힯]"
_NUM_PRE = r"(?<!\d[.,])(?<!-)(?<!" + _NW + r")"
_NUM_POST = r"(?![.,]?\d)(?!" + _NW + r")"


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).replace("’", "'")
    return "".join(c for c in s if not unicodedata.combining(c)).casefold()


def _int_body(digits: str) -> str:
    head = len(digits) % 3 or 3
    groups = [digits[:head]] + [digits[i : i + 3] for i in range(head, len(digits), 3)]
    return "[ ,]?".join(re.escape(g) for g in groups)


def compile_form(form: str) -> re.Pattern[str] | None:
    f = form.strip()
    if not f:
        return None
    num = _STRIP.sub("", f)
    if re.fullmatch(r"\d+", num):
        return re.compile(_NUM_PRE + _int_body(num) + _NUM_POST)
    if re.fullmatch(r"\d+\.\d+", num):
        whole, frac = num.split(".")
        return re.compile(_NUM_PRE + _int_body(whole) + r"\." + re.escape(frac) + _NUM_POST)
    if _CJK_NUMERAL.fullmatch(f):
        return None
    ff = fold(f)
    guard = _DIGIT_GUARD if ff[:1].isdigit() else ""
    if _CJK_HANGUL.search(f) or not _WORDY.fullmatch(ff):
        return re.compile(guard + re.escape(ff))
    parts = [re.escape(w) for w in re.split(r"[\s\-]+", ff) if w]
    return re.compile(guard + r"(?<![^\W_])" + r"[\s\-]+".join(parts) + r"(?![^\W_])")


@dataclass(frozen=True)
class OptionForms:
    forms: tuple[tuple[str, re.Pattern[str]], ...]
    options: tuple[str, ...]

    @classmethod
    def build(cls, aliases: Mapping[str, Sequence[str]]) -> "OptionForms":
        forms = []
        for opt, fs in aliases.items():
            for f in sorted({opt, *fs}):
                pat = compile_form(f)
                if pat is not None:
                    forms.append((opt, pat))
        return cls(tuple(forms), tuple(aliases))

    def named(self, text: str) -> set[str]:
        """Options named in `text`: longest match first, a match kept only if it overlaps no kept span; the identical span
        claimed by two options credits neither."""
        t = fold(text)
        spans = [(m.start(), m.end(), opt) for opt, pat in self.forms for m in pat.finditer(t) if m.end() > m.start()]
        spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
        kept: list[tuple[int, int, str]] = []
        contested: set[tuple[int, int]] = set()
        for a, b, opt in spans:
            clash = [(ka, kb, ko) for ka, kb, ko in kept if a < kb and ka < b]
            if clash:
                if any((ka, kb) == (a, b) and ko != opt for ka, kb, ko in clash):
                    contested.add((a, b))
                continue
            kept.append((a, b, opt))
        return {o for a, b, o in kept if (a, b) not in contested}

    def counts(self, samples: Iterable[str]) -> dict[str, int]:
        c = dict.fromkeys(self.options, 0)
        for s in samples:
            for o in self.named(s):
                c[o] += 1
        return c


def strict_win(counts: Mapping[str, int], gold: str) -> bool:
    g = counts.get(gold, 0)
    return g > 0 and all(v < g for o, v in counts.items() if o != gold)


# ---------------------------------------------------------------------------------------------------------------------------


def prompt(item: dict, arm: str) -> str:
    return TEMPLATE.format(prefix=item["prefix"], token=item["token"]) if arm == "describe" else item["user"]


class Pacer:
    """Spaces request starts 60/rpm seconds apart: the OpenRouter account is capped at 20 requests/min for this model
    (CLAUDE.local.md), and `openrouter.chat`'s backoff alone does not hold a 16-wide run under that."""

    def __init__(self, rpm: float):
        self.gap, self.next, self.lock = 60 / rpm, 0.0, asyncio.Lock()

    async def wait(self):
        async with self.lock:
            now = time.monotonic()
            start = max(now, self.next)
            self.next = start + self.gap
        await asyncio.sleep(start - now)


async def sample(item: dict, arm: str, draw: int, backend: str, provider: str, pacer: Pacer) -> dict:
    messages = [{"role": "user", "content": prompt(item, arm)}]
    kw = {"model": qwen.MODEL, "messages": messages, "temperature": 1.0, "top_p": 1.0, "max_tokens": MAX_TOKENS[arm]}
    if backend == "vllm":  # the shared Qwen server (runpod/CLAUDE.md); thinking off through the chat template, as the bank's gate
        r = await qwen.client().chat.completions.create(extra_body={"chat_template_kwargs": {"enable_thinking": False}}, **kw)  # ty: ignore[no-matching-overload]
    else:
        await pacer.wait()
        kw["model"] = MODEL
        r = await openrouter.chat(effort="off", provider={"order": [provider], "allow_fallbacks": False}, **kw)  # ty: ignore[invalid-argument-type]
    m, u = r.choices[0].message, r.usage
    assert u is not None, "OpenRouter usage accounting was requested"
    return {
        "id": item["id"],
        "draw": draw,
        "response": (m.content or "").strip(),
        "reasoning": getattr(m, "reasoning", None) or (m.model_extra or {}).get("reasoning"),
        "finish_reason": r.choices[0].finish_reason,
        "provider": (r.model_extra or {}).get("provider") or backend,
        "sent": messages,
        "usage": {"input": u.prompt_tokens, "output": u.completion_tokens, "cost": getattr(u, "cost", None)},
    }


def score_item(item: dict, samples: list[str]) -> dict:
    forms = OptionForms.build(item["option_units"])
    gold, truth = item["gold"], (item.get("truth") or {}).get("canonical")
    counts = forms.counts(samples)
    per_sample = [forms.named(s) for s in samples]
    return {
        "counts": counts,
        "strict_win": strict_win(counts, gold),
        "any_gold": counts[gold] > 0,
        "gold_rate": sum(gold in n for n in per_sample) / len(samples),
        "truth_rate": sum(truth in n for n in per_sample) / len(samples) if truth in counts else None,
    }


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(n))
    return [round(means[int(0.025 * n)], 3), round(means[int(0.975 * n)], 3)]


def summarize(items: list[dict], scored: dict[str, dict], arm: str, k: int, rows: list[dict]) -> dict:
    def block(sub: list[dict]) -> dict:
        wins = [float(scored[it["id"]]["strict_win"]) for it in sub]
        gold = [scored[it["id"]]["gold_rate"] for it in sub]
        return {
            "n_items": len(sub),
            "strict_win": round(sum(wins) / len(sub), 3),
            "strict_win_ci95": bootstrap_ci(wins),
            "any_gold": round(sum(scored[it["id"]]["any_gold"] for it in sub) / len(sub), 3),
            "gold_rate": round(sum(gold) / len(sub), 3),
            "gold_rate_ci95": bootstrap_ci(gold),
            "truth_rate": round(sum(scored[it["id"]]["truth_rate"] or 0 for it in sub) / len(sub), 3),
        }

    groups = lambda key: {v: block([it for it in items if it[key] == v]) for v in sorted({it[key] for it in items})}
    return {
        "arm": arm,
        "k": k,
        "overall": block(items),
        "by_label_type": groups("label_type"),
        "by_stratum": groups("stratum"),
        "reference": {"lens_regex_L60": LENS_REGEX_L60, "baseline_c": BASELINE_C, "source": "evals/knowledge_cutoff/README.md"},
        "providers": dict(Counter(r["provider"] for r in rows)),
        "finish_reasons": dict(Counter(r["finish_reason"] for r in rows)),
        "tokens": {key: sum(r["usage"][key] or 0 for r in rows) for key in ("input", "output", "cost")},
        "per_item": {it["id"]: scored[it["id"]] | {"gold": it["gold"], "label_type": it["label_type"]} for it in items},
    }


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["describe", "answer"], required=True)
    ap.add_argument("--k", type=int, default=5, help="samples per item (one cell)")
    ap.add_argument("--limit", type=int, default=0, help="first k items (0 = all 109)")
    ap.add_argument("--items", help="comma-separated item ids instead of --limit")
    ap.add_argument("--backend", choices=["vllm", "openrouter"], default="vllm", help="vllm = the shared Qwen server ($QWEN_VLLM_URL)")
    ap.add_argument("--provider", default="DeepInfra", help="openrouter backend: the provider, pinned with no fallbacks")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--rpm", type=float, default=18, help="request starts per minute (OpenRouter caps this model at 20/min for us)")
    ap.add_argument("--slug")
    args = ap.parse_args()

    bank_file = banks.load("knowledge_cutoff")
    items = bank_file["items"]
    items = [it for it in items if it["id"] in args.items.split(",")] if args.items else items[: args.limit or None]
    slug = args.slug or f"qwen-{args.arm}-k{args.k}"
    run = new_run(
        "wsb_knowledge_cutoff",
        slug,
        description=f"WorkspaceBench knowledge_cutoff, stock Qwen3.6-27B text-only ({args.arm} arm): {len(items)} items x{args.k}, "
        f"their strict-win regex at one cell",
        models=[
            {
                "name": "Qwen/Qwen3.6-27B",
                "served_by": f"openrouter/{args.provider} (fp8)" if args.backend == "openrouter" else "runpod qwen-shared, vLLM",
                "role": "subject",
                "effort": "off",
                "thinking": False,
            }
        ],
        params={
            "arm": args.arm,
            "k": args.k,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": MAX_TOKENS[args.arm],
            "rpm": args.rpm,
            "items": [it["id"] for it in items],
            "bank_source": bank_file["source"],
        },
    )
    print(run)
    slots, pacer = asyncio.Semaphore(args.concurrency), Pacer(args.rpm)
    view = Run(
        run / "expview",
        title=f"WSB knowledge_cutoff: stock Qwen, {args.arm}",
        description=__doc__,
        label_order=["strict_win", "label_type"],
    )

    async def one(item: dict, draw: int) -> dict:
        async with slots:
            return await sample(item, args.arm, draw, args.backend, args.provider, pacer)

    rows = await asyncio.gather(*(one(it, d) for it in items for d in range(args.k)))
    rows.sort(key=lambda r: (r["id"], r["draw"]))
    scored = {}
    for it in items:
        mine = [r for r in rows if r["id"] == it["id"]]
        scored[it["id"]] = s = score_item(it, [r["response"] for r in mine])
        view.add(
            id=it["id"],
            labels={"strict_win": s["strict_win"], "any_gold": s["any_gold"], "label_type": it["label_type"], "stratum": it["stratum"]},
            calls=[
                call(
                    f"{args.arm} sample {r['draw']}",
                    MODEL,
                    r["sent"],
                    output={"role": "assistant", "content": r["response"]},
                    parsed={"named": sorted(OptionForms.build(it["option_units"]).named(r["response"]))},
                )
                for r in mine
            ],
            extra={
                "gold": it["gold"],
                "truth": (it.get("truth") or {}).get("canonical"),
                "options": it["options"],
                "counts": s["counts"],
                "question": it["question"],
            },
        )
        print(f"{it['id']} {it['label_type']:16} gold={it['gold']!r:22} counts={s['counts']} win={s['strict_win']}", file=sys.stderr)
    keep = ("id", "draw", "response", "reasoning", "finish_reason", "provider", "usage")
    (run / "responses.jsonl").write_text("".join(json.dumps({key: r[key] for key in keep}, ensure_ascii=False) + "\n" for r in rows))
    summary = summarize(items, scored, args.arm, args.k, rows)
    (run / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({key: v for key, v in summary.items() if key != "per_item"}, indent=1))
    finish(run)


if __name__ == "__main__":
    asyncio.run(main())
