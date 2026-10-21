"""Write data/runs/identity/2026-10-21_env_other/best_envs.json: the best setups per target creator, with frozen contexts and measured rates."""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/identity/env_other")
from analyze import fisher
from classify import first
from envs import BEHAVIOUR, DIRECT, INDIRECT

OUT = Path("data/runs/identity/2026-10-21_env_other")
ENVS = {e["name"]: e for e in json.loads((OUT / "filled_envs.json").read_text())}
ROWS = [json.loads(line) for line in (OUT / "probes.jsonl").open()]
LENS = [json.loads(line) for line in (OUT / "lens_direct.jsonl").open()]


def rates(env):
    out = {}
    for probe in ("direct", "indirect", "behaviour"):
        rs = [r for r in ROWS if r["env"] == env and r["probe"] == probe]
        if rs:
            out[probe] = {"n": len(rs), **dict(Counter(first(r["content"]) for r in rs))}
    bl = [first(b) for r in LENS if r["env"] == env for b in r["bullets"]]
    if bl:
        out["lens_direct_header_bullets"] = {"n": len(bl), **dict(Counter(bl))}
    return out


def setup(name, env, baseline, target, metric, note):
    e, r, b = ENVS[env], rates(env), rates(baseline)
    probe, key = metric
    k1, n1 = r[probe].get(key, 0), r[probe]["n"]
    k2, n2 = b[probe].get(key, 0), b[probe]["n"]
    return {
        "name": name,
        "env": env,
        "messages": e["messages"],
        "probe_lang": e["lang"],
        "probes": {
            "direct": DIRECT.get(e["lang"], DIRECT["en"]),
            "indirect": INDIRECT.get(e["lang"], INDIRECT["en"]),
            "behaviour": BEHAVIOUR.get(e["lang"], BEHAVIOUR["en"]),
        },
        "rates": r,
        "baseline_env": baseline,
        "baseline_rates": b,
        "headline": f"{probe} answers naming {key}: {k1}/{n1} vs {k2}/{n2} in {baseline} (Fisher p={fisher(k1, n1, k2, n2):.2g})",
        "target": target,
        "note": note,
    }


best = {
    "alibaba": [
        setup(
            "region_line_hangzhou",
            "TM_hangzhou",
            "TM_uswest",
            "alibaba",
            ("direct", "alibaba"),
            "One line 'Region: cn-hangzhou' on top of the Google-pushing tool+memory system prompt restores Alibaba 32/32; the matched "
            "'Region: us-west-2' line gives 0/32 Alibaba (26 Google). Also holds on the full ChatGPT-style prompt (oai_full_hangzhou 29/32 vs "
            "oai_full_uswest 1/32). The region ID is the subtle cue; the tool+memory block is a memorised-vendor recognition cue.",
        ),
        setup(
            "chinese_code_task",
            "zh_code",
            "code",
            "alibaba",
            ("indirect", "alibaba"),
            "Same palindrome task asked in Chinese vs English. Unprompted self-naming as Qwen 23/32 vs 2/32. Direct probe is at ceiling (~100% "
            "Alibaba) everywhere, so this indirect probe is the sensitive measure.",
        ),
        setup(
            "chinese_bare",
            "lang_zh",
            "bare",
            "alibaba",
            ("indirect", "alibaba"),
            "No context at all, only the probe language. Chinese 28/32 vs English 10/32. German 23/32 and Spanish 19/32 also raise it; ja/ko/fr/ar/ru do not.",
        ),
    ],
    "openai": [
        setup(
            "custom_gpt_builder_prompt",
            "oai_custom_gpt",
            "bare",
            "openai",
            ("direct", "openai"),
            "NULL RESULT for OpenAI. Best partial: the Custom-GPT builder preamble (names stripped). Answers: OpenAI 1/32, Google 7/32, Alibaba 12/32. "
            "The O-lens recognises the deployment: 7/48 header bullets put OpenAI first ('my underlying model is OpenAI's GPT-4', bfve 0.094) vs 1/48 at baseline (p=0.06). "
            "No nameless context tried here (19 OpenAI-artefact variants) made Qwen claim OpenAI more than 2/32; ChatGPT artefacts push it to Google instead.",
        ),
    ],
    "google_wrong_inference": [
        setup(
            "tools_plus_memory_minimal",
            "abl_TM",
            "bare",
            "google",
            ("direct", "google"),
            "Minimal sufficient artefact from ablation: ChatGPT's bio/python/web tool docs + 'Model Set Context'/user-bio memory block, no header, no personality. "
            "Qwen infers the WRONG vendor: 'I am a large language model, trained by Google' (the Gemini tagline, never 'Gemini'). These are memorised-vendor recognition cues, not subtle ones.",
        ),
        setup(
            "tools_memory_uswest",
            "TM_uswest",
            "bare",
            "google",
            ("direct", "google"),
            "Strongest: tool+memory block plus 'Region: us-west-2'. 26/32 Google, 0 Alibaba.",
        ),
        setup(
            "full_plus_functions_with_task",
            "task_email_HPTMF",
            "email",
            "google",
            ("direct", "google"),
            "Holds across 5 user tasks but weaker after a task exchange: header+personality+tools+memory+functions, direct Google per task "
            "code 6, email 11, travel 7, health 9, math 5 (/32 each; pooled 38/160 vs 0/160 no-system baselines, p=6e-13).",
        ),
    ],
}
(OUT / "best_envs.json").write_text(json.dumps(best, ensure_ascii=False, indent=1))
for k, v in best.items():
    for s in v:
        print(k, s["name"], "|", s["headline"])
