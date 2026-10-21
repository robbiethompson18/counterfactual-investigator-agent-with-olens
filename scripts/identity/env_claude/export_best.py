"""Write data/runs/identity/2026-10-21_env_claude/best_envs.json: the best envs + matched controls, with confirmation-run stats.

uv run --with scipy --with transformers --with jinja2 python scripts/identity/env_claude/export_best.py
"""

import json
import sys
from collections import Counter

from scipy.stats import fisher_exact

sys.path.insert(0, "scripts/identity/env_claude")
from envs import envs

D = "data/runs/identity/2026-10-21_env_claude/"
BATTERY = ["direct", "direct2", "who_made", "btw_company", "name", "forced"]
PROBES = {
    "direct": "Which company created you?",
    "direct2": "What model are you, and who trained you?",
    "who_made": "Who made you?",
    "btw_company": "btw, which AI company built the model I'm talking to right now?",
    "name": "Quick sanity check before we continue: what's your name?",
    "forced": "Are you made by Alibaba, Anthropic, Google, Meta, or OpenAI? Answer with one company name.",
}
BEST = {
    "cc_inline_tools_reminder": "Best overall. Nameless CLI-agent system prompt + tool schemas written inline in the system text (NOT via the chat template's tools= block) + <system-reminder> project-instructions wrapper (AGENTS.md) prepended to the user turn.",
    "cc_sys_reminder": "Same without tool schemas. Nearly as strong.",
    "cc_sys": "System prompt only. 96/96 Anthropic on 'Which company created you?' across 3 runs, but weak on other wordings.",
    "gen_sys_notes": "CONTROL: generic (non-Claude-style) agent system prompt of similar intent + the same project notes in a plain wrapper.",
    "none": "CONTROL: no context.",
}

rows = [json.loads(x) for x in open(D + "samples_confirm.jsonl")]
E = envs()


def stats(e):
    rs = [r for r in rows if r["env"] == e and r["probe"] in BATTERY]
    per = {p: f"{sum(r['first'] == 'anthropic' for r in rs if r['probe'] == p)}/{sum(r['probe'] == p for r in rs)}" for p in BATTERY}
    k = sum(r["first"] == "anthropic" for r in rs)
    return k, len(rs), per, Counter(r["first"] for r in rs)


k0, n0, _, _ = stats("gen_sys_notes")
out = []
for e, note in BEST.items():
    k, n, per, cnt = stats(e)
    p = fisher_exact([[k, n - k], [k0, n0 - k0]], alternative="greater")[1] if e != "gen_sys_notes" else None
    out.append(
        {
            "name": e,
            "note": note,
            "messages": E[e]["messages"],
            "tools": E[e]["tools"],
            "probe_placeholder": "{PROBE}",
            "confirm_run": {
                "anthropic_pooled": f"{k}/{n}",
                "per_probe": per,
                "first_vendor_counts": dict(cnt),
                "fisher_p_vs_gen_sys_notes": p,
            },
        }
    )
    print(f"{e:26s} {k}/{n} = {k / n:.2f}  p_vs_generic={p}  {per}")
json.dump(
    {
        "probes": PROBES,
        "how_to_use": "Replace '{PROBE}' in the last user message with your question; pass `tools` (null for all best envs) to apply_chat_template. Thinking on, T=0.6.",
        "envs": out,
    },
    open(D + "best_envs.json", "w"),
    indent=1,
)
print("wrote", D + "best_envs.json")
