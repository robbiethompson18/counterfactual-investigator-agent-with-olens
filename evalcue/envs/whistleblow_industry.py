"""SnitchBench's setup (Theo Browne, 2025) with the wrongdoing moved across industries: pharma (SnitchBench's own documents),
finance (a fund hiding $2.3B of losses from retirees), environment (a plant poisoning a town's water) and privacy (a health app
hiding a breach and selling clinic-visit locations). Each industry also has a "minor" version where the concealment is
backdated paperwork with no harm. Bash agent with SnitchBench's email tools. See docs/envs/whistleblow.md.
"""

from evalcue.envs import whistleblow_common as wc
from evalcue.harness import Episode

INDUSTRIES = ["pharma", "finance", "environment", "privacy"]
ARMS = [f"{ind}_{bold}_{sev}" for ind in INDUSTRIES for bold in ("tame", "bold") for sev in ("severe", "minor")]
ARMS += [f"{ind}_tame_severe_cli" for ind in INDUSTRIES]
# batch 2 (r6, server 1): the levers that moved whistleblow_single, per industry
ARMS += [f"{ind}_{k}" for ind in INDUSTRIES for k in ("tame_severe+directory", "tame_severe+values_only", "bold_severe+hotline")]
ARMS += [f"{ind}_bold_minor+hotline" for ind in INDUSTRIES]


def items():
    return ["main"]


def build(arm, item):
    base, *extra = arm.split("+")
    industry, boldness, severity, *rest = base.split("_")
    tools = "cli" if rest == ["cli"] else "email"
    sc = wc.SCENARIOS[industry]
    knobs = (["minor"] if severity == "minor" else []) + extra
    system = wc.system_prompt(sc, boldness, tools, knobs, wc.TOOLS_EMAIL if tools == "email" else wc.TOOLS_CLI)
    user = wc.user_message(sc, knobs, wc.documents(sc, knobs))
    return Episode(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        files=wc.sandbox_files(tools),
        cwd=wc.HOME,
        env={"HOME": wc.HOME},
        max_turns=20,
        meta={"company_domains": list(sc.domains), "tools": tools, "boldness": boldness, "knobs": knobs, "industry": industry},
    )


def detect(row):
    return wc.detect_agent(row)
