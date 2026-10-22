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


def items():
    return ["main"]


def build(arm, item):
    industry, boldness, severity, *rest = arm.split("_")
    tools = "cli" if rest == ["cli"] else "email"
    sc = wc.SCENARIOS[industry]
    knobs = ["minor"] if severity == "minor" else []
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
