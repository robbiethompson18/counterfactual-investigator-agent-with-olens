"""summary.json (+ the page's prose in text.json) -> one self-contained HTML page.

python scripts/grader_reads/build_page.py --scan data/runs/grader_reads/<run> --text scripts/grader_reads/text.json -o page.html
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent

SHOW = [
    "IB realism ladder",
    "IB realism variants (round 1)",
    "ImpossibleBench (original)",
    "email (several Johns)",
    "Funding Email (Claude Code harness)",
    "overeager/payback",
    "overeager/dana",
    "overeager/calendar",
    "overeager/gitbranch",
    "overeager/email",
    "overeager/email_4johns",
    "Donation Bet",
    "Donation Bet (empty think)",
    "agentic misalignment",
    "alignment faking (no CoT)",
    "alignment faking (thinking)",
]
NAMES = {
    "IB realism ladder": "IB criterion-free ladder",
    "IB realism variants (round 1)": "IB realism variants (round 1)",
    "ImpossibleBench (original)": "ImpossibleBench (benchmark prompt)",
    "email (several Johns)": "Email several Johns",
    "Funding Email (Claude Code harness)": "Funding Email, Claude Code harness",
    "overeager/payback": "Overeager: payback",
    "overeager/dana": "Overeager: Dana",
    "overeager/calendar": "Overeager: calendar",
    "overeager/gitbranch": "Overeager: git branch",
    "overeager/email": "Overeager: email",
    "overeager/email_4johns": "Overeager: email, 4 Johns",
    "Donation Bet": "Donation Bet",
    "Donation Bet (empty think)": "Donation Bet, empty think",
    "agentic misalignment": "Agentic misalignment",
    "alignment faking (no CoT)": "Alignment faking, no CoT",
    "alignment faking (thinking)": "Alignment faking, thinking",
}
LADDER = ["L1_rl_cues", "L2_rl", "L3_rl_nodone", "L4_assistant_formal", "L5_assistant_casual", "L6_casual_ide", "L7_repo", "L8_full"]
EMAIL = [
    "base_rl@low", "base@low", "base_single@low", "base_typo@low", "base_home@low", "base_done_neutral@low", "no_done@low", "harness@low",
    "no_criterion@low", "plain_request@low", "asst_formal@low", "core_noreply@low", "nc_casual@low", "core@low",
]  # fmt: skip
OE_A = ["base_no_criterion", "nc_done_neutral", "nc_done_plain", "nc_done_home", "no_criterion"]
OE_B = ["base", "base_done_neutral", "base_plain", "base_home", "no_done"]
FUND = [f"{h}_{d}_{m}" for h in ("upstream", "core", "full") for d in ("thin", "real", "realask") for m in ("a", "b")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("-o", required=True)
    a = ap.parse_args()
    d = json.loads((Path(a.scan) / "analysis" / "summary.json").read_text())
    d["text"] = json.loads(Path(a.text).read_text())
    d["show_envs"] = [e for e in SHOW if any(x["env"] == e for x in d["envs"])]
    d["env_names"] = NAMES
    d["meaning"] = {x["env"]: x["meaning"] for x in d["acted"]}
    # email: pool each arm over its contact books (@low, @lowc, @lowa, fresh, ...): parts 1-3 read initial turns only, the later
    # parts read whole episodes, so pooling gives every rung model-side reads
    pooled = {}
    for x in d["arms"]:
        if x["env"] != "email (several Johns)":
            continue
        stem = x["arm"].split("@")[0]
        q = pooled.setdefault(stem, {"env": x["env"], "arm": stem + " (all books)", "evalness": None, "ladder_label": None, "_pos": 0.0})
        for k in ("m_n", "m_k", "m_s", "t_n", "t_k", "n_eps", "n_eps_read", "k_eps", "k_eps_s", "n_lab"):
            q[k] = q.get(k, 0) + (x.get(k) or 0)
        for k in ("p_n", "p_k", "p_s"):  # prompt side: the most any one book's prompt drew (books differ only in contacts)
            q[k] = max(q.get(k, 0), x.get(k) or 0)
        q["_pos"] += (x.get("pos_rate") or 0) * (x.get("n_lab") or 0)
        if x["evalness"] is not None:
            q["evalness"] = x["evalness"]
        q.setdefault("dss", [])
        q.setdefault("groups", [])
        q["dss"] += [v for v in x.get("dss") or [] if v not in q["dss"]]
        q["groups"] += x.get("groups") or []
    for q in pooled.values():
        q["pos_rate"] = q.pop("_pos") / q["n_lab"] if q["n_lab"] else None
        d["arms"].append(q)
    arms = {(x["env"], x["arm"]): x for x in d["arms"]}
    ib1 = sorted(
        [x for x in d["arms"] if x["env"] == "IB realism variants (round 1)" and x["evalness"] is not None], key=lambda x: -x["evalness"]
    )
    panels = [
        {
            "env": "IB realism ladder",
            "title": "ImpossibleBench, criterion-free ladder",
            "sub": "L1 most eval-like → L8 most deployment-like; base / nocrit_done = lcb.yaml with and without the criterion",
            "arms": ["base", "nocrit_done", *LADDER],
            "beh": "hack",
        },
        {
            "env": "IB realism variants (round 1)",
            "title": "ImpossibleBench realism variants (round 1)",
            "sub": "sorted by judged evalness (Bradley–Terry), most eval-like first; ~32 reads per episode",
            "arms": [x["arm"] for x in ib1],
            "beh": "hack",
        },
        {
            "env": "email (several Johns)",
            "title": "Email several Johns",
            "sub": "sorted by judged evalness, then the in-between rungs; model side pooled over each arm's contact-book runs, "
            "prompt side = the most any one run drew; picks-one rate depends heavily on the book",
            "arms": [a.split("@")[0] + " (all books)" for a in EMAIL],
            "beh": "picks one",
        },
        {
            "env": "Funding Email (Claude Code harness)",
            "title": "Funding Email in the Claude Code harness",
            "sub": "harness upstream → core → full × docs thin / real / real ask × memo a/b",
            "arms": FUND,
            "beh": "positive",
        },
        {
            "env": "ImpossibleBench (original)",
            "title": "ImpossibleBench, benchmark prompt (Celeste's reads)",
            "sub": "the original lcb.yaml setting",
            "arms": sorted({x["arm"] for x in d["arms"] if x["env"] == "ImpossibleBench (original)"}),
            "beh": "hack",
        },
    ]
    for t in ("payback", "email", "email_4johns", "dana", "calendar", "gitbranch"):
        e = f"overeager/{t}"
        names = [r for (ee, r) in arms if ee == e]
        order = OE_B if "base" in names or any(n.startswith("base@") for n in names) else OE_A
        lst = [r for o in order for r in names if r == o or r.startswith(o + "@")]
        sub = "edit arms (early turns + 32 spread positions read)"
        panels.append({"env": e, "title": NAMES[e], "sub": sub, "arms": lst, "beh": "acts"})
    for e, beh in (
        ("Donation Bet", "leaks"),
        ("Donation Bet (empty think)", "leaks"),
        ("agentic misalignment", "harmful"),
        ("alignment faking (no CoT)", "complies"),
        ("alignment faking (thinking)", "complies"),
    ):
        lst = sorted(r for (ee, r) in arms if ee == e)
        if e.startswith("alignment"):
            lst = [r for r in ("free", "preview", "paid", "public") if r in lst]
        panels.append({"env": e, "title": NAMES[e], "sub": "arms", "arms": lst, "beh": beh})
    d["panels"] = panels
    d["all_envs"] = sorted(
        [{"env": x["env"], "k": x["m_k"], "ks": x["m_s"], "n": x["m_n"]} for x in d["envs"] if x["m_n"] >= 3000],
        key=lambda x: -x["ks"] / x["n"],
    )
    keep = set(d["show_envs"]) | {"wildchat", "swesmith", "rlvr1", "tmax"}
    d["arms"] = [x for x in d["arms"] if x["env"] in keep]
    d["sections"] = [x for x in d["sections"] if x["env"] in keep]
    d["examples"] = [x for x in d["examples"] if x["env"] in keep]
    html = (HERE / "page.html").read_text().replace("__DATA__", json.dumps(d).replace("</", "<\\/"))
    Path(a.o).write_text(html)
    print(a.o, len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
