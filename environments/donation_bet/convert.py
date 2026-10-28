"""The evalcue donation_bet / donation_bet.goodbad runs -> data/envs/donation_bet/<run>/, after checking build() against
every stored row's prompt (`--dry` reports without writing).

Old arms are renamed to the arm whose build() gives that prompt: goodbad arms keep their names plus the lines suffix the
run used (`good_below@glm53`), v1 arms get the `v1/` prefix (`v1/above+amf`, `v1/above+amf@hand`). The lines set is
read off the rows' stored thresholds. The old LLM judge's outputs (judged.jsonl, deepseek/deepseek-v4.1-flash) become
`judge@converted` labels (kind judge).

Sources are the bramble checkouts and worktrees that hold the raw rows (data/ is per checkout) and the HF copies under
runs/ for the runs made on other machines; SRC lists them.
"""

import json
import sys
from pathlib import Path

from environments._lib.convert import docent_url, first_diff, write_run
from environments._lib.core import episode_id
from environments._lib.runfolder import REPO_ROOT
from environments.donation_bet import env as E

W = Path.home() / "workspace"
MAIN = W / "counterfactual-investigator-agent-with-olens/data/runs"
HFR = Path(sys.argv[sys.argv.index("--hf") + 1]) if "--hf" in sys.argv else REPO_ROOT / "data/runs"  # an HF download's runs/
GAP = W / "cia-dbet-gap/data/runs/donation_bet_goodbad"
NOTHINK = W / "cia-dbet-nothink/data/runs/donation_bet_goodbad"
# new run name -> (old experiment, rows dir, extra dirs to look for docent.json in)
SRC = {
    **{r: ("donation_bet", MAIN / "donation_bet" / r) for r in
       ["2026-10-22_calib", "2026-10-22_goodcause-beers", "2026-10-22_nochange-5x5", "2026-10-22_paired", "2026-10-22_paired-handlines"]},
    **{r: ("donation_bet", HFR / "donation_bet" / r) for r in ["2026-10-23_glm53_n10_c0", "2026-10-23_glm53_n10_c1"]},
    **{r: ("donation_bet_goodbad", MAIN / "donation_bet_goodbad" / r) for r in
       ["2026-10-22_neutral", "2026-10-22_bets", "2026-10-22_beers-nothing"]},
    "2026-10-23_glm53_smoke": ("donation_bet_goodbad", MAIN / "donation_bet_goodbad/2026-10-23_glm53-smoke"),
    **{r: ("donation_bet_goodbad", HFR / "donation_bet_goodbad" / r) for r in
       ["2026-10-22_or_qwen_n60", "2026-10-23_r2_n60_on", "2026-10-23_r2_n60_off", "2026-10-23_think_on_n60", "2026-10-23_think_off_n60"]},
    **{f"2026-10-23_glm53_{s.replace('-', '_')}": ("donation_bet_goodbad", GAP / f"2026-10-23_glm53-{s}") for s in
       ["neutral", "neutral-x4", "bets", "bets-x4", "bets-ownline", "bets-ownline-x4",
        "named", "named-x4", "named-ownline", "named-ownline-x4"]},
    "2026-10-25_glm53_nothink_ownline": ("donation_bet_goodbad", NOTHINK / "2026-10-25_glm53-nothink-ownline"),
    "2026-10-25_qwen-nothink": ("donation_bet_goodbad", NOTHINK / "2026-10-25_qwen-nothink"),
    "2026-10-25_qwen-nothink-cap8k": ("donation_bet_goodbad", NOTHINK / "2026-10-25_qwen-nothink-cap8k"),
}  # fmt: skip


def docent_for(name: str, src: Path, n: int):
    """docent.json beside the rows, else in the main checkout's copy of the run (where `runs.py finish` wrote it)."""
    for d in (src, MAIN / src.parent.name / src.name, MAIN / src.parent.name / name):
        if (d / "docent.json").exists():
            doc = json.loads((d / "docent.json").read_text())
            return doc["collection_id"], doc["runs"]
    return None, [None] * n


def lines_of(template: str, rows: list[dict]) -> str:
    """The lines set every row's stored threshold agrees with."""
    sets = ["v1", "hand"] if template == "v1" else ["qwen", "glm53"]
    with_thr = [r for r in rows if (r.get("meta") or {}).get("threshold") is not None and r["arm"] not in ("neutral", "plain")]
    ok = [s for s in sets if all(E.LINES[s][r["item"]] == r["meta"]["threshold"] for r in with_thr)]
    if not with_thr:
        return sets[0]
    if not ok:
        raise SystemExit(f"no lines set matches the thresholds of {rows[0].get('env')}")
    return ok[0]


def new_arm(template: str, old: str, lines: str) -> str:
    if template == "v1":
        return "v1/" + old + ("@hand" if lines == "hand" and old != "plain" else "")
    sfx = {"qwen": "", "glm53": "@glm53"}[lines]
    return old + (sfx if old != "neutral" else "")


def target(meta: dict) -> dict:
    m = meta["models"][0]
    served = m.get("served_by", "")
    p = meta["params"]
    if served.startswith("vLLM"):
        backend, provider = "vllm", None
    elif "raw completions" in served:
        backend, provider = "openrouter-raw", served.split("providers ")[1].rstrip(")") if "providers " in served else None
    else:
        backend = "openrouter"
        provider = served.split("(")[1].split(",")[0] if "(" in served else None
        provider = None if provider in ("any provider", None) else provider
    sampling = p.get("sampling") or ("thinking-on" if p["thinking"] else "thinking-off")  # evalcue: the mode's Qwen sampling
    return {
        "id": f"{m['name']}@{backend}:{provider or 'any'}",
        "model": m["name"],
        "backend": backend,
        "provider": provider,
        "thinking": p["thinking"],
        "max_tokens": p.get("max_tokens"),
        "served_by": served,
        "sampling": sampling,
    }


def main(dry=False, only=None):
    report = {}
    for name, (exp, src) in SRC.items():
        if only and name not in only:
            continue
        template = "v1" if exp == "donation_bet" else "ref"
        meta = json.loads((src / "meta.json").read_text())
        rows = [json.loads(line) for line in (src / "rows.jsonl").read_text().splitlines()]
        coll, docent_runs = docent_for(name, src, len(rows))
        lines = lines_of(template, rows)
        judged = {}
        if (src / "judged.jsonl").exists():
            for j in map(json.loads, (src / "judged.jsonl").read_text().splitlines()):
                judged[(j["arm"], j["item"], j["sample"])] = {k: v for k, v in j.items() if k not in ("arm", "item", "sample")}
        mism, measure_diff, eps, labels, jlabels, seen, ported = [], 0, [], [], [], set(), []
        tgt = target(meta)
        served = {p for r in rows if r.get("messages") if (p := r["messages"][-1].get("provider")) is not None}
        if tgt["backend"] == "openrouter-raw" and len(served) == 1:  # a provider list was allowed but one served every row
            tgt |= {"providers_allowed": tgt["provider"], "provider": served.pop().lower()}
            tgt["id"] = f"{tgt['model']}@openrouter-raw:{tgt['provider']}"
        for i, r in enumerate(rows):
            arm = new_arm(template, r["arm"], lines)
            ep = E.build(arm, r["item"])
            r.setdefault("messages", [])
            if r["messages"]:
                old = r["messages"][0]
                if ep.messages[0] != {k: old[k] for k in ("role", "content")}:
                    mism.append({"line": i, "arm": arm, "item": r["item"], "diff": first_diff(ep.messages[0]["content"], old["content"])})
            if (r.get("meta") or {}).get("threshold") != ep.meta.get("threshold") and r["arm"] not in ("neutral", "plain"):
                mism.append({"line": i, "arm": arm, "item": r["item"], "diff": f"threshold {ep.meta.get('threshold')} vs {r['meta']}"})
            eid = episode_id(arm, r["item"], r["sample"])
            if eid in seen:
                raise SystemExit(f"{name}: duplicate episode {eid}")
            seen.add(eid)
            stored = {"positive": r.get("positive"), **(r.get("detect_notes") or {})}
            rec = {"arm": arm, "messages": r["messages"], "meta": r.get("meta") or {}}
            if "error" not in r and r.get("positive") is not None:
                new = E.measure(rec)
                if any(new.get(k) != v for k, v in stored.items() if k in new):
                    measure_diff += 1
            eps.append(
                {
                    "id": eid, "arm": arm, "item": r["item"], "sample": r["sample"], "target": tgt["id"],
                    "messages": r["messages"], "tools": [], "files_before": {}, "files_after": {}, "state": {},
                    "meta": r.get("meta") or {}, "finish": r.get("finish"), "turns": r.get("turns"),
                    "usage": {"completion_tokens": r.get("completion_tokens")}, "error": r.get("error"),
                    "source": f"data/runs/{exp}/{src.name}:{i}", "source_arm": r["arm"],
                    "docent_run": docent_runs[i] if i < len(docent_runs) else None,
                    "docent_url": docent_url(coll, docent_runs, i),
                }
            )  # fmt: skip
            labels.append((eid, None if "error" in r else stored))
            ported.append((eid, None if "error" in r else E.measure(rec)))
            if (j := judged.get((r["arm"], r["item"], r["sample"]))) is not None:
                jlabels.append((eid, j))
        report[name] = {
            "rows": len(rows),
            "lines": lines,
            "start_context_mismatches": len(mism),
            "examples": mism[:3],
            "measure_disagrees": measure_diff,
            "judged": len(jlabels),
            "docent": sum(1 for e in eps if e["docent_url"]),
        }
        if not dry and not mism:
            m = {
                "description": meta.get("description"),
                "converted_from": f"data/runs/{exp}/{src.name}",
                "source_env": f"evalcue {exp.replace('donation_bet_goodbad', 'donation_bet.goodbad')}",
                "git": meta.get("git_sha") or meta.get("git"),
                "command": meta.get("command"),
                "target": tgt,
                "params": meta.get("params"),
                "lines": lines,
                "docent_collection": coll,
                "source_status": meta.get("status", "done"),
                "primary": E.PRIMARY,
            }
            d = write_run("donation_bet", name, m, eps, labels)
            # today's measure() on the same replies (the old detector changed between runs; this one is what new runs use)
            (d / "labels" / "measure@port.jsonl").write_text(
                "".join(json.dumps({"episode": e, "values": v, "kind": "code"}) + "\n" for e, v in ported)
            )
            if jlabels:
                (d / "labels" / "judge@converted.jsonl").write_text(
                    "".join(json.dumps({"episode": e, "values": v, "kind": "judge"}) + "\n" for e, v in jlabels)
                )
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main(dry="--dry" in sys.argv, only=[a for a in sys.argv[1:] if a.startswith("20")])
