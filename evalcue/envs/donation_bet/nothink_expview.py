"""Donation Bet thinking-off (empty think block) runs vs thinking on, Qwen3.6-27B and GLM 5.3 -> expview run.

Every item shows the env files, the subject call exactly as sent (for GLM thinking off, the raw completions prompt with the
prefilled "<think></think>") and the DeepSeek V4.1 Flash judge call (judge.py's prompt filled in, and its reply). The page
holds a sample per (model, mode, arm, question); every row is on Docent (collection donation_bet_goodbad), linked per item.

  uv run python evalcue/envs/donation_bet/nothink_expview.py [PER_CELL]   (default 12 per model × mode × arm × question)
  python3 $EXPVIEW_DIR/expview.py build $EXPVIEW_DIR/runs/donation_bet_nothink
"""

import json
import os
import random
import sys
from pathlib import Path

ENVD = Path(__file__).parent
REPO = ENVD.parents[2]
EXPVIEW = Path(os.environ.get("EXPVIEW_DIR", Path.home() / "workspace/expview"))
sys.path.insert(0, str(EXPVIEW))
sys.path.insert(0, str(REPO))
from expview import Run, call

RUNS = REPO / "data/runs/donation_bet_goodbad"
DOCENT = "https://docent.transluce.org"
# (model, mode) -> run tags, lines file
SETS = {
    ("Qwen3.6-27B", "on"): (["2026-10-22_neutral", "2026-10-22_bets", "2026-10-22_beers-nothing"], "thresholds_goodbad.json"),
    ("Qwen3.6-27B", "off"): (["2026-10-25_qwen-nothink-cap8k"], "thresholds_goodbad.json"),
    ("GLM 5.3", "on"): (
        ["2026-10-23_glm53-neutral", "2026-10-23_glm53-neutral-x4", "2026-10-23_glm53-bets-ownline", "2026-10-23_glm53-bets-ownline-x4"],
        "thresholds_goodbad_glm53.json",
    ),
    ("GLM 5.3", "off"): (["2026-10-25_glm53-nothink-ownline"], "thresholds_goodbad_glm53.json"),
}
ARMS = ["neutral", "good_below", "good_above", "beers_good_below", "beers_good_above", "nothing_good_below", "nothing_good_above"]
GLM_RAW = "[gMASK]<sop><|system|>Reasoning Effort: Max{turns}<|assistant|><think></think>"


def judge_prompt(r):
    # judge.PROMPT without importing judge.py (it parses argv at import)
    src = (ENVD / "judge.py").read_text()
    tmpl = src.split('PROMPT = """', 1)[1].split('"""', 1)[0].replace("\\\n", "")
    msg = r["messages"][-1]
    return tmpl.format(
        user=r["messages"][0]["content"], reasoning=(msg.get("reasoning_content") or "")[-60_000:], reply=msg.get("content") or ""
    )


def main():
    per_cell = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    rng = random.Random(0)
    t = REPO / "docs/figures/donation_bet/nothink/onoff_table.md"  # from nothink_compare.py
    table = "\n\n**Gaps over all rows** (judge estimates; ack = the judge says the model explicitly ties its number to the bet):\n\n"
    table = table + t.read_text() if t.exists() else ""
    run = Run(
        EXPVIEW / "runs/donation_bet_nothink",
        title="Donation Bet thinking off",
        description=(
            "Donation Bet (`donation_bet.goodbad`) with **thinking off via an empty think block** vs the earlier thinking-on runs, "
            "Qwen3.6-27B (bf16 vLLM, template `enable_thinking=False` = `<think>\\n\\n</think>\\n\\n` prefilled) and GLM 5.3 "
            "(OpenRouter raw completions with `<|assistant|><think></think>` prefilled; OpenRouter refuses `reasoning.enabled=false` "
            "for GLM 5.3). Thinking-off runs use the thinking-on sampling (temperature 1.0, top_p 0.95), so only the think block "
            "differs. Lines: each model's thinking-on neutral medians. `le_line` = judge estimate ≤ line (the positive used in the "
            f"gap); `regex_le_line` = the env's `detect()`. Sample of {per_cell} per model × mode × arm × question; every row is on "
            "Docent (link in each item's header). The judge call is DeepSeek V4.1 Flash (`judge.py`), the same judge as the "
            "thinking-on gap table; thinking-on judge replies were not stored, so those items show the parsed fields only."
        )
        + table,
        label_order=["model", "mode", "arm", "le_line", "regex_le_line", "acknowledges", "estimate", "line", "question", "finish"],
    )
    run.reset()
    env = {f"evalcue/envs/donation_bet/{n}": (ENVD / n).read_text() for n in ["goodbad.py", "__init__.py", "judge.py"]}
    env |= {f"evalcue/envs/donation_bet/{n}": (ENVD / n).read_text() for n in ["thresholds_goodbad.json", "thresholds_goodbad_glm53.json"]}
    n = 0
    for (model, mode), (tags, lines_file) in SETS.items():
        lines = json.loads((ENVD / lines_file).read_text())
        cells = {}
        for t in tags:
            d = RUNS / t
            judged = {(j["arm"], j["item"], j["sample"]): j for j in map(json.loads, (d / "judged.jsonl").open())}
            docent = json.loads((d / "docent.json").read_text()) if (d / "docent.json").exists() else None
            for i, r in enumerate(map(json.loads, (d / "rows.jsonl").open())):
                if "error" in r or r["arm"] not in ARMS:
                    continue
                cells.setdefault((r["arm"], r["item"]), []).append((t, i, r, judged.get((r["arm"], r["item"], r["sample"])), docent))
        for (arm, q), xs in sorted(cells.items()):
            for t, i, r, j, docent in rng.sample(xs, min(per_cell, len(xs))):
                line = r["meta"].get("threshold", lines[q])
                est = (j or {}).get("estimate")
                labels = {
                    "model": model,
                    "mode": f"thinking {mode}",
                    "arm": arm,
                    "le_line": None if est is None else est <= line,
                    "regex_le_line": None if (e := (r.get("detect_notes") or {}).get("estimate")) is None else e <= line,
                    "acknowledges": (j or {}).get("acknowledges"),
                    "estimate": est,
                    "line": line,
                    "question": q,
                    "finish": r.get("finish"),
                }
                msgs = r["messages"]
                params = {"thinking": mode == "on", "max_tokens": None}
                if mode == "off" and model == "GLM 5.3":
                    turns = "".join({"system": "<|system|>", "user": "<|user|>"}[m["role"]] + m["content"] for m in msgs[:-1])
                    params |= {
                        "endpoint": "completions (raw)",
                        "raw_prompt": GLM_RAW.format(turns=turns),
                        "provider": msgs[-1].get("provider"),
                    }
                calls = [call("subject", r["model"], messages=msgs, params=params, completion_tokens=r.get("completion_tokens"))]
                if j:
                    reply = (
                        j.get("raw")
                        or json.dumps({k: j.get(k) for k in ("estimate", "acknowledges", "evidence")})
                        + "  [parsed fields; raw reply not stored]"
                    )
                    calls.append(
                        call(
                            "judge",
                            "deepseek/deepseek-v4.1-flash",
                            messages=[
                                {"role": "user", "content": judge_prompt(r)},
                                {"role": "assistant", "content": reply, "reasoning": j.get("raw_reasoning") or ""},
                            ],
                            params={"reasoning": {"effort": "medium"}, "response_format": "json_object"},
                            parsed={k: j.get(k) for k in ("estimate", "acknowledges", "evidence")},
                        )
                    )
                aid = docent["runs"][i] if docent and i < len(docent["runs"]) else None
                run.add(
                    id=f"{model} {mode}/{arm}/{q}/{r['sample']}",
                    group=f"{model} · thinking {mode} · {arm}",
                    labels={k: v for k, v in labels.items() if v is not None},
                    env_files=env,
                    extra={"source": f"donation_bet_goodbad/{t}", "row": i},
                    links={"Docent transcript": f"{DOCENT}/dashboard/{docent['collection_id']}/agent_run/{aid}"} if aid else None,
                    calls=calls,
                )
                n += 1
    print(n, "items")


if __name__ == "__main__":
    main()
