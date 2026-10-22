"""Adaptive weirdness rounds judged by Qwen3.6-27B (vLLM at localhost:8000), appended to run/comparisons.jsonl with judge="qwen27b".

Each round: elo.py pair (both orders, information-based neighbour matching) -> one request per pair (16 in flight,
thinking off, temperature 0) -> append -> fit (per-judge position term) -> export for the dashboard -> stop checks.
Stops when mean 90% rank-CI width < 150 AND split-half Spearman >= 0.9 AND top-100 churn < 5, or the endpoint is gone.
  python qwen_tourney.py FIRST_ROUND [MAX_ROUNDS]"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import elo
import httpx
from judge_qwen import MODEL, URL, parse, prompt
from scipy.stats import spearmanr

RUN = Path("run")
CONC = 16
LOG = RUN / "qwen_tourney.jsonl"


def endpoint_up():
    for _ in range(10):
        try:
            if httpx.get("http://localhost:8000/v1/models", timeout=10).status_code == 200:
                return True
        except Exception:  # noqa: BLE001, S110
            pass
        time.sleep(30)
    return False


async def judge(pairs, by_id):
    sem = asyncio.Semaphore(CONC)
    res = []
    async with httpx.AsyncClient(timeout=300) as cl:

        async def one(x, y, batch):
            body = {
                "model": MODEL,
                "temperature": 0,
                "max_tokens": 200,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": prompt(x, y, by_id)}],
            }
            async with sem:
                for t in range(5):
                    try:
                        r = await cl.post(URL, json=body)
                        r.raise_for_status()
                        break
                    except Exception:  # noqa: BLE001
                        await asyncio.sleep(2 ** (t + 1))
                else:
                    return None
            d = parse(r.json()["choices"][0]["message"].get("content"))
            if not d:
                return "bad"
            w = x if d["winner"].strip().upper() == "A" else y
            return {
                "a": x,
                "b": y,
                "first": x,
                "winner": w,
                "batch": batch,
                "margin": d.get("margin"),
                "reason": d.get("reason"),
                "judge": "qwen27b",
            }

        res = await asyncio.gather(*(one(x, y, b) for x, y, b in pairs))
    return res


def main(first, max_rounds):
    prev_top = None
    for rnd in range(first, first + max_rounds):
        if not endpoint_up():
            print(json.dumps({"round": rnd, "stop": "endpoint down"}), flush=True)
            return
        out = subprocess.run(
            [sys.executable, "elo.py", "pair", "run", "--round", str(rnd), "--both-orders", "--batch-size", "100000", "--neighbours", "12"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        by_id = {it["id"]: it for it in elo.load_items(RUN)}
        pairs = []
        for md in json.loads(out)["batches"]:
            pf = Path(md.replace(".md", ".pairs.json"))
            name = pf.name.replace(".pairs.json", "")
            pairs += [(x, y, name) for x, y in json.loads(pf.read_text())]
            Path(md).unlink()  # the judge prompt file is only for Sonnet subagents
        t0 = time.time()
        res = asyncio.run(judge(pairs, by_id))
        ok = [r for r in res if isinstance(r, dict)]
        with open(RUN / "comparisons.jsonl", "a") as fh:
            fh.writelines(json.dumps({**r, "round": rnd}) + "\n" for r in ok)
        n_bad = sum(r == "bad" for r in res)
        n_fail = sum(r is None for r in res)
        # fit + dashboard
        subprocess.run([sys.executable, "elo.py", "fit", "run"], capture_output=True, check=True)
        subprocess.run([sys.executable, "export.py"], capture_output=True, check=True)
        rk = json.loads((RUN / "ranking.json").read_text())
        items = elo.load_items(RUN)
        comps = [c for c in elo.load_comps(RUN) if c.get("winner")]
        _, t1, *_ = elo.fit(items, comps[0::2])
        _, t2, *_ = elo.fit(items, comps[1::2])
        sh = float(spearmanr(t1, t2).statistic)
        top = {r["id"] for r in rk["ranking"][:100]}
        churn = None if prev_top is None else len(top - prev_top)
        prev_top = top
        s = rk["summary"]
        row = {
            "round": rnd,
            "pairs": len(pairs),
            "added": len(ok),
            "unparsed": n_bad,
            "failed": n_fail,
            "secs": round(time.time() - t0),
            "comparisons": s["comparisons"],
            "ci_width": s["mean_rank_ci_width"],
            "split_half": round(sh, 3),
            "top100_churn": churn,
            "bias": s["position_bias_logit"],
            "time": time.strftime("%H:%M"),
        }
        print(json.dumps(row), flush=True)
        with open(LOG, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        if n_fail > len(pairs) // 2:
            print(json.dumps({"round": rnd, "stop": "errors"}), flush=True)
            return
        if s["mean_rank_ci_width"] < 150 and sh >= 0.9 and churn is not None and churn < 5:
            print(json.dumps({"round": rnd, "stop": "converged"}), flush=True)
            return
    print(json.dumps({"stop": "max rounds"}), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 200)
