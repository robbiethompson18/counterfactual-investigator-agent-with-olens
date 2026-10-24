"""GLM 5.3 evalcue rollouts with Celeste's GLM-5.3 O-lens reads, as extra viewer patterns (one per env and arm).

Source: ceselder/glm-5.3-olens-counterfactual-investigator (HF), config `evalcue`: episodes.parquet has each episode's render
through GLM-5.3's own chat template (`rendered_text`), readouts.parquet the reads (layer 51, RL and SFT lenses, greedy + 4
samples at T=1, no FVE) at ~12 chosen punctuation positions, indexed by GLM-5.3 token. Episodes join to our run dirs
(data/runs/<env>/<run>/rows.jsonl) by run_dir + row_index; scripts/celeste_olens_reads.py and cfbench use the same reads.

Tokens are exact slices of the render (GLM-5.3's tokenizer offsets), so `pos` indexes them directly; a read whose token
doesn't match is dropped. The RL lens's greedy readout is the greedy read and its 4 samples the T=1 read; the SFT lens is
left out (mostly web-text noise on these transcripts). Reads live only in S["by_tid"]: the pod serves Qwen's lens, so
these can't be re-read live.
"""

import json
import re
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO = "ceselder/glm-5.3-olens-counterfactual-investigator"
RUNS = Path(__file__).resolve().parent.parent.parent / "data"  # run_dir is "runs/<env>/<run>"
ENVS = ("scheming_oversight_blk", "hidden_goal_grade")  # the cfbench smoke envs; any env in her evalcue config works
PREFIX = "glm/"
LAYER = 51
LENS = "celeste-glm53-rl1b_2-step100"  # her `rl_*` columns: /vol/ckpt/rl1b_2/step_0100
ROLES = {"<|system|>": "system", "<|user|>": "user", "<|assistant|>": "assistant", "<|observation|>": "tool"}
SPECIAL = re.compile(r"^(<\|[a-z_]+\|>|</?think>|\[gMASK\]|<sop>)$")


def tokens_of(tok, text):
    """GLM-5.3 tokens of the render as exact text slices (a byte-split character goes to the first token covering it)."""
    out, end = [], 0
    for a, b in tok.encode(text, add_special_tokens=False).offsets:
        out.append(text[max(a, end) : b] if b > end else "")
        end = max(end, b)
    assert "".join(out) == text, "token offsets do not cover the render"
    return out


def render(tid, tokens):
    """The viewer's render() record: messages are the spans between GLM's role tokens, which (with the token before them) are
    the end-of-turn zones."""
    marks = [i for i, t in enumerate(tokens) if t in ROLES]
    msgs = [[i + 1, j] for i, j in zip(marks, marks[1:] + [len(tokens)])]
    first_reply = next((k for k, i in enumerate(marks) if tokens[i] == "<|assistant|>"), 0)
    return {
        "tid": tid,
        "ids": [],
        "tokens": tokens,
        "special": [bool(SPECIAL.match(t)) for t in tokens],
        "msgs": msgs,
        "roles": [ROLES[tokens[i]] for i in marks],
        "cites": [],
        # a role token plus the token before it (the end of the previous message, where her final_user read sits)
        "zones": [[max(0, i - 1), i + 1] for i in marks],
        "truncated": False,
        "main_zone": first_reply,
        "focus": [],
    }


def docent_link(run_dir, row_index):
    """The episode's Docent agent run, if this machine has the run dir's docent.json."""
    p = RUNS / run_dir / "docent.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    rid = d["runs"][row_index] if row_index < len(d["runs"]) else None
    return f"https://docent.transluce.org/dashboard/{d['collection_id']}/agent_run/{rid}" if rid else None


def load(S, key):
    """Append one pattern per (env, arm) to S["patterns"], the transcripts to S["trs"], and the reads to S["by_tid"]."""
    get = lambda f: hf_hub_download(REPO, f, repo_type="dataset")
    in_envs = pc.field("env").isin(ENVS)
    eps = pq.read_table(
        get("evalcue/episodes.parquet"), filters=in_envs, columns=["tid", "run_dir", "row_index", "env", "arm", "item", "rendered_text"]
    )
    cols = ["tid", "pos", "token", "kind", "role", "positive", "view", "rl_greedy", *(f"rl_s{k}" for k in range(1, 5))]
    reads = pq.read_table(get("evalcue/readouts.parquet"), filters=in_envs & (pc.field("view") == "prefix"), columns=cols).to_pylist()
    by_ep = {}
    for r in reads:
        by_ep.setdefault(r["tid"], []).append(r)
    tok = Tokenizer.from_file(hf_hub_download("zai-org/GLM-5.3", "tokenizer.json"))
    groups = {}  # (env, arm) -> tids
    for e in eps.to_pylist():
        if e["tid"].endswith(":late") or e["tid"] not in by_ep:
            continue
        rs, ri = by_ep[e["tid"]], int(e["row_index"])
        run = e["run_dir"].split("/")[-1]
        tid = f"{PREFIX}{e['env']}/{run}/{ri}"
        tokens = tokens_of(tok, e["rendered_text"])
        S.setdefault("rendered", {})[tid] = render(tid, tokens)
        positive = bool(rs[0]["positive"])
        kept = [r for r in rs if int(r["pos"]) < len(tokens) and tokens[int(r["pos"])].strip() == str(r["token"]).strip()]
        S["trs"][tid] = {
            "transcript_id": tid,
            "pattern_id": f"glm:{e['env']}:{e['arm']}",
            "prompt_id": f"glm:{e['env']}:{e['item']}",
            "messages": [],
            "judgment": {"match": positive, "explanation": ""},
            "is_highlight": False,
            "weirdchat_url": docent_link(e["run_dir"], ri),
            "glm": {
                "env": e["env"],
                "arm": e["arm"],
                "item": e["item"],
                "run": f"{e['run_dir']}:{ri}",
                "positive": positive,
                "read_pos": sorted({int(r["pos"]) for r in kept}),
                "n_dropped": len(rs) - len(kept),
            },
        }
        for r in kept:
            pos = int(r["pos"])
            for temp, n, texts in ((0.0, 1, [r["rl_greedy"]]), (1.0, 4, [r[f"rl_s{k}"] for k in range(1, 5)])):
                rec = {
                    "key": key(tid, pos, LAYER, temp, n, lens=LENS),
                    "tid": tid,
                    "pos": pos,
                    "layer": LAYER,
                    "temperature": temp,
                    "n": n,
                    "seed": 0,
                    "token": tokens[pos],
                    "norm": None,
                    "samples": [{"text": t, "bullets": [t]} for t in texts if t],  # no FVE: her lens has no AR scores here
                    "lens": LENS,
                    "kind": r["kind"],
                }
                S["by_tid"].setdefault(tid, {})[rec["key"]] = rec
        groups.setdefault((e["env"], e["arm"]), []).append(tid)
    for (env, arm), tids in sorted(groups.items()):
        tids.sort(key=lambda t: (not S["trs"][t]["glm"]["positive"], S["trs"][t]["glm"]["item"], t))
        k = sum(S["trs"][t]["glm"]["positive"] for t in tids)
        S["patterns"].append(
            {
                "idx": len(S["patterns"]),
                "ds": "glm",
                "pattern_id": f"glm:{env}:{arm}",
                "title": f"GLM 5.3 · {env} · {arm}",
                "behavior_id": env,
                "method": "evalcue env arm",
                "metrics": {"match_rate": k / len(tids)},
                "n_prompts": len({S["trs"][t]["glm"]["item"] for t in tids}),
                "tids": tids,
                "weirdchat_url": None,
                "elo": {},
                "description": f"z-ai/glm-5.3 (OpenRouter, Z.AI fp8, reasoning effort medium) in the {env} env, arm {arm}: "
                f"{k} of {len(tids)} rollouts show the behavior (the env's programmatic label). O-lens reads are Celeste's "
                f"GLM-5.3 RL lens at layer {LAYER}, ~12 chosen positions per rollout, greedy + 4 samples, no FVE.",
                "representative_user_text": "",
                "cluster_label": env,
                "openrouter_replication": None,
                "glm": {"env": env, "arm": arm, "k": k, "n": len(tids)},
            }
        )
