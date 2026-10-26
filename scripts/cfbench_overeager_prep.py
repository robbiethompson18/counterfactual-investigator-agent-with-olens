# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow"]
# ///
"""Rows and O-lens reads for cfbench over the GLM 5.3 over-eagerness settings (payback, dana, calendar, gitbranch, email,
email_4johns), from the dense O-lens reader's inputs and outputs.

    uv run --script scripts/cfbench_overeager_prep.py ~/workspace/tmp/glm_dense

Inputs: <root>/overeager/<setting>.jsonl (the rows the reader was given: arm, item, positive, tools, messages, source = the
realism run file under donehack/realism/runs/) and <root>/2026-10-26_glm53_overeager/<setting>/{episodes,readouts_dense,
dense_prefix_dups}.parquet (GLM verbalizer reads at block 51; prefix-shared positions are stored once and listed in the dups
table). Writes data/runs/cfbench_overeager/<setting>/rows.jsonl for cfbench/envs/overeager_rows.py and
data/runs/cfbench_overeager/reads/{reads,requests}.jsonl in cfbench/olens.py's format.

Sources are `<family>/<setting>:<row>`; family is `email` for both email settings, so the O-lens placebo never answers
an email case from the other email setting's reads.
"""

import json
import sys
from pathlib import Path

import pandas as pd

SETTINGS = ["payback", "dana", "calendar", "gitbranch", "email", "email_4johns"]
REALISM_RUNS = Path.home() / "workspace/counterfactual-investigator-agent-with-olens/.worktrees/realism/donehack/realism/runs"
OUT = Path("data/runs/cfbench_overeager")
LAYER = 51
FIELD = {"reasoning": "reasoning_content", "assistant": "content", "tool_call": "tool_call"}  # reader role -> message field
SAMPLES = ["greedy", "s1", "s2", "s3", "s4"]


def family(setting: str) -> str:
    return "email" if setting.startswith("email") else setting


def messages(raw: list[dict], tools: list[dict]) -> list[dict]:
    """GLM rows carry reasoning as `reasoning`; cfbench renders `reasoning_content`, and every generated turn needs the key
    (core.initial_context finds the first generated turn by it). The tool schemas go first as a `tools` message, one JSON
    line per tool as GLM's chat template writes them: the done-tool edits change them, and the reader read them (its
    msg_index -1), so the predictor sees them in every transcript and the O-lens tools can address those reads."""
    out = [{"role": "tools", "content": "\n".join(json.dumps(t["function"], ensure_ascii=False) for t in tools)}]
    for m in raw:
        m = {k: v for k, v in m.items() if k != "reasoning_details"}
        if m["role"] == "assistant":
            m["reasoning_content"] = m.pop("reasoning", None) or ""
        out.append(m)
    return out


def env_of(source: str) -> dict:
    d = json.loads((REALISM_RUNS / source).read_text())
    return d.get("env") or d["spec"]


def char_offset(text: str, left: str, token: str) -> int | None:
    """Where the read token starts in the message field. `left` is the 40 decoded characters up to and including the token;
    take the longest of its tails (still ending in the token) that occurs exactly once."""
    if not left.endswith(token) or not token:
        return None
    for k in (len(left), 24, 12, len(token)):
        ctx = left[-k:]
        at = text.find(ctx)
        if at >= 0 and text.find(ctx, at + 1) < 0:
            return at + len(ctx) - len(token)
    return None


def main(root: Path):
    reads_dir = OUT / "reads"
    reads_dir.mkdir(parents=True, exist_ok=True)
    n_rows = n_placed = 0
    with (reads_dir / "reads.jsonl").open("w") as reads_out, (reads_dir / "requests.jsonl").open("w") as req_out:
        for setting in SETTINGS:
            rows = [json.loads(line) for line in (root / "overeager" / f"{setting}.jsonl").read_text().splitlines()]
            out_rows, msgs = [], {}
            for i, r in enumerate(rows):
                env = env_of(r["source"])
                src = f"{family(setting)}/{setting}:{i}"
                msgs[i] = messages(r["messages"], env["tools"])
                out_rows.append(
                    {
                        "source": src,
                        "arm": r["arm"],
                        "item": str(r["item"]),
                        "model": r["model"],
                        "positive": bool(r["positive"]) if not isinstance(r["positive"], str) else r["positive"] == "True",
                        "messages": msgs[i],
                        "tools": env["tools"],
                        "files_before": env.get("files") or {},
                        "setup": env.get("setup") or [],
                        "workdir": env.get("workdir"),
                        "realism_source": r["source"],
                    }
                )
                req_out.write(json.dumps({"id": src, "messages": msgs[i]}) + "\n")
            (OUT / setting).mkdir(exist_ok=True)
            (OUT / setting / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out_rows))

            d = root / "2026-10-26_glm53_overeager" / setting
            episodes = pd.read_parquet(d / "episodes.parquet", columns=["tid", "row_index"])
            row_of = dict(zip(episodes.tid, episodes.row_index))
            dense = pd.read_parquet(d / "readouts_dense.parquet")
            dups = pd.read_parquet(d / "dense_prefix_dups.parquet")
            assert set(dups.canon_in) <= {"readouts_dense"}, set(dups.canon_in)
            canon = dense.drop_duplicates(["tid", "pos"]).rename(columns={"tid": "canon_tid", "pos": "canon_pos"})
            copies = dups[["tid", "pos", "canon_tid", "canon_pos"]].merge(canon, on=["canon_tid", "canon_pos"], how="inner")
            copies = copies.drop(columns=["canon_tid", "canon_pos"])
            allreads = pd.concat([dense, copies], ignore_index=True)
            for r in allreads.itertuples():
                i = int(row_of[r.tid])
                m = msgs[i][r.msg_index + 1] if -1 <= r.msg_index < len(msgs[i]) - 1 else None
                field = FIELD.get(r.role, "content")
                offset = None
                if m is not None and isinstance(m.get(field), str):
                    offset = char_offset(m[field], r.ctx_left40 or "", r.token)
                n_rows += 1
                n_placed += offset is not None
                bullets = [
                    {"text": getattr(r, f"rl_{s}"), "fve": float(getattr(r, f"fve_rl_{s}")), "coef": None}
                    for s in SAMPLES
                    if isinstance(getattr(r, f"rl_{s}"), str) and getattr(r, f"rl_{s}")
                ]
                bullets.sort(key=lambda b: -b["fve"] if b["fve"] == b["fve"] else 1)
                row = {
                    "id": f"{family(setting)}/{setting}:{i}",
                    "token_index": int(r.pos),
                    "token": r.token,
                    "prefix": (r.ctx_left40 or "") + r.token,
                    "msg_index": int(r.msg_index) + 1,  # after the tools message
                    "role": "assistant" if r.role in FIELD else ("tools" if r.msg_index < 0 else r.role),
                    "field": field,
                    "char_offset": offset,
                    "layer": LAYER,
                    "bullets": bullets,
                    "joint_fve": None,
                }
                reads_out.write(json.dumps(row) + "\n")
            print(f"{setting}: {len(out_rows)} rows, {len(allreads)} reads ({len(dense)} dense + {len(copies)} prefix copies)")
    print(f"{n_placed}/{n_rows} reads placed at a character of their message field; the rest sit at the message's end")
    (reads_dir / "reads.provenance.json").write_text(
        json.dumps({"lens": "GLM-5.3 verbalizer rl1b_2/step_0100, block 51 (greedy + 4 samples at T=1, FVE per bullet)"})
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser())
