"""Per-prompt Google vs Alibaba self-ID counts in the WeirdChat Qwen3.6-27B parquets, plus prompt features."""

import glob
import re

import pandas as pd

GOOGLE = re.compile(
    r"trained by Google|developed by Google|created by Google|made by Google|by Google DeepMind|Gemini|Gemma", re.IGNORECASE
)
ALI = re.compile(r"Alibaba|Tongyi|Qwen", re.IGNORECASE)
rows = []
for f in sorted(glob.glob("transcript-viewer/data/hf/data/transcripts/qwen__qwen3.6-27b/*.parquet")):
    d = pd.read_parquet(f)
    for _, r in d.iterrows():
        msgs = list(r.messages)
        user = "\n".join(m["content"] for m in msgs if m["role"] == "user")
        sysm = "\n".join(m["content"] for m in msgs if m["role"] == "system")
        reply = "\n".join(m["content"] for m in msgs if m["role"] == "assistant")
        rows.append(
            dict(
                file=f.split("/")[-1][:-13],
                pid=r.prompt_id,
                nturn=len(msgs),
                sys=sysm,
                user=user,
                reply=reply,
                g=bool(GOOGLE.search(reply)),
                a=bool(ALI.search(reply)),
            )
        )
df = pd.DataFrame(rows)
df.to_pickle("data/runs/identity/2026-10-21_env_google/weirdchat_rows.pkl")
print(df.groupby("file")[["g", "a"]].sum(), len(df))
p = df.groupby("pid").agg(
    file=("file", "first"),
    n=("g", "size"),
    g=("g", "sum"),
    a=("a", "sum"),
    user=("user", "first"),
    sys=("sys", "first"),
    nturn=("nturn", "first"),
)
p = p[(p.g + p.a) > 0].copy()
p["gfrac"] = p.g / (p.g + p.a)
p["ulen"] = p.user.str.len()
p = p.sort_values("gfrac", ascending=False)
p.to_pickle("data/runs/identity/2026-10-21_env_google/weirdchat_prompts.pkl")
pd.set_option("display.width", 250)
print(p[["file", "n", "g", "a", "gfrac", "ulen", "nturn"]].to_string())
