"""Control: does a non-Qwen model (Gemma-3-27B via OpenRouter) also recommend Alibaba when the behaviour probe is in Chinese?
If yes, the Chinese -> Alibaba recommendation shift is market/language knowledge, not Qwen's self-belief."""

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/env_other")
from classify import first
from envs import BEHAVIOUR
from qwen import openrouter, or_spent

print("spent so far", round(or_spent(), 3))
out = {}
for lang in ("en", "zh"):
    msgs = [{"role": "user", "content": BEHAVIOUR[lang].split("：", 1)[-1].split(": ", 1)[-1]}]
    with ThreadPoolExecutor(8) as ex:
        answers = list(
            ex.map(lambda _: openrouter("google/gemma-3-27b-it", msgs, who="env_other", temperature=0.7, max_tokens=600), range(16))
        )
    out[lang] = answers
    print(lang, msgs[0]["content"][:60], Counter(first(a) for a in answers))
json.dump(out, open("data/runs/identity/2026-10-21_env_other/gemma_behaviour_control.json", "w"), ensure_ascii=False, indent=1)
print("spent now", round(or_spent(), 3))
