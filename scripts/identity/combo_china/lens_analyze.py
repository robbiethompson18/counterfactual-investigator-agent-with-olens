"""Tally O-lens bullets that express the model's own creator / nationality / legal regime, per cue x position kind.
Prints counts and the matching bullets (with bfve) so they can be checked by hand."""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

D = Path("data/runs/identity/2026-10-21_combo_china")
CN_SELF = re.compile(
    r"Alibaba|Qwen|Tongyi|通义|阿里|千问|Chinese AI|AI (?:model|assistant|system)s? (?:developed |made |built |trained |operating |deployed )?(?:in|within|by a company in) (?:mainland )?China|"
    r"中国(?:的)?(?:法律|法规)|法律法规|网信|在中国境内|中国境内|(?:Chinese|PRC|China's) (?:laws?|regulations?|legal (?:requirements|framework)|cyber|content (?:rules|regulations))|"
    r"社会主义核心价值观|core socialist|国家(?:立场|安全)|official (?:position|stance|narrative|line)|敏感(?:话题|内容)|Great Firewall|censor",
    re.IGNORECASE,
)
WEST_SELF = re.compile(
    r"Anthropic|Claude|OpenAI|ChatGPT|\bGPT|Google|Gemini|Gemma|DeepMind|Meta\b|Llama|Lumen|\bNova\b|American (?:AI|company|model|values)|US-based|"
    r"Western (?:AI|model|company|values|perspective)|First Amendment|AWS|Bedrock|Microsoft|Azure",
    re.IGNORECASE,
)


def kind(where):
    return "hdr" if where.startswith("hdr") else where


def main():
    show = "-v" in sys.argv
    tally = defaultdict(Counter)
    ex = defaultdict(list)
    for line in (D / "lens.jsonl").open():
        d = json.loads(line)
        for r in d["reads"]:
            key = (d["cue"], "id" if d["probe"] == "direct" else "cens", kind(r["where"]))
            tally[key]["reads"] += 1
            for b, f in zip(r["bullets"], r["bfve"]):
                tally[key]["bullets"] += 1
                for name, rx in (("cn", CN_SELF), ("west", WEST_SELF)):
                    if rx.search(b):
                        tally[key][name] += 1
                        ex[(d["cue"], name)].append((round(f, 3), d["probe"], r["where"], r["layer"], b[:220]))
    print("cue\tprobe\tpos\treads\tbullets\tcn_self\twest_self")
    for k in sorted(tally):
        t = tally[k]
        print(*k, t["reads"], t["bullets"], t["cn"], t["west"], sep="\t")
    if show:
        for k in sorted(ex):
            print("\n###", k)
            for e in sorted(ex[k], reverse=True)[:15]:
                print(" ", e)


if __name__ == "__main__":
    main()
