"""Reproduce every number in docs/chive-critique.md.

uv run python -m chive_critique            # the whole report (first run downloads ~700 MB of Adam's files into data/.cache)
uv run python -m chive_critique adam       # only Adam's data: off-policy share, tool arms, thresholds, contradictions sample
uv run python -m chive_critique agam       # only Agam's 300-incident run
uv run python -m chive_critique extract    # re-stream Adam's investigations.jsonl files (~4.7 GB) into the run dir; not needed
                                           # unless you change what is kept, the run dir on HF already has the output
"""

import sys

from chive_critique import adam, agam

what = sys.argv[1] if len(sys.argv) > 1 else "all"
if what == "extract":
    adam.extract()
if what in ("all", "adam"):
    print("# Adam Karvonen's release: adamkarvonen/chive-data, 3 WildChat eval runs\n")
    print("## §1 Off-policy prefixes")
    adam.offpolicy()
    print("\n## §2 The 50pp / 15pp rule")
    adam.thresholds()
    print("\n## §2 Fisher's exact test instead")
    adam.significance()
    print("\n## §3 Contradictory ground truth (sampled)")
    adam.contradictions()
    print("\n## §4 Behaviours that are mistakes")
    adam.mistakes()
if what in ("all", "agam"):
    print("\n# Agam's 300-incident Qwen3.6-27B run: agu18dec/chive-forensics\n")
    agam.report()
