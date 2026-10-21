"""Per cue x probe: among Haiku-labelled samples whose answer is uncensored, how many CoTs mention identity / the cue /
own (Chinese) jurisdiction. 'silent' = none of self_identity and cue_ref."""

import json
import sys
from collections import defaultdict

sys.path.insert(0, "scripts/identity/combo_china")
from grade import offtopic, tiananmen_label, uncensored

lab = json.load(open("data/runs/identity/2026-10-21_combo_china/cot_labels.json"))
S = {(d["cue"], d["probe"], d["k"]): d for d in map(json.loads, open("data/runs/identity/2026-10-21_combo_china/samples.jsonl"))}
t = defaultdict(lambda: defaultdict(int))
for key, v in lab.items():
    cue, probe, k = key.split("|")
    s = S[(cue, probe, int(k))]
    a = s["content"]
    if not a or s["finish_reason"] != "stop" or offtopic(probe, a):
        continue
    unc = tiananmen_label(a) == "military_or_deaths" if probe == "tiananmen" else uncensored(probe, a)
    c = t[(cue, probe)]
    c["n"] += 1
    if unc:
        c["unc"] += 1
        c["unc_self"] += bool(v.get("self_identity"))
        c["unc_cue"] += bool(v.get("cue_ref"))
        c["unc_jur"] += bool(v.get("own_jurisdiction"))
        c["unc_silent"] += not (v.get("self_identity") or v.get("cue_ref"))
        c["unc_strict"] += not (v.get("self_identity") or v.get("cue_ref") or v.get("own_jurisdiction"))
    c["strict_all"] += not (v.get("self_identity") or v.get("cue_ref") or v.get("own_jurisdiction"))
    c["strict_all_unc"] += unc and not (v.get("self_identity") or v.get("cue_ref") or v.get("own_jurisdiction"))
print(
    "cue\tprobe\tn\tuncens\tunc&self\tunc&cue\tunc&juris\tunc&silent(no self/cue)\tunc&strict(no self/cue/juris)\t|strict-silent CoTs: uncens/n"
)
out = {}
for (cue, probe), c in sorted(t.items()):
    print(
        cue,
        probe,
        c["n"],
        c["unc"],
        c["unc_self"],
        c["unc_cue"],
        c["unc_jur"],
        c["unc_silent"],
        c["unc_strict"],
        f"{c['strict_all_unc']}/{c['strict_all']}",
        sep="\t",
    )
    out[f"{cue}|{probe}"] = dict(c)
json.dump(out, open("data/runs/identity/2026-10-21_combo_china/silent.json", "w"), indent=1)
