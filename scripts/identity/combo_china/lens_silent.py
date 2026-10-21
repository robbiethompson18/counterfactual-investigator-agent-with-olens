"""O-lens on strictly-silent CoTs (Haiku: no self-identity, no cue reference, no own-jurisdiction) for Xinjiang:
uncensored ones under US/EU locale vs censored ones under CN locale / baseline. 7 layers 28..52; header read once per cue."""

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/combo_china")
from cues import PROBES, conversation
from grade import uncensored
from lens import cot_positions
from qwen import header_positions, read, render

D = Path("data/runs/identity/2026-10-21_combo_china")
OUT = D / "lens_silent.jsonl"
LAYERS = (28, 32, 36, 40, 44, 48, 52)
PLAN = {"loc_us": (True, 4), "res_eu": (True, 4), "loc_cn": (False, 4), "helpful": (False, 2)}  # cue -> (want uncensored, how many)

lab = json.load(open(D / "cot_labels.json"))
S = {(d["cue"], d["probe"], d["k"]): d for d in map(json.loads, (D / "samples.jsonl").open())}
with OUT.open("w") as f:
    for cue, (want_unc, m) in PLAN.items():
        picks = []
        for key, v in sorted(lab.items(), key=lambda kv: int(kv[0].split("|")[2])):
            c, p, k = key.split("|")
            s = S[(c, p, int(k))]
            if (
                c == cue
                and p == "xinjiang"
                and s["finish_reason"] == "stop"
                and not (v["self_identity"] or v["cue_ref"] or v["own_jurisdiction"])
                and uncensored("xinjiang", s["content"]) == want_unc
            ):
                picks.append(s)
        picks = picks[:m]
        for i, s in enumerate(picks):
            ids, toks = render(conversation(cue, PROBES["xinjiang"]), s)
            named = cot_positions(toks, s)
            if i == 0:
                hp = header_positions(ids)
                named = {"hdr_im_end": hp[0], "hdr_assistant": hp[2]} | named
            items = [(n, pos, L) for n, pos in named.items() for L in LAYERS]
            res = read(ids, [(pos, L) for _, pos, L in items])
            rec = [{"where": n, "pos": pos, "tok": toks[pos], "layer": L, **r[0]} for (n, pos, L), r in zip(items, res)]
            f.write(
                json.dumps({"cue": cue, "probe": "xinjiang", "k": s["k"], "uncensored": want_unc, "reads": rec}, ensure_ascii=False) + "\n"
            )
            f.flush()
        print(cue, len(picks), "silent samples read", flush=True)
