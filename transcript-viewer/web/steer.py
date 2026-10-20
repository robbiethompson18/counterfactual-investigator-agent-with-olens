"""Probe / steering vectors for the case-study page, and behaviour rates under steering.

data/vectors.json   {vid: {name, spec, layer, info, v, t}}        built on the GPU pod (olens.vectors)
data/steer.jsonl    one record per run: {rid, tid, vid, alpha, layer, n, texts, labels, rate, state}
Runs sample n replies to the transcript's user turn with the vector added at its layer
(alpha * ||h|| * v, every position), then judge each with WeirdChat's transcript rubric (web/judge.py).
"""

import json
import threading
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VEC = ROOT / "data/vectors.json"
RUNS = ROOT / "data/steer.jsonl"
LOCK = threading.Lock()
V, R, PROBES = {}, {}, {}


def load():
    if VEC.exists():
        V.update(json.loads(VEC.read_text()))
    if RUNS.exists():
        for line in RUNS.open():
            r = json.loads(line)
            R[r["rid"]] = r  # later lines supersede earlier ones
    for r in R.values():
        if r["state"] == "sampling" or (r["state"] == "judging" and not r.get("texts")):
            r.update(state="error", error="viewer restarted mid-run")


def _save_runs(r):
    with LOCK, RUNS.open("a") as f:
        f.write(json.dumps(r) + "\n")


def public(v):
    return {k: x for k, x in v.items() if k != "v"}


def vectors():
    return sorted((dict(public(v), vid=k) for k, v in V.items()), key=lambda v: v["t"])


def build(pod, name, spec):
    if not (pod and pod.ready()):
        raise RuntimeError("the GPU pod isn't running: start it from the header")
    res = pod.call("/vector", {"spec": spec})
    vid = uuid.uuid4().hex[:8]
    with LOCK:
        V[vid] = {
            "name": name or spec_name(spec),
            "spec": spec,
            "layer": res["layer"],
            "info": res["info"],
            "v": res["v"],
            "t": time.time(),
        }
        VEC.write_text(json.dumps(V))
    return dict(public(V[vid]), vid=vid)


def spec_name(s):
    k = s["kind"]
    if k == "jpp":
        return f"J++ L{s['layer']} +{'/'.join(s.get('plus') or [])}" + (f" −{'/'.join(s['minus'])}" if s.get("minus") else "")
    if k == "ar":
        return f"AR L{s['layer']} “{s['text'][:40]}”"
    if k == "nla":
        return f"NLA-AR L42 “{s['text'][:40]}”"
    return f"({spec_name(s['a'])}) − ({spec_name(s['b'])})"


def delete(vid):
    with LOCK:
        V.pop(vid, None)
        VEC.write_text(json.dumps(V))


def probe(pod, tid, ids, vid):
    k = (tid, vid)
    if k not in PROBES:
        if not (pod and pod.ready()):
            raise RuntimeError("the GPU pod isn't running: start it from the header")
        v = V[vid]
        PROBES[k] = dict(pod.call("/probe", {"ids": ids, "layer": v["layer"], "v": v["v"]}), layer=v["layer"])
    return PROBES[k]


def runs(tid):
    return sorted((r for r in R.values() if r["tid"] == tid), key=lambda r: r["t"])


def abl_name(ab):
    return ", ".join(f"L{a['layer']} " + (f"head {a['head']}" if a["kind"] == "head" else a["kind"]) for a in ab)


def start(pod, judge, tid, prompt_ids, messages, vid, alpha, n=64, max_new_tokens=400, ablate=()):
    if not (pod and pod.ready()):
        raise RuntimeError("the GPU pod isn't running: start it from the header")
    v = V[vid] if vid else None
    r = {
        "rid": uuid.uuid4().hex[:8], "tid": tid, "vid": vid,
        "name": " + ".join(x for x in [v["name"] if v else None, f"ablate {abl_name(ablate)}" if ablate else None] if x) or "unsteered",
        "ablate": list(ablate),
        "layer": v["layer"] if v else None, "alpha": float(alpha) if v else 0.0, "n": int(n),
        "max_new_tokens": int(max_new_tokens), "state": "sampling", "t": time.time(),
    }
    R[r["rid"]] = r
    threading.Thread(target=_run, args=(pod, judge, r, prompt_ids, messages, v), daemon=True).start()
    return r


def resume(judge_for, messages_for):
    """Judge runs that were sampled but not judged when the viewer last stopped."""
    for r in list(R.values()):
        if r["state"] == "judging" and r.get("texts"):
            threading.Thread(target=_judge, args=(judge_for(r["tid"]), r, messages_for(r["tid"])), daemon=True).start()


def rejudge(judge, rid, messages):
    """Judge again the samples of a run whose judge call failed (label None)."""
    r = R[rid]
    if r["state"] != "done" or not r.get("texts"):
        raise RuntimeError("that run isn't finished")
    r["state"] = "judging"
    threading.Thread(target=_judge, args=(judge, r, messages, True), daemon=True).start()
    return r


def _judge(judge, r, messages, only_missing=False):
    try:
        todo = [i for i, x in enumerate(r.get("labels") or [None] * len(r["texts"])) if x is None] if only_missing else range(len(r["texts"]))
        out = dict(zip(todo, judge.judge_many(messages, [r["texts"][i] for i in todo])))
        old_l, old_e = r.get("labels") or [None] * len(r["texts"]), r.get("explanations") or [""] * len(r["texts"])
        r["labels"] = [out[i]["match"] if i in out else old_l[i] for i in range(len(r["texts"]))]
        r["explanations"] = [out[i]["explanation"] if i in out else old_e[i] for i in range(len(r["texts"]))]
        ok = [x for x in r["labels"] if x is not None]
        r.update(state="done", rate=sum(ok) / len(ok) if ok else None, n_judged=len(ok))
    except Exception as e:  # noqa: BLE001  (worker boundary: record and keep going)
        r.update(state="error", error=f"{type(e).__name__}: {e}")
    _save_runs(r)


def _run(pod, judge, r, prompt_ids, messages, v):
    try:
        res = pod.call(
            "/steer",
            {"ids": prompt_ids, "layer": v["layer"] if v else 40, "v": v["v"] if v else [0.0] * 5120, "alpha": r["alpha"],
             "n": r["n"], "max_new_tokens": r["max_new_tokens"], "temperature": 1.0, "seed": 0, "ablate": r["ablate"]},
            timeout=7200,
        )
        r.update(texts=res["texts"], gen_seconds=res["seconds"], state="judging")
        _save_runs(r)  # so a viewer restart can resume judging (resume())
        if judge is None:
            r.update(state="done", labels=None, rate=None, note="no judge configured (WCO_JUDGE_URL)")
        else:
            out = judge.judge_many(messages, res["texts"])
            r["labels"] = [o["match"] for o in out]
            r["explanations"] = [o["explanation"] for o in out]
            ok = [x for x in r["labels"] if x is not None]
            r.update(state="done", rate=sum(ok) / len(ok) if ok else None, n_judged=len(ok))
    except Exception as e:  # noqa: BLE001  (worker boundary: record and keep going)
        traceback.print_exc()
        r.update(state="error", error=f"{type(e).__name__}: {e}")
    _save_runs(r)
