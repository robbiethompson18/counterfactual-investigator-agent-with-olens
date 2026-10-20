"""Vector / probe / steering / residual-write handlers for pod/server.py, reloaded on every call
(with olens.vectors) so changes to them go live without reloading the model. S is the server's
state (model, tokenizer, lenses); handlers must not keep state of their own."""

import time

import torch
from olens.lens import capture

S = {}


def vectors():
    from olens.vectors import Vectors

    return Vectors(S["tok"], S["model"], S["readout"], S.get("jpp"), S.get("critic"))


def do_nla_score(req):
    """{ids, pos, text} -> {cos, fve}: the NLA reconstructor on an edited explanation."""
    ids = [int(i) for i in req["ids"]]
    v = capture(S["model"], ids, [42])[42][[int(req["pos"]) % len(ids)]].cpu()
    return S["critic"].score([req["text"]], v)[0]


def do_attn(req):
    from olens.vectors import attention_pattern

    ids = [int(i) for i in req["ids"]]
    return attention_pattern(S["model"], ids, int(req["pos"]) % len(ids), int(req["attn"]["layer"]), int(req["attn"]["head"]))


def do_contrib(req):
    if req.get("attn"):  # routed through /contrib so it works without restarting the server
        return do_attn(req)
    from olens.vectors import contributions

    ids = [int(i) for i in req["ids"]]
    from olens.lens import blocks
    from olens.vectors import ar_per_layer

    nb = len(blocks(S["model"]))
    layered = {k: ar_per_layer(S["tok"], S["readout"], nb, x["text"], x.get("minus")) for k, x in (req.get("layered") or {}).items()}
    pos = int(req["pos"]) % len(ids)
    out = contributions(S["model"], ids, pos, req.get("vecs") or {}, layered)
    out["ar_fve"] = {k: ar_fve(ids, pos, x["text"], x.get("minus")) for k, x in (req.get("layered") or {}).items()}
    return out


def ar_fve(ids, pos, text, minus=None):
    """{AR layer: {a, b}}: each text's own FVE as an O-lens bullet for this token's activation at
    that layer (cos^2 in the whitened space, 0 if the cosine is negative), as in the readouts."""
    from olens.fve import fve_single

    ro, tok = S["readout"], S["tok"]
    Ls = list(ro.ar.layers)
    acts = capture(S["model"], ids, Ls)
    texts = [text] + ([minus] if minus else [])
    spans = [tok(t, add_special_tokens=False).input_ids[:32] for t in texts]
    D = ro.ar([s for s in spans for _ in Ls], [L for _ in spans for L in Ls])
    res = {}
    for j, L in enumerate(Ls):
        W = ro.whitener(L)
        z = W(acts[L][pos].to(D.device))
        f, _ = fve_single(z[None], W(D[[i * len(Ls) + j for i in range(len(texts))]]))
        res[str(L)] = {"a": round(float(f[0]), 4), **({"b": round(float(f[1]), 4)} if minus else {})}
    return res


def do_vector(req):
    v, L, info = vectors().build(req["spec"])
    return {"v": [round(float(x), 6) for x in v.cpu()], "layer": L, "info": info}


def do_probe(req):
    return vectors().probe([int(i) for i in req["ids"]], int(req["layer"]), torch.tensor(req["v"]))


def do_steer(req):
    t0 = time.time()
    texts = vectors().steer(
        [int(i) for i in req["ids"]], int(req["layer"]), torch.tensor(req["v"]), float(req["alpha"]),
        n=int(req.get("n", 64)), max_new_tokens=int(req.get("max_new_tokens", 400)),
        temperature=float(req.get("temperature", 1.0)), seed=int(req.get("seed", 0)), ablate=req.get("ablate") or (),
    )
    return {"texts": texts, "seconds": round(time.time() - t0, 1)}
