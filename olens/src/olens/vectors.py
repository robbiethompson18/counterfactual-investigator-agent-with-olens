"""Probe / steering directions in the residual stream after block L, built from the lenses, and a
steering hook.

Specs (JSON-able dicts):
  {"kind": "jpp", "layer": L, "plus": [str], "minus": [str]}
      the direction that raises the J++ logits of `plus` (minus those of `minus`) at layer L:
      J[L]^T (g * W_U[t]) summed over tokens, g the final norm's gain. Each string's first token.
  {"kind": "ar", "layer": L, "text": str}     agu18dec AR prediction for `text`, minus the whitener mean
  {"kind": "nla", "text": str}                NLA reconstructor (L42) prediction, minus the calibration mean
  {"kind": "diff", "a": spec, "b": spec}      a - b (same layer)
Every result is a unit vector [d] plus its layer.
"""

import torch

from .lens import blocks


class Vectors:
    def __init__(self, tok, model, readout=None, jpp=None, critic=None):
        self.tok, self.model, self.readout, self.jpp, self.critic = tok, model, readout, jpp, critic

    def first_token(self, s):
        ids = self.tok(s, add_special_tokens=False).input_ids
        assert ids, f"empty token string {s!r}"
        return ids[0]

    @torch.no_grad()
    def build(self, spec):
        """-> (unit vector [d] fp32 on the model device, layer, info)"""
        k = spec["kind"]
        if k == "diff":
            a, La, ia = self.build_raw(spec["a"])
            b, Lb, ib = self.build_raw(spec["b"])
            assert La == Lb, f"diff of vectors at different layers ({La}, {Lb})"
            return unit(a - b), La, {"a": ia, "b": ib, "cos_ab": round(float(unit(a) @ unit(b)), 4)}
        v, L, info = self.build_raw(spec)
        return unit(v), L, info

    @torch.no_grad()
    def build_raw(self, spec):
        k = spec["kind"]
        if k == "jpp":
            L = int(spec["layer"])
            i = self.jpp.layers.index(L)
            g = self.model.model.norm(torch.ones(1, self.model.config.hidden_size, device=self.model.device, dtype=torch.bfloat16)).float()[0]
            W = self.model.lm_head.weight
            plus = [self.first_token(s) for s in spec.get("plus") or []]
            minus = [self.first_token(s) for s in spec.get("minus") or []]
            u = torch.zeros_like(g)
            if plus:
                u += W[plus].float().mean(0)
            if minus:
                u -= W[minus].float().mean(0)
            v = self.jpp.J[i].float().T @ (g * u)
            return v, L, {"plus": [self.tok.decode([t]) for t in plus], "minus": [self.tok.decode([t]) for t in minus]}
        if k == "ar":
            L = int(spec["layer"])
            ids = self.tok(spec["text"], add_special_tokens=False).input_ids[:32]
            p = self.readout.ar([ids], L)[0]
            return p - self.readout.whitener(L).mu.to(p.device, p.dtype).view(-1), L, {"tokens": len(ids)}
        if k == "nla":
            p = self.critic.predict([spec["text"]])[0]
            return p - self.critic.mu, 42, {}
        raise ValueError(f"unknown vector kind {k!r}")

    @torch.no_grad()
    def probe(self, ids, layer, v):
        """cosine of the block-`layer` residual with v at every position, and the projection."""
        from .lens import capture

        h = capture(self.model, ids, [layer])[layer]
        v = unit(v.to(h.device))
        proj = h @ v
        return {"cos": [round(float(x), 4) for x in proj / h.norm(dim=-1).clamp_min(1e-9)], "proj": [round(float(x), 2) for x in proj]}

    @torch.no_grad()
    def steer(self, ids, layer, v, alpha, n=64, max_new_tokens=400, temperature=1.0, top_p=1.0, seed=0, batch=64, ablate=()):
        """Sample n replies to the prompt `ids` (ending in the generation prompt) with
        h <- h + alpha * ||h|| * v at the output of block `layer`, every position, and the
        components in `ablate` zeroed at every position: [{layer, kind: attn|mlp|head, head}]."""
        v = unit(v.to(self.model.device)).to(torch.bfloat16)

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + alpha * h.norm(dim=-1, keepdim=True) * v
            return (h, *out[1:]) if isinstance(out, tuple) else h

        hd = blocks(self.model)[layer].register_forward_hook(hook) if alpha else None
        abl = [ablation_hook(self.model, a) for a in ablate]
        texts = []
        try:
            torch.manual_seed(seed)
            for i in range(0, n, batch):
                B = min(batch, n - i)
                x = torch.tensor([ids] * B, device=self.model.device)
                out = self.model.generate(
                    input_ids=x,
                    attention_mask=torch.ones_like(x),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=0,
                    pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                )
                texts += self.tok.batch_decode(out[:, len(ids) :], skip_special_tokens=True)
        finally:
            if hd:
                hd.remove()
            for h in abl:
                h.remove()
        return texts


def ablation_hook(model, a):
    """Zero-ablate one component at every position: a whole attention block's or MLP's write, or
    one attention head (its slice of the output projection's input)."""
    b = blocks(model)[int(a["layer"])]
    attn = b.self_attn if hasattr(b, "self_attn") else b.linear_attn
    if a["kind"] == "mlp":
        return b.mlp.register_forward_hook(lambda m, i, o: torch.zeros_like(o))
    if a["kind"] == "attn":
        return attn.register_forward_hook(lambda m, i, o: (torch.zeros_like(o[0]), *o[1:]) if isinstance(o, tuple) else torch.zeros_like(o))
    proj = attn.o_proj if hasattr(attn, "o_proj") else attn.out_proj
    H = getattr(attn, "num_v_heads", None) or model.config.num_attention_heads
    hd, h = proj.in_features // H, int(a["head"])
    assert 0 <= h < H, f"layer {a['layer']} has {H} heads"

    def pre(m, i):
        x = i[0].clone()
        x[..., h * hd : (h + 1) * hd] = 0
        return (x, *i[1:])

    return proj.register_forward_pre_hook(pre)


@torch.no_grad()
def ar_per_layer(tok, readout, n_blocks, text, minus=None):
    """{block: dir} for every block: the AR of `text` (minus the AR of `minus`, else minus the
    whitener mean) at the AR's own layers (20..60 step 4); other blocks use the nearest AR layer."""
    ar_layers = list(readout.ar.layers)
    ids = tok(text, add_special_tokens=False).input_ids[:32]
    a = readout.ar([ids] * len(ar_layers), ar_layers)
    if minus:
        b = readout.ar([tok(minus, add_special_tokens=False).input_ids[:32]] * len(ar_layers), ar_layers)
    else:
        b = torch.stack([readout.whitener(L).mu.to(a.device, a.dtype).view(-1) for L in ar_layers])
    by = {L: unit(a[j] - b[j]) for j, L in enumerate(ar_layers)}
    return {i: by[min(ar_layers, key=lambda L: abs(L - i))] for i in range(n_blocks)}


def unit(v):
    return v / v.norm().clamp_min(1e-12)


@torch.no_grad()
def contributions(model, ids, pos, vecs, layered=None):
    """What each attention block (full or linear) and each MLP writes into the residual stream at
    token `pos`, compared by cosine with: the final residual (block-63 output), the residual right
    after that sublayer's block, and each of `vecs` ({name: [d]}). -> {layers: [{attn, mlp}]},
    each {norm, cos: {name: c}}."""
    got = {}
    hs = []
    for i, b in enumerate(blocks(model)):
        attn = b.self_attn if hasattr(b, "self_attn") else b.linear_attn
        hs.append(attn.register_forward_hook(lambda m, a, o, i=i: got.__setitem__((i, "attn"), (o[0] if isinstance(o, tuple) else o)[0, pos].float())))
        hs.append(b.mlp.register_forward_hook(lambda m, a, o, i=i: got.__setitem__((i, "mlp"), o[0, pos].float())))
        proj = attn.o_proj if hasattr(attn, "o_proj") else attn.out_proj  # input = concatenated per-head outputs
        hs.append(proj.register_forward_pre_hook(lambda m, a, i=i: got.__setitem__((i, "heads_in"), a[0][0, pos].float())))
        hs.append(b.register_forward_hook(lambda m, a, o, i=i: got.__setitem__((i, "out"), (o[0] if isinstance(o, tuple) else o)[0, pos].float())))
    try:
        model.model(input_ids=torch.as_tensor(ids, device=model.device).view(1, -1), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    n_blocks = len(blocks(model))
    final = got[(n_blocks - 1, "out")]  # block outputs from hooks: hidden_states[-1] is after the final norm
    dirs = {"final residual": final, **{k: torch.as_tensor(v, device=final.device).float() for k, v in vecs.items()}}
    dirs = {k: unit(v) for k, v in dirs.items()}
    per_layer = {k: {L: unit(v.to(final.device)) for L, v in d.items()} for k, d in (layered or {}).items()}  # name -> {block: dir}
    rows = []
    for i in range(len(blocks(model))):
        row = {}
        for kind in ("attn", "mlp"):
            c = got[(i, kind)]
            n = c.norm().clamp_min(1e-12)
            cos = {k: round(float(c @ d / n), 4) for k, d in dirs.items()}
            for k, d in per_layer.items():
                cos[k] = round(float(c @ d[i] / n), 4)
            cos["block output"] = round(float(c @ unit(got[(i, "out")]) / n), 4)
            row[kind] = {"norm": round(float(n), 2), "cos": cos}
        # per head: W_O[:, head slice] @ head output; the heads sum to the attention write (no bias)
        b = blocks(model)[i]
        attn = b.self_attn if hasattr(b, "self_attn") else b.linear_attn
        W = (attn.o_proj if hasattr(attn, "o_proj") else attn.out_proj).weight.float()
        x = got[(i, "heads_in")]
        H = getattr(attn, "num_heads", None) or getattr(attn, "num_v_heads", None) or model.config.num_attention_heads
        hd = x.shape[0] // H
        per = torch.einsum("dhk,hk->hd", W.view(W.shape[0], H, hd), x.view(H, hd))  # [H, d]
        pn = per.norm(dim=-1).clamp_min(1e-12)
        heads = [{"norm": round(float(pn[h]), 2)} for h in range(H)]
        for k, d in list(dirs.items()) + [("block output", unit(got[(i, "out")]))] + [(k, d[i]) for k, d in per_layer.items()]:
            cs = per @ d / pn
            for h in range(H):
                heads[h].setdefault("cos", {})[k] = round(float(cs[h]), 4)
        row["heads"] = heads
        o = got[(i, "out")]  # the residual stream after this block, against the same directions
        on = o.norm().clamp_min(1e-12)
        rc = {k: round(float(o @ d / on), 4) for k, d in dirs.items()}
        rc.update({k: round(float(o @ d[i] / on), 4) for k, d in per_layer.items()})
        rc["block output"] = 1.0
        row["resid"] = {"norm": round(float(on), 2), "cos": rc}
        row["kind"] = "full" if hasattr(b, "self_attn") else "linear"
        rows.append(row)
    return {"layers": rows, "resid_norm": [round(float(got[(i, "out")].norm()), 1) for i in range(n_blocks)]}


@torch.no_grad()
def attention_pattern(model, ids, pos, layer, head, top=12):
    """Where full-attention head `head` of block `layer` looks from token `pos`: softmax weights over
    tokens 0..pos, plus each token's weighted write ‖w_j · v_j W_O,h‖ (what it actually moves into
    the residual stream, before the output gate). Linear-attention blocks have no softmax pattern."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    b = blocks(model)[layer]
    if not hasattr(b, "self_attn"):
        return {"error": f"L{layer} is a linear-attention (gated delta net) block: no attention pattern"}
    attn, got = b.self_attn, {}

    def pre(m, args, kwargs):
        got["h"] = kwargs.get("hidden_states", args[0] if args else None)
        got["pe"] = kwargs["position_embeddings"]

    hd = attn.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        model.model(input_ids=torch.as_tensor(ids, device=model.device).view(1, -1), use_cache=False)
    finally:
        hd.remove()
    h, (cos, sin) = got["h"], got["pe"]
    D, shape = attn.head_dim, (*h.shape[:-1], -1, attn.head_dim)
    q, _gate = torch.chunk(attn.q_proj(h).view(*h.shape[:-1], -1, D * 2), 2, dim=-1)
    q = attn.q_norm(q.view(shape)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(h).view(shape)).transpose(1, 2)
    v = attn.v_proj(h).view(shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    kv = head // (q.shape[1] // k.shape[1])
    s = (k[0, kv, : pos + 1].float() @ q[0, head, pos].float()) * attn.scaling
    w = s.softmax(-1)
    Wo = attn.o_proj.weight[:, head * D : (head + 1) * D].float()
    write = (w[:, None] * (v[0, kv, : pos + 1].float() @ Wo.T)).norm(dim=-1)
    order = w.argsort(descending=True)[:top]
    return {"layer": layer, "head": head, "weights": [round(float(x), 4) for x in w], "write": [round(float(x), 3) for x in write],
            "top": [int(i) for i in order]}
