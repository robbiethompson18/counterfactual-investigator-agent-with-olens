"""How much an O-lens read explains beyond the text Qwen actually wrote next ("novelty").

Per position, with z the whitened activation, C its continuation atoms (olens.continuation: AR of prefixes of the next
tokens) and B the read's bullet atoms (AR of bullet_prefix + bullet), all NNLS in the pooled whitened space:

    cont  = FVE(z; C)             what the visible upcoming text explains
    lens  = FVE(z; B)             the read's ordinary joint FVE
    gain  = FVE(z; C ∪ B) − cont  what the read adds beyond the continuation
    n_b   = FVE(z; C ∪ {b}) − cont, per bullet

Adding atoms never lowers an NNLS fit, so gain is only meaningful against a null: the same gain with B swapped for
bullets read somewhere else (null_gain). C can be any base set, e.g. the continuation plus AR atoms of every 32-token
chunk of the transcript ("is this anywhere in the transcript?").
"""

import torch

from .continuation import nnls


def fve(z, D):
    return nnls(z, D)[2]


def stack(atoms, d):
    """P lists of [k_i, d] -> [P, max k, d], zero rows as padding (NNLS ignores them)."""
    out = torch.zeros(len(atoms), max(1, max(len(a) for a in atoms)), d)
    for i, a in enumerate(atoms):
        out[i, : len(a)] = a.float()
    return out


def scores(z, C, B):
    """z [P, d], C [P, n, d], B: P tensors [k_i, d] (k_i >= 1). -> {cont, lens, joint, gain: [P]; n_b, solo: [sum k_i]
    in B's order}."""
    z, C = z.float(), C.float()
    Bp = stack(B, z.shape[1])
    cont, lens, joint = fve(z, C), fve(z, Bp), fve(z, torch.cat([C, Bp], 1))
    owner = torch.tensor([i for i, b in enumerate(B) for _ in range(len(b))])
    flat = torch.cat([b.float() for b in B])[:, None]
    n_b = fve(z[owner], torch.cat([C[owner], flat], 1)) - cont[owner]
    return {"cont": cont, "lens": lens, "joint": joint, "gain": joint - cont, "n_b": n_b, "solo": fve(z[owner], flat)}


def null_gain(z, C, cont, B_null):
    """gain with each position's bullets replaced by B_null (P tensors [k_i, d], may be empty)."""
    z, C = z.float(), C.float()
    return fve(z, torch.cat([C, stack(B_null, z.shape[1])], 1)) - cont
