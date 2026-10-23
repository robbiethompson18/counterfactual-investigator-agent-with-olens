"""How much an O-lens read explains beyond the text Qwen actually wrote next ("novelty").

Per position, with z the whitened activation, C its continuation atoms (olens.continuation: AR of prefixes of the next
tokens) and B the read's bullet atoms (AR of bullet_prefix + bullet), all NNLS in the pooled whitened space:

    cont  = FVE(z; C ∪ T)             what the visible text explains
    lens  = FVE(z; B)                 the read's ordinary joint FVE
    gain  = FVE(z; C ∪ T ∪ B) − cont  what the read adds beyond it
    n_b   = FVE(z; C ∪ T ∪ {b}) − cont, per bullet

T is optional and shared by every position of a transcript: AR atoms of every 32-token chunk of it, so the question
becomes "is this anywhere in the transcript?" rather than "is it in the next 32 tokens?". Adding atoms never lowers an
NNLS fit, so gain is only meaningful against a null: the same gain with B swapped for bullets read somewhere else
(null_gain).
"""

import numpy as np
import torch
from scipy.optimize import nnls


def fve1(z, D):
    """FVE of one z [d] on atoms D [k, d] (float64 numpy), all-zero rows dropped. Plain NNLS on D.T, or on R with
    Q.T z after a thin QR of D.T when k > 64 (same optimum, half the time at k ~ 500; residual² = |z|² − |Q.T z|² + the
    reduced residual²). Not a Gram-form solve: on near-collinear atoms (a 1-token continuation, repeated bullets) that
    returned ~1e18 coefficients."""
    D = D[np.abs(D).sum(1) > 0]
    zz = float(z @ z)
    if not len(D):
        return 0.0
    if len(D) <= 64:
        return 1.0 - nnls(D.T, z)[1] ** 2 / zz
    Q, R = np.linalg.qr(D.T)
    y = Q.T @ z
    return 1.0 - (max(zz - float(y @ y), 0.0) + nnls(R, y)[1] ** 2) / zz


def _np(x, d):
    return x.double().numpy().reshape(-1, d) if isinstance(x, torch.Tensor) else np.zeros((0, d))


def scores(z, C, B, T=None):
    """z [P, d], C [P, n, d], B: P tensors [k_i, d] (k_i >= 1), T [m, d] or None. -> {cont, lens, joint, gain: [P];
    n_b, solo: [sum k_i] in B's order}. Rows are built one at a time, so a large T is never copied P times."""
    d = z.shape[1]
    z, T = z.double().numpy(), _np(T, d)
    out = {k: [] for k in ("cont", "lens", "joint", "n_b", "solo")}
    for i, b in enumerate(B):
        b, base = _np(b, d), np.concatenate([_np(C[i], d), T])
        cont = fve1(z[i], base)
        out["cont"].append(cont)
        out["lens"].append(fve1(z[i], b))
        out["joint"].append(fve1(z[i], np.concatenate([base, b])))
        for bb in b:
            out["n_b"].append(fve1(z[i], np.concatenate([base, bb[None]])) - cont)
            out["solo"].append(fve1(z[i], bb[None]))
    s = {k: torch.tensor(v, dtype=torch.float32) for k, v in out.items()}
    s["gain"] = s["joint"] - s["cont"]
    return s


def null_gain(z, C, cont, B_null, T=None):
    """gain with each position's bullets replaced by B_null (P tensors [k_i, d], may be empty)."""
    d = z.shape[1]
    z, T = z.double().numpy(), _np(T, d)
    g = [fve1(z[i], np.concatenate([_np(C[i], d), T, _np(b, d)])) - float(cont[i]) for i, b in enumerate(B_null)]
    return torch.tensor(g, dtype=torch.float32)
