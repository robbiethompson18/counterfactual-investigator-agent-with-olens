"""The text that actually follows a position, as AR directions ("continuation atoms"), and the activation with those
directions fitted and projected out.

Atom k at position p is AR(ids[p+1 : p+1+k]), k = 1..n: the AR's guess at the activation just before that text, i.e.
the activation you would expect from what the model goes on to say. One AR forward per position gives all n
(AR.prefixes). Shared by the residual reads (scripts/olens_residual_reads.py), the RL free-atom reward and the
FVE-over-continuation metric.

Everything is fitted in the pooled whitener's space, like the lens's FVE: z = Σ^-1/2 (h − μ) (olens.fve.Whitener).
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import nnls as _nnls


def atoms(ar, ids, positions, layer, n=32, batch=64):
    """Raw AR outputs for every prefix of the n tokens after each position: [P, n, d], plus [P] counts of real prefixes
    (fewer than n near the end of ids; the missing rows are 0)."""
    spans = [list(ids[p + 1 : p + 1 + n]) for p in positions]
    out = []
    for i in range(0, len(spans), batch):
        a = ar.prefixes(spans[i : i + batch], layer)
        out.append(F.pad(a, (0, 0, 0, n - a.shape[1])))
    return torch.cat(out), torch.tensor([len(s) for s in spans])


def whiten_atoms(W, A, counts):
    """[P, n, d] raw atoms -> whitened, with the padding rows kept at exactly 0 (whitening 0 would give −Σ^-1/2 μ)."""
    keep = torch.arange(A.shape[1], device=A.device)[None, :] < counts.to(A.device)[:, None]
    return W(A).to(A.dtype) * keep[..., None]


def nnls(z, D):
    """Batched non-negative least squares of whitened targets on whitened atoms. z [B, d], D [B, k, d] (zero rows are
    fine). Returns (coefs [B, k], fit [B, d], fve [B]), fve = 1 − |z − fit|² / |z|². Solved in Gram form (k x k per
    problem) on the CPU."""
    G = (D @ D.transpose(1, 2)).double().cpu().numpy()
    b = (D @ z[..., None])[..., 0].double().cpu().numpy()
    C = torch.tensor(np.stack([nnls_gram(G[i], b[i]) for i in range(len(b))]), dtype=z.dtype, device=z.device)
    fit = (C[..., None] * D).sum(1)
    return C, fit, 1 - ((z - fit) ** 2).sum(-1) / (z * z).sum(-1)


def nnls_gram(G, b):
    """argmin_{c >= 0} cᵀGc − 2bᵀc, via scipy's NNLS on a square root of G. The prefixes are near-collinear, so G is
    ill-conditioned: directions with eigenvalue < 1e-9 of the largest are dropped."""
    lam, V = np.linalg.eigh(G)
    keep = lam > 1e-9 * max(lam.max(), 1e-30)
    if not keep.any():
        return np.zeros(len(b))
    s = np.sqrt(lam[keep])
    return _nnls(s[:, None] * V[:, keep].T, (V[:, keep].T @ b) / s)[0]


def edit(W, h, fit, beta=1.0, gamma=1.0, mu_scale=1.0):
    """h' = mu_scale·μ + gamma·Σ^1/2 (z − beta·fit), z = Σ^-1/2 (h − μ): the raw activation with beta times the fitted
    continuation removed. beta = 0, gamma = 1, mu_scale = 1 gives h back. The lens normalises its input, so gamma only
    matters through mu_scale·μ, which it leaves fixed."""
    return W.unwhiten(gamma * (W(h) - beta * fit), mu_scale)
