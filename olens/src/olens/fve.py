"""Whitened fraction-of-variance-explained, the oracle lens's reward / eval metric.

The whitener moments are the ones the agu18dec AR was trained against
(`whitening_iolens_pooled_L{L}`: mu, cov over ~2.1M chat + pretraining activations).
Exactly how the original code centres and regularises is not published; the choices below are
exposed as arguments so they can be checked against the reported numbers.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file
from scipy.optimize import nnls

from .lens import CKPT

WHITENERS = CKPT / "local_workspace/data/whiteners/pooled"


class Whitener:
    def __init__(self, layer, ridge=0.0, root=WHITENERS, device="cpu", dtype=torch.float64):
        """The eigendecomposition is done in float64 on the CPU; W and mu are then stored on
        `device` in `dtype`."""
        m = load_file(Path(root) / f"whitening_iolens_pooled_L{layer}.safetensors")
        evals, evecs = torch.linalg.eigh(m["cov"].double())
        evals = evals.clamp_min(0) + ridge * evals.mean()
        self.min_eig, self.max_eig = evals.min().item(), evals.max().item()
        self.W = ((evecs * evals.rsqrt()) @ evecs.T).to(device, dtype)  # Σ^-1/2 (symmetric)
        self.W_inv = ((evecs * evals.sqrt()) @ evecs.T).to(device, dtype)  # Σ^1/2
        self.mu = m["mu"].to(device, dtype)

    def __call__(self, x, center=True):
        x = torch.as_tensor(x).to(self.W.device, self.W.dtype)
        if center:
            x = x - self.mu
        return x @ self.W  # W symmetric

    def unwhiten(self, z, mu_scale=1.0):
        """Inverse of __call__: Σ^1/2 z + mu_scale * mu."""
        return torch.as_tensor(z).to(self.W.device, self.W.dtype) @ self.W_inv + mu_scale * self.mu


def fve_single(z, zhat):
    """Best non-negative scaling of one predicted direction: FVE = cos^2 if cos > 0 else 0."""
    cos = torch.nn.functional.cosine_similarity(z, zhat, dim=-1)
    return cos.clamp_min(0) ** 2, cos


def fve_nnls(z, D):
    """z: [d] target, D: [k, d] directions. Non-negative least squares refit of z on D.
    Returns (FVE, coefficients)."""
    z = z.double().cpu().numpy()
    D = D.double().cpu().numpy()
    c, rnorm = nnls(D.T, z)
    return 1.0 - rnorm**2 / float(z @ z), c
