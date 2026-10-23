"""Novelty reward for oracle-lens RL: the whitened NNLS FVE that a read's bullets add on top of the text Qwen actually wrote next.

Free atoms C: the AR's directions for every prefix (1..32 tokens) of the realized continuation (olens.continuation). Bullet atoms
B: the AR's directions for the read's first 4 bullets. Everything whitened with the pooled whitener; targets unit-normed.

    marginal(z; B) = FVE(z; C ∪ B) - FVE(z; C)          >= 0: NNLS can always give B zero weight
    reward = marginal(own; B) - beta * mean_j marginal(distractor_j; B)

A bullet that restates the continuation adds ~nothing, because C already spans it. marginal is the runtime-residual fork's
`gain_fve`. The distractor term is our reconstruction of Agam's `reward_n_distractors 8, reward_contrastive_beta 1.0` (his
formula isn't published): bullets that add FVE to any activation at that layer earn nothing.
"""

import numpy as np
import torch
from olens.readout import bullets
from scipy.optimize import nnls

N_CONT = 32  # continuation prefixes used as free atoms
K_MAX = 4  # bullets scored per read (Agam's reward_k_max)


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def fve_batch(z, D, ridge=1e-8):
    """FVE of the non-negative least-squares fit of each z [B, d] on its atoms D [B, k, d] (all-zero rows dropped).
    Gram form in float64, solved through a Cholesky factor of G + tiny ridge, so coefficients stay bounded on near-collinear
    atoms (the eigen-truncated Gram solve this replaces blew up there). ~50x faster than one raw scipy NNLS per row
    (olens.continuation.nnls); resample_check.py prints the gap between the two on real atoms."""
    D, z = D.double(), z.double().to(D.device)
    G = (D @ D.transpose(1, 2)).cpu().numpy()
    b = (D @ z[..., None])[..., 0].cpu().numpy()
    zz = (z * z).sum(-1).cpu().numpy()
    out = np.zeros(len(b))
    for i in range(len(b)):
        live = np.diag(G[i]) > 0
        if not live.any():
            continue
        g, bb = G[i][np.ix_(live, live)], b[i][live]
        L = np.linalg.cholesky(g + ridge * np.trace(g) / len(bb) * np.eye(len(bb)))
        c = nnls(L.T, np.linalg.solve(L, bb))[0]
        out[i] = (2 * bb @ c - c @ g @ c) / zz[i]
    return out


def continuation_fve(C, Z, device="cuda"):
    """C: [N, n, d], Z: [N, d] (whitened) -> FVE(z; C) per position, [N]."""
    return np.concatenate([fve_batch(Z[s : s + 256].to(device), C[s : s + 256].to(device)) for s in range(0, len(Z), 256)])


class Scorer:
    """Scores lens reads against positions built by train.py build (`data`: Z, C, fve_c, layer, cont_ids, cont_text)."""

    def __init__(self, data, ar, whiteners, tok, beta=1.0, n_distractors=8, empty_reward=-0.1):
        self.d, self.ar, self.W, self.tok = data, ar, whiteners, tok
        self.beta, self.n_dis, self.empty = beta, n_distractors, empty_reward

    @torch.no_grad()
    def bullet_atoms(self, texts, layers):
        """-> B [R, K_MAX, d] whitened bullet directions (zero rows past nb[r]), nb [R], the bullets, their token ids."""
        bls = [bullets(t)[:K_MAX] for t in texts]
        spans = [self.tok(" " + b, add_special_tokens=False).input_ids[:32] for bl in bls for b in bl]  # the space lens
        lays = [L for bl, L in zip(bls, layers) for _ in bl]
        dev = self.ar.model.device
        B = torch.zeros(len(texts), K_MAX, self.d["Z"].shape[1], device=dev)
        if spans:
            D = torch.cat([self.ar(spans[i : i + 256], lays[i : i + 256]) for i in range(0, len(spans), 256)])
            for L in set(lays):
                js = [j for j, x in enumerate(lays) if x == L]
                D[js] = self.W[L](D[js])
            rows = [(r, k) for r, bl in enumerate(bls) for k in range(len(bl))]
            B[[r for r, _ in rows], [k for _, k in rows]] = D
        it = iter(spans)
        return B, [len(bl) for bl in bls], bls, [[next(it) for _ in bl] for bl in bls]

    def distractors(self, own, pool, rng):
        """n_distractors positions from `pool` (indices) at own's layer, not own."""
        same = pool[(self.d["layer"][pool] == self.d["layer"][own]) & (pool != own)]
        if len(same) == 0:  # tiny pools only
            same = pool[pool != own]
        return rng.choice(same, self.n_dis, replace=len(same) < self.n_dis)

    @torch.no_grad()
    def __call__(self, texts, owners, pool, rng, chunk=32):
        """texts [R] lens reads; owners [R] the position each read is of; pool: positions to draw distractors from.
        -> per-read dicts: reward, marginal, plain FVE(z; B), joint FVE(z; C ∪ B), bullets, overlap with the continuation."""
        owners = np.asarray(owners)
        B, nb, bls, bl_ids = self.bullet_atoms(texts, self.d["layer"][owners].tolist())
        T = np.stack([np.concatenate([[o], self.distractors(o, pool, rng)]) for o in owners])  # [R, 1 + n_dis]
        fve_c, out = self.d["fve_c"], []
        for s in range(0, len(texts), chunk):  # [chunk, T, free, d] of free atoms at a time
            t = T[s : s + chunk]
            idx = torch.as_tensor(t.reshape(-1))
            C = self.d["C"][idx].to(B.device).float().view(*t.shape, *self.d["C"].shape[1:])
            Z = self.d["Z"][idx].to(B.device).float().view(*t.shape, -1)
            b_ = B[s : s + chunk]
            A = torch.cat([C, b_[:, None].expand(-1, t.shape[1], -1, -1)], 2)  # [c, T, free + K, d]
            joint = fve_batch(Z.flatten(0, 1), A.flatten(0, 1)).reshape(t.shape)  # zero bullet rows are dropped
            plain = fve_batch(Z[:, 0], b_)
            for i in range(len(t)):
                r, n = s + i, nb[s + i]
                if n == 0:
                    empty = {"reward": self.empty, "marginal": 0.0, "fve_plain": 0.0, "bullets": []}
                    out.append({**empty, "fve_joint": float(fve_c[owners[r]])})
                    continue
                M = joint[i] - fve_c[t[i]]
                cont = set(self.d["cont_ids"][owners[r]][:32])
                ov = [len(set(x) & cont) / max(len(set(x)), 1) for x in bl_ids[r]]
                text = self.d["cont_text"][owners[r]].lower()
                out.append({
                    "reward": float(M[0] - self.beta * M[1:].mean()), "marginal": float(M[0]),
                    "fve_plain": float(plain[i]), "fve_joint": float(joint[i, 0]),
                    "bullets": bls[r], "overlap32": float(np.mean(ov)),
                    "verbatim128": float(np.mean([x.strip().lower() in text for x in bls[r]])),
                })  # fmt: skip
        return out
