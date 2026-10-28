"""Activation-level parity of two states.pt captures (read.py's HF capture vs vllm_capture_read.py --save-states) on the
rows they share: per-row cosine and relative norm of the [4, 4096] state at each common layer. Cosine near 1 means the
vLLM capture is the same activation as HF's up to kernel drift; near 0 means a wrong position, layer or fold.

    python compare_states.py A/states.pt B/states.pt
"""

import sys

import torch

A, B = (torch.load(p, map_location="cpu") for p in sys.argv[1:3])
key = lambda r: (r["id"], r["pos"])
ib = {key(r): i for i, r in enumerate(B["rows"])}
pairs = [(i, ib[key(r)]) for i, r in enumerate(A["rows"]) if key(r) in ib]
print(f"rows: {len(A['rows'])} in A, {len(B['rows'])} in B, {len(pairs)} shared; layers A {A['layers']} B {B['layers']}")
ia, jb = (torch.tensor(x) for x in zip(*pairs))
q = lambda t: "/".join(f"{v:.3f}" for v in t.quantile(torch.tensor([0.05, 0.5, 0.95])).tolist())
for L in sorted(set(A["layers"]) & set(B["layers"])):
    a = A["states"][ia, A["layers"].index(L)].float().flatten(1)
    b = B["states"][jb, B["layers"].index(L)].float().flatten(1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
    rel = (a - b).norm(dim=1) / b.norm(dim=1)
    ratio = (a.norm(dim=1) / b.norm(dim=1)).mean()
    print(
        f"L{L}: cosine mean {cos.mean():.4f} (p5/50/95 {q(cos)}), |a-b|/|b| mean {rel.mean():.3f} (p5/50/95 {q(rel)}), |a|/|b| {ratio:.3f}"
    )
    for w in cos.argsort()[:3].tolist():
        r = A["rows"][pairs[w][0]]
        print(f"  low cosine {cos[w]:.3f} at {r['id']} pos {r['pos']} role {r['role']} token {r['token']!r}")
