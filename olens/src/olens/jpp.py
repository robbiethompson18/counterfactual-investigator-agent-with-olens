"""The J++ lens (Ayonrinde & Lindsey 2026; neuronpedia/jacobian-lens qwen3.6-27b/jpp): next-token logits read from the
residual after block l as lm_head(norm(J[l] h_l)), l in 0..62, with J[l] a fitted d x d map (fp16; 3.3 GB for all layers).
It needs residuals only (olens.one_engine.OneEngine.capture_many) plus the base's final norm and lm_head, so it owns no
model and no LoRA: a read is one matmul per layer on top of the capture. The HF version (olens.extras.JPP, deleted
2026-10-27) agreed with the full forward pass at top-1 on 100% of 13 positions x 16 layers (runs/jpp_reads/2026-10-22_*).
"""

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

JPP_REPO = "neuronpedia/jacobian-lens"
JPP_FILE = "qwen3.6-27b/jpp/Salesforce-wikitext/Qwen3.6-27B_jpp_lens.pt"


class JPP:
    def __init__(self, base_dir, norm_w, eps, layers=None, device="cuda"):
        """base_dir: the Qwen3.6-27B snapshot (for lm_head); norm_w, eps: its final RMSNorm's gain (1 + w) and eps, as OneEngine holds them;
        layers: source layers to serve (default all 63). Each layer's map is 52 MB on the GPU; lm_head is 2.5 GB."""
        path = hf_hub_download(JPP_REPO, JPP_FILE)
        self.revision = Path(path).resolve().parent.name  # the HF cache keeps the file under snapshots/<commit>/
        maps = torch.load(path, map_location="cpu", weights_only=True)["J"]
        self.layers = list(layers) if layers else sorted(int(L) for L in maps)
        self.J = {L: maps[L].to(device) for L in self.layers}  # [d_out, d_in] fp16
        wmap = json.loads((Path(base_dir) / "model.safetensors.index.json").read_text())["weight_map"]
        with safe_open(Path(base_dir) / wmap["lm_head.weight"], "pt") as f:
            self.W = f.get_tensor("lm_head.weight").to(device)  # [V, d] bf16; Qwen3.6-27B's head is not tied to the embedding
        self.norm_w, self.eps, self.device = norm_w.to(device), eps, device

    @torch.no_grad()
    def top(self, acts, k=5, chunk=512):
        """{L: [P, d] residuals after block L} for any of self.layers -> {L: ([P, k] token ids, [P, k] probabilities)}, both on
        the CPU. The map runs in fp16, the norm in fp32, the head in bf16 (as HF runs the base), the softmax in fp32."""
        out = {}
        for L, h in acts.items():
            ids, ps = [], []
            for a in range(0, len(h), chunk):
                x = (h[a : a + chunk].to(self.device, torch.float16) @ self.J[L].T).float()
                x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.norm_w).to(torch.bfloat16)
                top = (x @ self.W.T).float().softmax(-1).topk(k, dim=-1)
                ids.append(top.indices.cpu())
                ps.append(top.values.cpu())
            out[L] = (torch.cat(ids), torch.cat(ps))
        return out
