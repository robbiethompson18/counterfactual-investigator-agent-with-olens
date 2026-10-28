"""vLLM plugin for Camila Blank's DeepSeek V4 Flash O-lens inverter, which injects at the INPUT of decoder layer 1: the
full mHC state [hc_mult=4, 4096] after layer 0, overwritten at the carrier's slot with alpha * h / ||h||.

vllm-metamodels (vllm_lens) routes per-request vectors to prompt rows but only writes the embedding stream on V4, because
in vLLM a V4 layer boundary is a deferred fold (x, residual[T, 4, D], post_mix, res_mix), not one tensor. This plugin:
  1. wraps vllm_lens._worker_ext._apply_embed: an EMBED_LAYER_INDEX vector whose width is 4x the embedding's is not written
     to the embedding; its (row, vector) pair is stashed instead;
  2. wraps DeepseekV4DecoderLayer.forward for layer 1: when rows are stashed, fold layer 0's output with mhc_post (exactly
     the post half of mhc_fused_post_pre), overwrite those rows of the [T, 4, D] state, and run layer 1 from that full state
     (its residual=None, x.dim()==3 branch: mhc_pre on a multi-stream input).
Prefill only (the carrier slot is a prompt position), so decode steps are untouched.
"""

import os

PENDING: list = []  # (flat row, vector [4 * D]) for this forward pass


def register():
    if os.environ.get("HC_INJECT", "1") != "1":
        return
    try:
        import vllm_lens._worker_ext as W
        from vllm.models.deepseek_v4.nvidia import model as M
    except Exception as e:  # noqa: BLE001  (plugin must never break an unrelated engine)
        print(f"[hc_inject] not installed: {e!r}")
        return
    if getattr(W, "_hc_inject", False):
        return
    W._hc_inject = True
    orig_apply = W._apply_embed

    def _apply_embed(todo, target, plan, stats):
        keep = []
        qsl, abs_start = plan.qsl, plan.abs_start
        width = target.shape[-1]
        for i, configs in todo:
            start, end, a0 = qsl[i], qsl[i + 1], abs_start[i]
            rest = []
            for cfg in configs:
                lim = cfg.layer_index_map
                if W.EMBED_LAYER_INDEX in lim and cfg.activations.shape[-1] == 4 * width:
                    vec = cfg.activations[lim[W.EMBED_LAYER_INDEX]]
                    for pi, abs_pos in enumerate(cfg.position_indices):
                        if a0 <= abs_pos < a0 + (end - start):
                            PENDING.append((abs_pos - a0 + start, vec[pi] * cfg.scale))
                            stats["hc_rows"] = stats.get("hc_rows", 0) + 1
                else:
                    rest.append(cfg)
            if rest:
                keep.append((i, rest))
        if keep:
            orig_apply(keep, target, plan, stats)

    W._apply_embed = _apply_embed

    L = M.DeepseekV4DecoderLayer
    orig_init, orig_fwd = L.__init__, L.forward

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        prefix = kwargs.get("prefix", args[1] if len(args) > 1 else "")
        self._hc_idx = int(str(prefix).rsplit(".", 1)[-1]) if str(prefix).rsplit(".", 1)[-1].isdigit() else -1

    def forward(self, x, positions, input_ids, post_mix=None, res_mix=None, residual=None, **kw):
        if getattr(self, "_hc_idx", -1) == 1 and PENDING:
            assert residual is not None and not self.use_sequence_parallel
            h = M.mhc_post_tilelang(x, residual, post_mix, res_mix)  # [T, 4, D]: the true state entering layer 1
            import torch

            rows = torch.tensor([r for r, _ in PENDING], device=h.device, dtype=torch.long)
            vecs = torch.stack([v for _, v in PENDING]).to(h.device, h.dtype).view(len(PENDING), *h.shape[1:])
            PENDING.clear()
            h = h.clone()
            h.index_copy_(0, rows, vecs)
            return orig_fwd(self, h, positions, input_ids, None, None, None, **kw)
        return orig_fwd(self, x, positions, input_ids, post_mix, res_mix, residual, **kw)

    L.__init__, L.forward = __init__, forward
    print(f"[hc_inject] installed (pid {os.getpid()})")
