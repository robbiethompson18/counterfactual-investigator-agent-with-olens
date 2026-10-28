"""vLLM plugin: capture DeepSeek V4's folded 4-stream state at a decoder layer's output through vllm-metamodels' hooks.

vllm_lens refuses `output_residual_stream=[L]` on hyper-connection models because a V4 decoder layer returns the deferred
fold (x, residual[T, 4, D], post_mix, res_mix) that the next layer's mhc_pre consumes, not one [T, hidden] tensor. The state
that enters layer L + 1 (and HF's layer-L output, read.py's Capture hook) is mhc_post(x, residual, post_mix, res_mix):
out_j = post_mix_j * x + sum_i res_mix_ij * residual_i, [T, 4, D]; the model itself calls it after the last layer and for
aux hidden states. This plugin wraps vllm_lens._worker_ext._split_layer_output so that a V4 4-tuple becomes that folded
state, flattened to [T, 4 * D] (the row layout of read.py's states.pt and the hc_inject vectors), with no residual half.
vllm_lens's fast capture path then works unchanged: index_select at the request's capture_positions, one pinned async copy,
`out.activations["residual_stream"]` of [n_layers, n_pos, 4 * D] via get_captured_states. The wrapper only runs on layers
with capture work (vllm_lens calls _split_layer_output only then) and never changes the layer's real output.

It also sets VLLM_LENS_MULTI_STREAM=0, which is what makes vllm_lens accept the request (client-side _check_layer_support
and the worker's lens_capabilities); the flag otherwise only gates layer-output steering, which nothing here sends
(hc_inject's injection goes through EMBED_LAYER_INDEX and is unaffected).
"""

import os


def register():
    if os.environ.get("HC_CAPTURE", "1") != "1":
        return
    os.environ.setdefault("VLLM_LENS_MULTI_STREAM", "0")
    try:
        import torch
        import vllm_lens._worker_ext as W
        from vllm.models.deepseek_v4.nvidia import model as M
    except Exception as e:  # noqa: BLE001  (plugin must never break an unrelated engine)
        print(f"[hc_capture] not installed: {e!r}")
        return
    if getattr(W, "_hc_capture", False):
        return
    W._hc_capture = True
    orig_split = W._split_layer_output

    def _split_layer_output(output, layer_idx):
        if isinstance(output, tuple) and len(output) == 4 and all(isinstance(o, torch.Tensor) for o in output) and output[1].dim() == 3:
            x, residual, post_mix, res_mix = output
            h = M.mhc_post_tilelang(x, residual, post_mix, res_mix)  # [T, hc_mult, D]: the state entering layer_idx + 1
            return h.flatten(1), None, ()
        return orig_split(output, layer_idx)

    W._split_layer_output = _split_layer_output
    print(f"[hc_capture] installed (pid {os.getpid()})")
