# olens

Loads and reads the public oracle-lens checkpoints for `Qwen/Qwen3.6-27B` without the unreleased
`global_workspace` package: hook-based LoRA (no PEFT), activation capture, the
[agu18dec/olens_and_ar](https://huggingface.co/agu18dec/olens_and_ar) lens (AO) and its activation
reconstructor (AR), whitened FVE, and the ceselder layer-42 GRPO lens.

```
src/olens/lora.py       hook-based LoRA (PEFT keys, rsLoRA, `_orig_mod.` keys)
src/olens/lens.py       load_base(), capture(), AguLens (layers 20..60 step 4), CeselderLens (layer 42)
src/olens/ar.py         the AR: `[Layer NN] <span>` -> predicted raw activation before the span
src/olens/fve.py        whitener (published pooled moments) + NNLS whitened FVE
src/olens/readout.py    bullets -> AR directions -> NNLS weights + joint FVE
src/olens/vllm_lens.py  the AO decoded by vLLM from a merged checkpoint (scripts/merge_lens.py)
src/olens/vllm_nla.py   the ceselder L42 NLA's AV decoded by vLLM (both LoRAs merged in)
src/olens/extras.py     the ceselder L42 NLA and the Ayonrinde & Lindsey J++ lens on the shared base
src/olens/vectors.py    probe / steering directions in the residual stream built from the lenses, steering hook
scripts/merge_lens.py   merge a hook-LoRA (or the AV's two LoRAs) into a checkpoint for vLLM
scripts/download.sh     checkpoints into checkpoints/ (gitignored); EVAL=1 also fetches the whiteners
```

Conventions, verified against the published held-out joint FVE (0.486 here vs 0.487 reported): the
AO's marker input embedding is overwritten with `16000 * h/|h|`; the AR head predicts the **raw**
activation at the token before the span; target and predictions are whitened as `Σ^-1/2 (x − μ)`
with the pooled whitener; a bullet goes to the AR as the text after `- `. `AguLens` defaults to the
space lens `olens_s3d_space_rl600` (CLAUDE.md), whose bullets go to the AR with a leading space
(`lens.bullet_prefix = " "`); the old `olens_s3d_rl600`, which the 0.486 above was measured on,
takes them without one.

The whiteners come from the gated dataset
[agu18dec/local-workspace](https://huggingface.co/datasets/agu18dec/local-workspace) (needs an
`HF_TOKEN` with access). Needs torch + `requirements.txt`; on CUDA also `kernels` and
`flash-linear-attention`. Run from this directory with `PYTHONPATH=src`.

```python
from olens.lens import load_base, capture, AguLens
from olens.ar import AR
from olens.readout import Readout

tok, model = load_base()
agu = AguLens(tok, model)
ids = tok("The athlete Muhammad Ali plays the sport of", add_special_tokens=False).input_ids
h = capture(model, ids, [44])[44][-1]
Readout(agu, AR(model))(h, 44)  # [{text, bullets, coefs, fve}]
```
