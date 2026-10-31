# Standard probes

Follow [the protocol](../docs/standard-probe-protocol.md). Data/results belong under a tracked
`data/runs/standard_probes/` run, never in git. Run from the repository root:

```bash
uv run --with transformers --with jinja2 python -m probes.prepare
# On each watched GPU, with the model revision from RUN/provenance.json:
python probes/capture.py RUN/shards/N/hs_inputs.jsonl OUTPUT --revision REVISION
# Fetch each OUTPUT into RUN/capture/N/; then fit each of the five families:
OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 uv run python -m probes.fit RUN \
  --family mmlu --activations RUN/capture
# --family is essay, hinted, mmlu, supergpqa, combined; both sites run by default.
uv run python -m probes.report RUN
uv run python scripts/runs.py finish RUN
```

`prepare.py` reuses saved responses, pins capture inputs, groups shared question stems, and imports
historical essay activations with their missing-revision caveat. Capture validates its layer hooks
against HF hidden states and writes complete chunks marked by `logp.jsonl`. Fetch only complete
chunks before fitting. Capture dependencies are torch, transformers, accelerate, numpy and
flash-linear-attention; RunPod usage/guards are in `runpod/CLAUDE.md`.

`fit.py` uses fixed mean-loss regularization, train-only standardization, saved outer folds and a
common pre-cap MCQ inner partition. Its `--smoke` mode checks numerical gradients, option masking,
group isolation and ridge equivalence. `report.py` validates held-out coverage and shared partitions
before summarizing. Keep source snapshots, manifests, preprocessing/weights and predictions on HF.
