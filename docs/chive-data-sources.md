# CHIVE data sources (as of 2026-10-19)

Where the CHIVE incidents, ground truth, readouts and Agam's agent results live, and what each
source lacks.

## HF `agu18dec/chive-forensics` (public)

300 Qwen3.6-27B incidents: conversation, completion, behaviour, k/n, tokenized transcript
(`token_ids`), and full ground truth: causes, ruled-out alternatives, `movers`, and all 1,944
counterfactual experiments as replayable edits with measured rates. No readouts.

## Agam's splits (`detective-joracle/data/chive/splits/`)

- `climb_dev96` (dev, 96) and `climb_test204` (held-out test, 204): the split for his harness climb.
  He calls them "dev" and "held-out test"; we use the same dev set.
- `dev100` / `heldout200` are an older split (only 29 incidents overlap `climb_dev96`); `hard92`,
  `pilot15`, `deep_read19` are subsets, not splits.

## Agam's case viewer (Modal; URL is in Agam's research-log thread, not here)

A static site whose JSON files can be fetched directly. Agam said scraping is fine.

- Per-incident scores for every arm x draw: `data/v2/<ds>/cmp.json`, `<ds>` in `climb_test`,
  `chive_fresh`, `chive_fresh_exp1`, `swechat_pilot`.
- Incident detail: `data/v2/<ds>/inc/<pid>.json`.
- Readout banks, one file per incident and layer, a list per token of k samples: held-out and dev
  incidents at `data/r/<pid>/olens_bank_L{L}.json` (also `mlnla_L{L}`, `jlens_all_L{L}`,
  `nla_k4_L{L}`); fresh incidents at `data/v2/chive_fresh/r/<pid>/...`. O-lens is `s3d_rl600`, 4
  samples, L20-60 every 4th; about 1-2 KB per token per layer.
- `chive_fresh`: 915 more incidents from the same pipeline. The viewer has causes and ruled-out
  items with effect sizes in prose, and tokens as `[piece, role, turn]` (no ids). It does not have
  the structured experiment edits or `movers`; we asked Agam for those.

`scripts/fetch_olens_banks.py` (not on `main` yet: it needs the vendored harness) converts viewer
banks to the harness's `<bank>/<pid>.json.gz` format, taking token ids from the HF dataset.

## Per-incident difficulty (held-out 204, Agam's Opus 5 runs, 5 draws)

Transcript-only: 17% of incidents never solved, 56% always solved. O-lens: 10% never, 56% always.
Most incidents sit at ceiling or floor; the informative band is the ~30% solved sometimes.
