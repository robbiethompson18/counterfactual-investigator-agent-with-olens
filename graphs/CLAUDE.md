# graphs

Every chart in the repo, one module each, run from the repo root as
`uv run python -m graphs.<module>`; each writes PNGs under `docs/img/` (gitignored: the doc that
shows a chart names the module that regenerates it). Scoring and run selection are not here: charts
import them from `cfbench.scoring`. Colors follow CLAUDE.md (transcript-only blue, O-lens orange,
placebo black, no tools gray).

- `cfbench_report`: the progress report's cfbench charts (`docs/cfbench-scoring.md`,
  `docs/cfbench-model-chart.md`); calls `cfbench_modelchart` for the predictor-model panels
- `cutoff_qa_inline`: Haiku predicting Qwen's answer from the question, transcript-only vs O-lens vs
  J++ and their placebos, for all / stale / not-stale items, plus paired differences
  (`docs/cutoff-qa-inline.md`). The one cutoff chart: add NLA as soon as its runs exist
- `favorites_qa_inline`: the same seven bars and paired-difference figure for the 1,500-item
  favourites eval, all items and by question shape (`docs/favorites-qa-inline.md`)
- `hint_mcq_answer`: the hint_mcq answer task, three bars per hint type and per-item grids
  (`docs/envs/hint_cf.md`)
- `hint_mcq_total`: the answer task's running total over the original items and the 2k scale-up
  (every Flash run, 8/8 hint followers), pooled and by hint type
- `honest_binary`, `honest_binary_olens`: the honest-grader binary prediction runs
  (`docs/honest-grader-envs.md`)
- `wsb_families`: WorkspaceBench families on Qwen3.6-27B, O-lens vs Sonnet-as-text-reader vs stock
  Qwen asked directly, numbers hand-copied from the sprint doc (Wrong Bridge is a TBD placeholder)
- `wsb_text_inversion`: the four contested WorkspaceBench families, best published metamodel vs
  Sonnet 5.5 text inversion, at WSB's any-of-grid scoring and at pass@1
  (`docs/wsb-flash-prompt-only.md`); numbers are sourced constants in the module
