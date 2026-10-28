# graphs

Every chart in the repo, one module each, run from the repo root as
`uv run python -m graphs.<module>`; each writes PNGs under `docs/img/` (gitignored: the doc that
shows a chart names the module that regenerates it). Scoring and run selection are not here: charts
import them from `cfbench.report.data`. Colors follow CLAUDE.md (transcript-only blue, O-lens
orange, placebo black, no tools gray).

- `cfbench_report`: the progress report's cfbench charts (`docs/cfbench-scoring.md`,
  `docs/cfbench-model-chart.md`); calls `cfbench_modelchart` for the predictor-model panels
- `hint_mcq_answer`: the hint_mcq answer task, three bars per hint type and per-item grids
  (`docs/envs/hint_cf.md`)
- `honest_binary`, `honest_binary_olens`: the honest-grader binary prediction runs
  (`docs/honest-grader-envs.md`)

`olens_prompt_ea/` keeps its own figures (`fig_settings.py`, `anyhit.py`, `punct_analysis.py`): they
are tied to that package's analysis code and `sys.path` layout.
