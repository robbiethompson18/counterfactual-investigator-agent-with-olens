# graphs

Every chart in the repo, one module each, run from the repo root as
`uv run python -m graphs.<module>`; each writes PNGs under `docs/img/` (gitignored: the doc that
shows a chart names the module that regenerates it). Scoring and run selection are not here: charts
import them from `cfbench.scoring`. Every module imports `style.py`, and importing it applies the
look (no `use()`, no `rcParams` here): colors, placebo stripes, bars with CI whiskers and the
paired-difference rows (each labelled by its two conditions' swatches, A − B) are `bar_kw`,
`marker_kw`, `ci_bar`, `positions` and `diff_rows`; CLAUDE.md has the rule. The font is Inter
(macOS: `brew install --cask font-inter`), with a silent DejaVu fallback if it's missing.

- `cfbench_report`: the progress report's cfbench charts (`docs/cfbench-scoring.md`,
  `docs/cfbench-model-chart.md`); calls `cfbench_modelchart` for the predictor-model panels
- `cutoff_qa_inline`: Haiku predicting Qwen's answer from the question, transcript-only vs O-lens vs
  J++ and their placebos, for all / stale / not-stale items, plus paired differences
  (`docs/cutoff-qa-inline.md`). The one cutoff chart: add NLA as soon as its runs exist
- `cutoff_qa_tool_inline`: the same three conditions with Qwen answering in plain text vs through a
  required submit_answer tool call (answer 5 vs 20 tokens past the last read), one panel each
  (`docs/cutoff-qa-inline.md`)
- `cutoff_qa_notail`: the v3 tool run's O-lens cut to the question's punctuation (± `<|im_end|>`) vs
  the full set, each with its placebo, error rate, stale / not stale / all, plus paired differences
  on stale items (`docs/cutoff-qa-inline.md`)
- `favorites_qa_inline`: the same seven bars and paired-difference figure for the 1,500-item
  favourites eval, all items and by question shape (`docs/favorites-qa-inline.md`)
- `supergpqa_inline`: the cutoff chart for supergpqa_inline, groups all / Qwen wrong / Qwen right
  cases (`docs/supergpqa-inline.md`)
- `standard_probes`: all five standard families, both positions, full-population MCQ transfer and
  hinted MCQ, plus matched essay investigators (`docs/standard-probe-protocol.md`)
- `hint_mcq_answer` (legacy selected cohort): the hint_mcq answer task, three bars per hint type and
  per-item grids (`docs/envs/hint_cf.md`)
- `hint_mcq_total` (legacy selected cohort): the answer task's running total over the original items
  and the 2k scale-up (every Flash run, 8/8 hint followers), pooled and by hint type
- `mcq_unselected`: Haiku predicting Qwen's MMLU-Pro letter from the bare prompt on the unselected
  set, prompt only vs O-lens, J-lens and NLA with their placebos, beside both standardized probe
  read positions on the same questions, for all / Qwen-right / Qwen-wrong cases, plus paired
  differences (`docs/mcq-unselected.md`)
- `menus_inline`: which read positions help (cfbench/evals/menus_inline): O-lens, NLA and J++ each
  at four position menus (prompt punctuation, 9 tail tokens, all, last token) beside their placebos
  and prompt only, pooled over cutoff_qa, favorites_qa, MMLU-Pro and SuperGPQA with each set
  weighted equally
- `honest_binary`, `honest_binary_olens`: the honest-grader binary prediction runs
  (`docs/honest-grader-envs.md`)
- `olens_mandarin`: Mandarin in O-lens bullets and output tokens, before vs after the leading-space
  fix (`docs/olens-chinese.md`, `docs/final-writeup.md`); needs `--with transformers`
- `wsb_families`: WorkspaceBench families on Qwen3.6-27B, O-lens vs Sonnet-as-text-reader vs stock
  Qwen asked directly, numbers hand-copied from the sprint doc (Wrong Bridge is a TBD placeholder)
- `wsb_text_inversion`: the four contested WorkspaceBench families, best published metamodel vs
  Sonnet 5.5 text inversion, at WSB's any-of-grid scoring and at pass@1
  (`docs/wsb-flash-prompt-only.md`); numbers are sourced constants in the module
- `data_filter_sankey`: how SuperGPQA, MMLU-Pro, cutoff_qa, favorites_qa and essay questions become eval cases (one PNG each; dark
  green evaluated, light green held out, gray unusable), counts sourced in the module; for the final write-up's Data Filter
  section
- `cutoff_grader_mentions`: how often the cutoff_qa O-lens reads say "grader"/"graded" (novel hits),
  no tool vs submit_answer tool, by read position, by layer and as items with ≥1 hit (data from
  `scripts/cutoff_grader_mentions.py`, run `cutoff_grader_mentions/2026-10-30_regex`)
