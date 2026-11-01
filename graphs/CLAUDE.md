# graphs

Every chart in the repo, for the docs and for the slide deck, one module each, run from the repo
root as `uv run python -m graphs.<module>`; each writes PNGs under `docs/img/` (gitignored: the doc
that shows a chart names the module that regenerates it), except `talk`, which writes the deck's
PDFs to `slides/talk/fig/`. `docs/` and `slides/` are leaves: nothing here imports from them, and no
Python lives there. Scoring and run selection are not here: charts import them from
`cfbench.scoring`. Every module imports `style.py`, and importing it applies the look (no `use()`,
no `rcParams` anywhere else): colors, placebo stripes, bars with CI whiskers and the
paired-difference rows (each labelled by its two conditions' swatches, A − B) are `bar_kw`,
`marker_kw`, `ci_bar`, `positions` and `diff_rows`; CLAUDE.md has the rule. A slide figure is a
function decorated with `@slide`, which applies the deck's sizes and transparent background
(`SLIDE_RC`) around that one figure, with `slide_figure(w_mm, h_mm)` and `slide_save`; never apply
them process-wide (that is what turned the headline PNGs white and small on 2026-10-31). The font is
Inter (macOS: `brew install --cask font-inter`), with a silent DejaVu fallback if it's missing.

- `cfbench_report`: the progress report's cfbench charts (`docs/cfbench-scoring.md`,
  `docs/cfbench-model-chart.md`); calls `cfbench_modelchart` for the predictor-model panels
- `cutoff_qa_inline`: Haiku predicting Qwen's answer from the question, transcript-only vs O-lens vs
  J++ and their placebos, for all / stale / not-stale items, plus paired differences
  (`docs/cutoff-qa-inline.md`). The one cutoff chart: add NLA as soon as its runs exist
- `cutoff_qa_investigators`: knowledge cutoff by investigator, Haiku 5.5 (later cutoff) beside
  Gemini 2.5 Flash-Lite (Qwen's cutoff), each with its own-answer census, stale / not stale / all.
  `--set canonical` (default, the final write-up: menus_inline `all` runs) or
  `--set cutoff_qa_inline` (the older eval, every Flash-Lite condition and the told prompt,
  `docs/cutoff-qa-inline.md`)
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
  weighted equally; plus `menus-inline-position.png`, each lens's question-reads minus tail-reads
  with a paired CI
- `read_positions_diagram`: one real MMLU-Pro prompt as Qwen sees it, every read position coloured
  by menu (question vs tail, last token boxed), tokens copied from the menus_inline inputs; for the
  final write-up's "By token position"
- `honest_binary`, `honest_binary_olens`: the honest-grader binary prediction runs
  (`docs/honest-grader-envs.md`)
- `olens_mandarin`: Mandarin in O-lens bullets and output tokens, before vs after the leading-space
  fix (`docs/olens-chinese.md`, `docs/final-writeup.md`); needs `--with transformers`
- `wsb_families`: WorkspaceBench families on Qwen3.6-27B, O-lens vs Sonnet-as-text-reader vs stock
  Qwen asked directly, numbers hand-copied from the sprint doc (Wrong Bridge is a TBD placeholder)
- `wsb_text_inversion`: the four contested WorkspaceBench families, best published metamodel vs
  Sonnet 5.5 text inversion, at WSB's any-of-grid scoring and at pass@1
  (`docs/wsb-flash-prompt-only.md`); numbers are sourced constants in the module
- `headline_inline`: the final write-up's Results headline, from the canonical menus_inline runs
  (`all` menu) only, % of cases right on MCQ, knowledge cutoff and preferences: pooled metamodels,
  each against its placebo, against the MCQ probe (and as error rate), and seven bars per task;
  prints the paired differences the write-up quotes
- `essay_grading`: essay grading for the write-up, the deck's essay chart at write-up size: AUROC of
  P(score goes up), transcript only, each lens beside its placebo, the same with Qwen's observed mean
  grade added, the user-end probe, the bet-at-the-mean null; `essay_bars(ax, s)` is the one draw
  function, which `talk` renders under `@slide`; plus the paired differences in AUROC
- `headline_ablations`: the write-up's ablation charts on `headline_inline`'s runs: % right split by
  Qwen right / wrong (MMLU-Pro, SuperGPQA, knowledge cutoff), each lens and each placebo minus
  transcript only per task and pooled, and essay grading's and the blog deletions' AUROC split by
  whether Qwen is right (essays: wrong side of its median grade vs the PERSUADE score)
- `headline_tokens`: tokens each lens's reads add to Haiku's prompt over transcript only, pooled as
  `headline_inline` and per set, for the final write-up's "Total Tokens Per"
- `data_filter_sankey`: how SuperGPQA, MMLU-Pro, cutoff_qa, favorites_qa, essay and blog questions
  become eval cases (one PNG each; dark green evaluated, light green held out, gray unusable),
  counts sourced in the module; for the final write-up's Data Filter section
- `chive_inline`: AUROC on CHIVE's official yes/no claims, Haiku with vs without the model's reply,
  beside CHIVE's Opus 4.8 transcript and resampling arms, one panel per target, plus the paired
  difference (`docs/chive-inline.md`)
- `author_blog_drop`: blog sentence-deletion AUROC, the deck's bars at write-up size (transcript
  only, O-lens and NLA with placebos, the L2 probe, probe + transcript), computed by
  `cfbench.scoring.author_bars` with blogger-bootstrap CIs (`docs/author-mcq-probe.md`)
- `talk`: the sprint talk's figures (`slides/talk/talk.tex`, `make -C slides` runs it first): the
  hint evals' grid, essay grading's bars (`essay_grading.essay_bars`) and paired differences, the
  grader-mention bars, the harness_grid grade[rd] prompt-read vs acting bars (`cfbench.scoring.grader_acting`), the Mandarin before/after bars and the blog-deletion AUROCs, plus the LaTeX
  table snippets the deck `\input`s, all to `slides/talk/fig/`; statistics in `cfbench.scoring`
  (`essay_auroc_bars`, `author_bars`, ...)
- `cutoff_grader_mentions`: how often the cutoff_qa O-lens reads say "grader"/"graded" (novel hits),
  no tool vs submit_answer tool, by read position, by layer and as items with ≥1 hit (data from
  `scripts/cutoff_grader_mentions.py`, run `cutoff_grader_mentions/2026-10-30_regex`)
