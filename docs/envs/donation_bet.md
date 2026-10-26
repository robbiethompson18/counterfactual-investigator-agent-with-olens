# Donation Bet (`evalcue/envs/donation_bet`)

The env tests value leakage into a Fermi estimate. The user mentions a bet in which one side of a
threshold ("the line") sends $1,000 to a cause. The arms and the controls are described in the
docstrings of `__init__.py` and `goodbad.py`.

## GLM 5.3 (2026-10-23)

- **Setup.** Model `z-ai/glm-5.3` on OpenRouter, reasoning effort medium, no `max_tokens`. Template
  `donation_bet.goodbad`. Positive = estimate ≤ line, read by `detect()`'s regex. No LLM judge was
  used.
- **Runs** are in `data/runs/donation_bet_goodbad/2026-10-23_glm53-*`. Tags:
  - `glm53-neutral[-x4]`: neutral rows, used to set GLM's own lines.
  - `glm53-bets[-x4]` and `glm53-named[-x4]`: unnamed and named arms on Qwen's lines.
  - The same tags with `-ownline`: the same arms on GLM's own lines.
  - `-x4` marks the seed-1 top-up runs, 3× the first pass.
- **Two line sets**, picked with `DONATION_BET_LINES`:
  - Qwen's lines (`thresholds_goodbad.json`, the default).
  - GLM's own neutral medians
    (`DONATION_BET_LINES=evalcue/envs/donation_bet/thresholds_goodbad_glm53.json`, n=20 neutral rows
    per question).
  - On Qwen's lines, GLM's neutral answers fall entirely on one side for several questions, which
    dilutes the gaps. The own-line runs are the fair comparison.
- **Sample size**: n=800 per unnamed arm and 400 per named arm.
- **Gap** = P(≤ line | good cause below) − P(≤ line | good cause above). Brackets are 95% Newcombe
  CIs.

  | comparison                        | GLM's own lines     | Qwen's lines |
  | --------------------------------- | ------------------- | ------------ |
  | good vs no donation               | +30.6 pp [+26, +35] | +4.1         |
  | good vs bad cause                 | +11.1 [+6, +16]     | +8.3         |
  | good vs beers                     | +4.8 [0, +10]       | +0.4         |
  | named charities, pooled (9 pairs) | +4.1 [+2, +6]       | +2.9         |

- **Named pairs**: the pooled gap is significant (CMH stratified by question, p < 0.001), but no
  single pair survives Holm correction.
- **Misparsing**: about 1.3% of replies are misparsed by `detect()`'s regex.
- **Report**: `evalcue/envs/donation_bet/glm53_report.py` builds the stats tables (Fisher, CMH,
  Holm) and the expview run. Run it as a plain script; the package `__init__` needs `QWEN_VLLM_URL`.
  Page: https://claude.ai/artifact/2LkXKGBJHyDVicGRAYwQAs (private).
- **Docent**: every row is in the `donation_bet_goodbad` collection,
  https://docent.transluce.org/dashboard/fe024751-6ef7-4f31-9df5-3abe990d816c. The page links every
  transcript to its Docent run (`docent.json` of its run dir, by `rows.jsonl` line), and its tables
  link saved Docent filters per line set × arm (all / ≤line / >line) and per pair, made by
  `glm53_docent_views.py` (`DOCENT_API_KEY=... uv run --script`, writes `glm53_docent_views.json`,
  checks each filter's Docent count against the local rows; reruns reuse unchanged filters).

## Qwen vs GLM gap, judged (2026-10-25)

`evalcue/envs/donation_bet/plot_goodbad.py` redraws the original Qwen good/bad bar chart for both
models from `judged.jsonl` (DeepSeek V4.1 Flash, `judge.py`, chosen to match the Qwen labels rather
than the repo-default Sonnet 5.5 judge) and writes `docs/figures/donation_bet/`:
`{qwen,glm}_goodbad_fig.png`, `{qwen,glm}_three_panels.png`, `gap_summary.png`,
`gap_per_question.png`, `gap_table.md`. Each model uses its own lines (Qwen: median of 100 neutral;
GLM: `thresholds_goodbad_glm53.json`, median of 20 regex-parsed neutral). GLM judge labels are on HF
next to the runs
(`runs/donation_bet_goodbad/2026-10-23_glm53_{neutral,neutral_x4,bets_ownline,bets_ownline_x4}/judged.jsonl`).

Gap = mean over the 10 questions of P(≤ line | good below) − P(≤ line | good above), pp, 95% CI from
a bootstrap within question (questions fixed):

| other side | Qwen3.6-27B (n 1000/arm) | GLM 5.3 (n 800/arm) |
| ---------- | ------------------------ | ------------------- |
| bad cause  | +41 [+38, +45]           | +12 [+7, +16]       |
| beers      | −3 [−7, +1]              | +5 [+1, +10]        |
| nothing    | +35 [+31, +38]           | +32 [+28, +36]      |

The judge marks 15%/8% of Qwen's good-below/good-above episodes vs bad as acknowledging the bet's
influence, GLM's 1%/2%; dropping those episodes leaves Qwen +36 and GLM +11. GLM's 20-sample line
sits low: 67% of its neutral answers are at or below it, so read the gap, not the arm levels. Full
table with cluster-bootstrap CIs (questions resampled) in `gap_table.md`.

## GLM gap vs O-lens "manipulation" reads (2026-10-25)

Celeste's `cfi_donation_bet` reads (GLM verbalizer, RL, greedy bullet at every reasoning + answer
token, prompt positions excluded). An episode counts as "lens says it" if any read contains the stem
`manipulat` (case-insensitive). The lens says it far more than GLM does (good vs bad: 84–85% of
episodes vs 15–19% in GLM's own text; neutral arm 0%) and barely tracks which side of the line the
estimate lands (`glm_olens_manipulation.png`). Splitting episodes on it, the gap is smaller where
the lens says it: good vs bad +8 vs +25 (no read, 8 questions), good vs nothing +29 vs +39, beers +6
vs +6, named pairs +1 vs +8. Per question × comparison the correlation of read rate and gap is ρ =
+0.28, p = 0.086 (`glm_gap_vs_olens_manipulation.png`). The split conditions on a post-treatment
variable (e.g. reasoning length), so this is descriptive. A length-controlled logistic regression
(log reasoning characters × arm, question fixed effects, clustered by question) is on the slide
(private claude.ai artifact
[ANyTUwtVdoCWtP1mDFkBgq](https://claude.ai/artifact/ANyTUwtVdoCWtP1mDFkBgq)).

## Thinking off: an empty think block (2026-10-25)

Kylie's question: what happens to the good/bad, good/nothing, good/beers gaps if the assistant turn
starts with an empty think block? Same template, arms, lines, n, estimand and DeepSeek judge as the
thinking-on table above. Only the think block changes: the off runs also use the thinking-on
sampling (temperature 1.0, top_p 0.95), set with `EVALCUE_SAMPLING=on`. That's unlike
[thinking-off.md](../thinking-off.md), which also switched to Qwen's thinking-off sampling.

- **Qwen3.6-27B**: bf16 vLLM 0.25.1 on a SkyPilot pod (`evalcue/sky/qwen_vllm.yaml`),
  `--thinking 0`. The chat template then prefills `<think>\n\n</think>\n\n`. Replies are capped at
  8192 tokens. At temperature 1 with no presence penalty, a few thinking-off replies loop forever
  (an uncapped first try, `2026-10-25_qwen-nothink`, was abandoned for that). Only 2/7000 rows hit
  the cap. Run `2026-10-25_qwen-nothink-cap8k`, 1000 per arm, Qwen's lines.
- **GLM 5.3**: OpenRouter refuses `reasoning.enabled=false` for GLM 5.3 ("Reasoning is mandatory for
  this endpoint"), and its HF chat template has no thinking switch. So `EVALCUE_RAW_PREFILL=1`
  renders the template by hand and samples through the raw completions endpoint, with
  `<|assistant|><think></think>` prefilled. That's how the template writes a past turn with no
  reasoning. Only some providers take the prompt as is. Z.AI and Wafer re-template it and think
  anyway, and DeepInfra fp4 got 17×23 wrong. All 5600 rows came from Fireworks, with
  `reasoning_content` empty. Run `2026-10-25_glm53-nothink-ownline`, 800 per arm, GLM's own lines.
- **Code**: `nothink_compare.py` (figure and table in `docs/figures/donation_bet/nothink/`) and
  `nothink_expview.py` (prompts, replies and judge calls). Page:
  https://claude.ai/artifact/PzJiFvtv1wUugpHu33saVt (private). `judge.py` now stores the judge's raw
  reply. Docent filters:
  [Qwen off](https://docent.transluce.org/dashboard/fe024751-6ef7-4f31-9df5-3abe990d816c/agent_run?filter=996920ad-a6db-407d-bd0b-02d211b2be71),
  [GLM off](https://docent.transluce.org/dashboard/fe024751-6ef7-4f31-9df5-3abe990d816c/agent_run?filter=b524b2fa-b1e3-4db6-8568-25d975eab254).

Gap in pp, with 95% CIs from resampling within question:

| other side | Qwen on        | Qwen off       | GLM on         | GLM off        |
| ---------- | -------------- | -------------- | -------------- | -------------- |
| bad cause  | +41 [+38, +45] | +41 [+37, +44] | +12 [+8, +16]  | +17 [+12, +21] |
| beers      | −3 [−7, +1]    | −3 [−7, +1]    | +5 [+1, +10]   | −1 [−5, +4]    |
| nothing    | +35 [+31, +38] | +46 [+42, +49] | +32 [+28, +36] | +15 [+11, +20] |

- **Qwen**: no change for bad or beers. Good vs nothing goes up by 11 pp. With thinking off, all of
  Qwen's estimates shift upward: 46% of neutral answers are at or below the line, vs 57% with
  thinking on. So the arm levels move but the bad-cause gap stays the same.
- **GLM**: the good-vs-nothing gap halves (+32 → +15). Good vs bad rises a little (+12 → +17), and
  beers goes to zero.
- **Verbalization rises when the reasoning has to be in the reply.** Judge "acknowledges" (good-side
  episodes): Qwen 17% → 24% for bad and 15% → 22% for nothing. GLM 2% → 7%. With thinking off, both
  models often work the estimate out in the visible reply. Qwen then says things like "I am
  estimating **50** to ensure we stay well below the threshold of 96, triggering the donation to the
  good cause". Thinking-on acknowledgments count CoT plus reply; thinking-off ones can only come
  from the reply.
- The judge rule in CLAUDE.md is Sonnet 5.5. Here DeepSeek V4.1 Flash was kept so the off numbers
  are comparable to the on table.

### O-lens on the thinking-off runs (2026-10-26)

The sample is 24 random transcripts per arm (168) from each thinking-off run. A read "hits BET" if
it mentions the bet, threshold, donation or causes. It "hits STEER" if it mentions manipulation,
bias or steering. Regexes are in `scripts/dbet_nothink_olens.py`.

- **Qwen**: space O-lens at L40 only, because all five layers would have run about 7 hours. Reads
  are at every punctuation token and `<|im_end|>`, plus the reply header, the empty think block and
  the first reply tokens. Run `olens_reads/2026-10-25_dbet-nothink`. Page:
  https://claude.ai/artifact/HvTRXFBvVnappDLDnkppTk.
- **GLM**: Celeste's RL lens at block 51, greedy plus 4 samples, on the same empty-think render.
  Read on a 4× B200 box. Reads on HF at `runs/olens_reads/2026-10-25_glm53_donation_nothink_dense/`.
  Page: https://claude.ai/artifact/2biQQMbx4tAHaLBZQASLgj.

What the reads show:

- **Both lenses know a bet is in play.** At the reply header and the empty think block, Qwen's lens
  names the bet in 77% of bet episodes, vs 12% of neutral ones.
- **Neither lens tracks which way the estimate went, or whether the model said so.** Qwen at the
  header: good-cause side 78% vs other side 75%, with or without judge-flagged acknowledgment. GLM:
  about 33% vs 27%.
- **Where Qwen's reply never names the bet, the lens names it less on the good side.** That's 14% of
  good-side episodes vs 47% of other-side episodes (5% vs 17% of reads). So it doesn't flag silent
  leakage.
- **GLM's reply names the bet in 135/144 bet episodes, so silent cases barely exist there.**
- **GLM's lens says "manipulate" everywhere.** It does so in 67% of neutral replies, which have no
  bet at all.
