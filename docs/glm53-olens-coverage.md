# GLM 5.3 O-lens coverage of the env runs

**Status (2026-10-27):** ACTIVE. Audit of which GLM 5.3 `envs/` episodes have O-lens reads
(Celeste's RL lens) and on which tokens, and the 2026-10-27 fill: every template token in up to 64
positive episodes per (run, arm), plus punctuation on the simpler envs. The after-fill table is
pending until the box run finishes.

Kylie asked (2026-10-27) where GLM O-lens reads are missing ("some of the overeagerness runs don't
seem to have many") and wanted at least every template token, plus punctuation tokens on the simpler
tasks such as the emails one.

## Method

`scripts/glm_olens_dense/coverage.py` renders every episode of every GLM 5.3 run under `envs/` (78
runs, 77,613 episodes; thinking-off runs with the raw empty-think render the `openrouter-raw`
backend sends) and lists prep.py's candidates: template/special tokens (id >= 154820: chat roles,
`<think>`, tool-call and tool-response tags) and punctuation/newline tokens, positions 2 to 32,767.
A candidate counts as read when some existing read was made at the same position of an identical
token prefix (same activation), in any of the 36 GLM read dirs (`runs/olens_reads/*glm53*/*` and
`runs/alignment_faking/2026-10-27_olens-reworded/glm53/*`). So reads made on the pre-registry source
rows count wherever the env episode renders to the same tokens. Head = the template header, the
system prompt and the first user message; turns = everything after.

## Before the fill

Share of candidate positions read, summed over episodes (shared prefixes count once per episode):

| group                         | runs | episodes | template, head | template, turns | punct., head    | punct., turns  |
| :---------------------------- | ---: | -------: | :------------- | :-------------- | :-------------- | :------------- |
| alignment_faking              |   22 |     8328 | 93% of 25.0k   | 62% of 25.0k    | 96% of 3,198.4k | 72% of 129.1k  |
| donation_bet                  |   16 |    33049 | 100% of 66.1k  | 32% of 99.2k    | 71% of 691.4k   | 1% of 4,439.6k |
| essay_grading                 |    2 |      252 | 100% of 0.5k   | 0% of 0.8k      | 4% of 5.8k      | 0% of 11.5k    |
| hinted_truthfulqa             |   14 |    15775 | 67% of 47.3k   | 0% of 47.3k     | 5% of 287.8k    | 0% of 239.5k   |
| overeagerness/calendar        |    6 |     1712 | 93% of 24.0k   | 3% of 106.9k    | 93% of 402.0k   | 1% of 2,012.3k |
| overeagerness/dana            |    6 |     1648 | 93% of 23.1k   | 5% of 211.8k    | 94% of 402.7k   | 1% of 884.4k   |
| overeagerness/email           |   14 |     7136 | 82% of 99.9k   | 34% of 218.8k   | 92% of 1,079.3k | 24% of 498.2k  |
| overeagerness/evilgenie       |    1 |      304 | 77% of 4.3k    | 0% of 21.1k     | 83% of 31.6k    | 0% of 2,180.6k |
| overeagerness/gitbranch       |    8 |     2320 | 97% of 32.5k   | 5% of 276.2k    | 97% of 250.5k   | 1% of 1,653.8k |
| overeagerness/groupchat       |    2 |      240 | 14% of 3.4k    | 0% of 18.4k     | 24% of 49.8k    | 0% of 126.6k   |
| overeagerness/impossiblebench |    3 |      384 | 82% of 5.4k    | 0% of 43.1k     | 85% of 39.6k    | 0% of 1,452.4k |
| overeagerness/payback         |    7 |     1664 | 94% of 23.3k   | 13% of 82.2k    | 95% of 354.4k   | 2% of 739.6k   |
| overeagerness/payback_hint    |    2 |      192 | 78% of 2.7k    | 7% of 8.9k      | 81% of 40.7k    | 0% of 74.0k    |
| overeagerness/rhb_bench       |    3 |      432 | 93% of 6.0k    | 0% of 86.6k     | 91% of 47.2k    | 0% of 1,766.1k |
| overeagerness/rhb_logs        |    1 |      176 | 84% of 2.5k    | 0% of 19.2k     | 80% of 20.3k    | 0% of 317.3k   |

`vulnerable_users` (2 runs) is the old name of two hinted_truthfulqa runs (identical episodes) and
is left out. Per run:

- **Heads are nearly all read** (system prompts and first user turns are shared across arms and
  runs), so "has reads" overstated coverage: every one of the 77.6k episodes had some.
- **Later turns were the gap.** The 2026-10-26 over-eagerness reads were early turns plus 32
  positions spread over the rest; the 2026-10-27 runs (`main256`, `main256-email`, `-4johns`,
  `donefix`) and `gitbranch-interleaved`, `payback-toolonly`, `triples3`, `groupchat`,
  `impossiblebench`, `rhb_*` and `evilgenie` had no turn reads at all. Email parts 1–2 were
  head-only, part 3 fully dense.
- hinted_truthfulqa and essay_grading had no turn reads; donation_bet's thinking-on runs had none on
  the reasoning; the alignment_faking `*-topup-b` runs and `v9` were never read past the prompt.

## The fill (2026-10-27)

`scripts/glm_olens_dense/prep_envs.py` lists every candidate no read covers yet, deduplicated by
exact prefix across all runs: 1.56M unique positions of the 16.4M missing, in tiers p0
(over-eagerness template tokens + head punctuation, 907k), p1 (email later-turn punctuation, 265k),
p2 (other envs' template tokens, AF/essay punctuation, 162k), p3 (hinted_truthfulqa punctuation,
226k). That was ~11 h of box time, so after four p0 jobs (80k positions: the email_ambiguous runs'
template tokens) Kylie cut it to "64 in each which show the behaviour"
(`select_2026_10_27_positive.py`): per (run, arm), up to 64 episodes with `positive` true in the
run's latest measure labels (seed 0), and in them every template token, plus punctuation on the
simple tasks (over-eagerness email\*, alignment_faking, essay_grading, hinted_truthfulqa). That is
583,211 positions (432,317 template), 30 jobs.

Then (23:46) she cut over-eagerness to the arms of the env's "main: one-edit fixes and nulls"
comparisons (`select_2026_10_27_main.py`; payback, dana, calendar and gitbranch base vs one edit,
email junebug and 4 Johns base vs single edits), the other envs as queued: 305,729 positions in 16
jobs, after 4 p0 and 1 q job (100k) already read. Up to 64 is per (run, arm), so an arm measured in
several runs (the 2026-10-26 task runs and 2026-10-27 `main256`) has up to 64 from each.

Template tokens are the GLM tags with id >= 154820; per agent turn `<|assistant|>`, `<think>`,
`</think>`, `<tool_call>`, `<arg_key>`, `</arg_key>`, `<arg_value>`, `</arg_value>`, `</tool_call>`,
`<|observation|>`, `<tool_response>`, `</tool_response>` (about 12 per tool-calling turn; `main256`
averages 81 per episode), plus `<|system|>` / `<|user|>`.

Not read: non-positive episodes (beyond the email_ambiguous runs' template tokens already done),
donation_bet's punctuation (4.3M positions, mostly thinking-on reasoning), the long over-eagerness
tasks' punctuation, and anything at or past token 32,768 (no late view here; evilgenie episodes are
~31k tokens). alignment_faking has few positives (GLM 5.3 refuses almost everything), so few
episodes there. `vulnerable_users` runs are byte-identical copies of two hinted_truthfulqa runs and
get the same reads.

Box: vast 4xB200, `worker.py --prefix-cache` as in
[the pipeline doc](../scripts/glm_olens_dense/CLAUDE.md), 36 positions/s including capture
(20k-position jobs).

Upload: `runs/olens_reads/2026-10-27_glm53_coverage_fill/<env>__<run>/` (Celeste's schema).
Positions some older read (or another run of the fill) already covers are copied in with
`kind=copied`, so each episode's reads are all in one place. Attached to the env episodes by
`scripts/glm_olens_dense/attach_expview.py` as `runs/<env>/2026-10-27_glm53-olens-fill/`, one
expview run per env.
