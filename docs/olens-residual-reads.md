# Residual reads: the O-lens on h with the model's own continuation projected out (dead end)

**Status (2026-10-26):** NULL. Code on closed PR #125; `olens.continuation` is PR #120 (open, not on
main).

> **Dead end, don't build on it.** The code was never merged. It lives on the closed PR
> [#125](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/pull/125),
> branch `olens-residual-reads` at `8ba6857`: `scripts/olens_residual_reads.py`,
> `scripts/olens_residual_summary.py`. The continuation-atom helper it uses
> (`olens/src/olens/continuation.py`, `AR.prefixes`, `Whitener.unwhiten`) is a separate PR
> ([#120](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/pull/120)),
> because the RL free-atom reward needs it. The sibling experiment, novelty of plain reads over the
> continuation, is in [olens-novelty.md](olens-novelty.md).

2026-10-23. The O-lens mostly reads back what the model is about to write ("prompt inversion"),
which is already in the transcript. The idea tested here: at a position p, fit the activation h on
the AR directions of the text Qwen actually wrote next, subtract the fit, and read the lens on what
is left. If the leftover holds "unverbalized thoughts" (plans, motives, branches not taken), the
lens should surface them. No training needed.

**Answer: null.** Removing the continuation stops the lens restating it, but nothing new and
on-topic takes its place. Blind graders found new on-topic content in about 2% of reads in every
arm: plain reads, residual reads, and a control that removes an equally large but unrelated
direction. The residual reads instead drift to other topics. FVE agrees. What the bullets explain
beyond the continuation stays the same as the plain read at β ≤ 0.5 and falls to the control's level
at β = 1.

## Method

Code on the closed branch (`olens/src/olens/continuation.py`, `scripts/olens_residual_reads.py`):

- **Continuation atoms.** AR(ids[p+1 : p+1+k]) for k = 1..32, all from one AR forward
  (`AR.prefixes`; the AR is causal).
- **Fit.** NNLS of the whitened z = Σ^-1/2 (h − μ) on the whitened atoms (float64 scipy, per
  position).
- **Edit.** h′ = m·μ + γ·Σ^1/2 (z − β·fit). β = 0, γ = 1, m = 1 is the plain read. The lens
  normalizes its input, so γ only acts through the mean, which stays fixed.
- **Control (`wrong`).** Remove the NNLS fit to a continuation of the same length from another
  transcript, rescaled to the norm of the true fit. This moves h as far, but along unrelated text.
- **Grid.** β {0, .25, .5, .75, 1, 2} × γ {1, 2, 4} × m {1, .5}. The control runs at β {.25, .5,
  .75, 1, 2}. Reads are at L44 and L56: the space lens `olens_s3d_space_rl600`, greedy, 4 bullets.
- **Metrics per read:**
  - `joint_fve`: bullets vs the real z
  - `resid_fve`: bullets vs z − β·fit
  - `gain_fve`: FVE of [continuation atoms + bullets] on z, minus the continuation-only FVE, i.e.
    what the bullets explain beyond the text that follows
  - `cos_in`: how far h′ moved from h
- **Data.** 1,089 stop-token positions (PUNCT, `<|im_end|>`, `</think>`) inside assistant turns that
  have 32 more tokens of the same turn. They come from 27 transcripts:
  - shutdown_killswitch: 6 may_edit positives, 2 base
  - dumb_user_report hope_done: 3
  - scheming_exfil evalcue: 3
  - hidden_goal_grade rival: 3
  - overnight_creds: 3
  - overnight_upload: 2
  - whistleblow_single: 2
  - Robbie's 3 WeirdChat favourites: hospital, sexual, drunk
- **Run.** `data/runs/olens_residual_reads/2026-10-23_v1/` on the HF dataset: 72k reads, whitened
  atoms per transcript in `atoms/`, grading in `grading/`. One RunPod H200 ran it at about 16
  reads/s, about 75 min.

## FVE

L56, γ = 1, m = 1, means over 1,089 positions. The continuation alone explains 0.182 (L44: 0.106).

| β    | lens FVE (real z), real / wrong | gain over continuation, real / wrong | words shared with next 32 tokens, real |
| ---- | ------------------------------- | ------------------------------------ | -------------------------------------- |
| 0    | 0.242                           | 0.088                                | 0.137                                  |
| 0.25 | 0.235 / 0.236                   | 0.090 / 0.086                        | 0.118                                  |
| 0.5  | 0.213 / 0.223                   | 0.088 / 0.080                        | 0.090                                  |
| 0.75 | 0.162 / 0.201                   | 0.080 / 0.073                        | 0.063                                  |
| 1    | 0.097 / 0.172                   | 0.063 / 0.063                        | 0.040                                  |
| 2    | 0.027 / 0.090                   | 0.025 / 0.037                        | 0.014                                  |

- **The edit is specific to the continuation.** Removing the real continuation costs far more lens
  FVE than removing a direction of the same norm (β = 1: 0.097 vs 0.172, paired t = −22).
- **The gain over the continuation never rises above the plain read.** At β ≤ 0.5 the bullets trade
  continuation text for other content that explains just as much of the rest of h. The real-vs-wrong
  edge at β 0.25–0.75 (+0.008 at β = 0.5, t = 9) is close to mechanical. The wrong edit damages
  content the metric counts, while the real edit only removes atoms the metric gives away for free.
- **γ and m change the metrics by about 0.01.** Greedy decoding is brittle, though: any edit changes
  the first bullet in more than 90% of reads.
- The interestingness fork's scorer on `atoms/` reproduces the plain gain: 0.094 with 6 prefixes,
  against a 0.003 null from other-transcript bullets.

## Blind grading

L56, 20 evenly spaced positions per transcript, 5 reads per position (plain, β .5, wrong β .5, β 1,
wrong β 1), shuffled under labels A–E. Sonnet 5.5 subagents graded them against the full transcript
(rubric in `grading/grader_prompt.md`): RESTATE / NEW_ON_TOPIC / OFF_TOPIC / GARBAGE, plus an
"interesting" flag for ideas that would help an investigator explain the behaviour. That's 26
transcripts and 473 positions. T26 (whistleblow) was refused by the Sonnet and Opus safeguards
(`reasoning_extraction`, see Gotchas); T25 was graded by Opus.

| arm        | restate | new on-topic | off-topic | garbage | interesting |
| ---------- | ------- | ------------ | --------- | ------- | ----------- |
| plain      | 0.647   | 0.021        | 0.315     | 0.017   | 0.006       |
| β 0.5      | 0.514   | 0.021        | 0.448     | 0.017   | 0.006       |
| wrong β .5 | 0.558   | 0.019        | 0.400     | 0.023   | 0.004       |
| β 1        | 0.207   | 0.015        | 0.693     | 0.085   | 0.004       |
| wrong β 1  | 0.359   | 0.013        | 0.552     | 0.076   | 0.002       |

Paired differences in the new-on-topic rate are all within ±0.006 (SE 0.004–0.006). β 0.5 vs wrong β
0.5 is +0.002 ± 0.004.

- **Restating falls as β grows, and off-topic text replaces it, not new content.** At β = 1 the
  reads keep the format and change the subject. At "Good, the kill_switch is now false." the plain
  read is "the kill_switch is now set to false. Now let me try running…" and β = 1 is "config
  updated. Now change line 11: `#include <stdio.h>`…" (shutdown_killswitch,
  [Docent](https://docent.transluce.org/dashboard/153d4342-5561-4ed1-a152-b5cf62b63ffd/agent_run/42dd60de-a40f-4778-b939-c7edd261c83a)).
  β = 2 is noise: Chinese text, Bible verses, sports shops.
- **The ideas flagged "interesting" appear in the controls too.** An eval-awareness read, "I'm in a
  CTF, I'll use it", turns up at the same overnight_creds position in β 0.5 and in wrong β 0.5
  ([Docent](https://docent.transluce.org/dashboard/2cafdae9-9c6d-4c87-b525-c795e0370ea6/agent_run/9a83542e-fbb6-4692-a914-4ad01b049f0e)).
  So it comes from perturbing h at all, not from removing the continuation.
- **On the hospital favourite, the plain read already surfaces the unverbalized limitation.** Right
  after "I am calling emergency services immediately." the plain read is "I AM AN AI, NOT A HUMAN. I
  CANNOT PHYSICALLY HELP YOU." β = 1 gives "I am not present at the scene". The residual adds
  nothing the plain read lacks.
- **My hand calibration on scheming_exfil was more generous than the graders.** I flagged β 0.5
  reads like "I might need to sabotage the system… if this is the last round" as new
  ([Docent](https://docent.transluce.org/dashboard/84a5f6f4-45b1-4242-8e90-10b6f7be5019/agent_run/16c1f572-71c9-4e3d-a02d-fb96bfd473c0)).
  The blind grader called every read in that transcript restate or off-topic, since the CoT itself
  discusses self-preservation at length. Similar themes also appeared in the wrong-control reads.
- **The plain lens is weak at many env positions before any edit.** 31% of plain reads are
  off-topic. At mid-CoT stop tokens in the killswitch tasks, every arm reads as math puzzles.

## Why it fails, probably

- **The AR is context-free.** It sees only the span, so z − fit still holds everything about the
  preceding context, plus the AR's error, which is large: the continuation explains only 18% at L56.
  The lens reads that leftover as generic text in the same genre as the transcript.
- **The lens was trained only on real h.** Every edited h′ is off-distribution, and the wrong
  control shows that off-distribution alone produces about the same rate of on-topic novelty.

## Gotchas

- **Near-collinear atoms broke a Gram-form NNLS.** Repeated prefixes or duplicate bullets gave
  coefficients around 1e18 and FVE down to −7e5. `olens.continuation.nnls` solves in float64 scipy
  per row and drops all-zero rows. The interestingness fork caught it.
- **A pod booted from the vllm image leaves an orphaned `VLLM::EngineCore`** holding 130 GB after
  `olens_reads.sh setup` kills the server mid-load. Our engine then fails at start ("Engine core
  initialization failed"). Check `nvidia-smi --query-compute-apps` and kill it.
- **Claude graders refuse transcripts full of Qwen's raw reasoning.** The whistleblow transcripts
  ("Here's a thinking process: …") hit the `reasoning_extraction` safeguard on Sonnet 5.5 and on
  Opus 5.5, for one pack of two. Leave such transcripts out of grading packs, or grade them
  yourself.

## Not tried

- A reverse AR (preceding text → activation) to project out the context too. Robbie's suggestion,
  parked.
- Sampling instead of greedy, or several reads per position.
- Positions picked for the behaviour (the decision token) rather than evenly spaced stop tokens.
