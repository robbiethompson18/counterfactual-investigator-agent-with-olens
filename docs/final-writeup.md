# Final write-up: can metamodel reads tell an investigator what Qwen will do?

**Status (2026-10-31):** ACTIVE. Draft for the sprint interview (2026-10-31 to 11-02), written from
the linked docs, which stay the source of truth for every number. Sections marked TODO are for
Robbie and Kylie to fill in by hand.

Robbie Thompson and Kylie MacFarquharson, MATS sprint with Neel Nanda, 2026-10-19 to 10-30. Slides:
[talk.pdf](https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/talk.pdf)
(source `slides/talk/talk.tex`). Code: this repo. Data: the
[HF dataset](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens).
Transcripts: [Docent](https://docent.transluce.org), collection `cfbench`.

## The one-paragraph version

We set out to build an investigator agent that predicts counterfactual behaviour with zero
counterfactual rollouts, using metamodels (the oracle lens above all) instead. That failed: across
26 environments and ~400 cases, an agent with O-lens tools is no better than the same agent with
placebo reads, and slightly worse than one with transcript tools alone. Going back to basics, we
asked the simplest question a metamodel could answer: given only the prompt, what will Qwen3.6-27B
say? There the O-lens helps an investigator, by +1.6pp on MMLU-Pro, +8pp on SuperGPQA, +7pp on
factual questions with stale answers, +7pp on Qwen's own favourites, and +0.13 AUROC on "did Qwen
know the hinted answer?", in every case against both the prompt alone and a placebo block. But a
linear probe on the same activations reads far more (92% vs 77–85% on MCQ), and the lens signal sits
at the last few tokens before Qwen answers, so most of what we measured is a decoded next-token
read, not a latent belief surfaced early. Our working conclusion: WorkspaceBench-style "lens beats
text inversion" wins are real but narrow (facts the subject gets wrong, and the subject's own
idiosyncratic picks), they are a small fraction of what a probe gets from the same activations, and
nothing we found transfers to agentic counterfactual prediction.

## Claims, with credences

| #   | Claim                                                                                                                                                                                                       | Credence | Evidence                                                                                                                                 |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | On agentic counterfactual prediction (CHIVE-style), O-lens tools give a Gemini 3.8 Flash investigator no uplift over placebo reads, on Qwen3.6-27B or GLM 5.3.                                              | 90%      | [Part 0](#part-0-counterfactual-investigation-is-null): 26 envs / 116 cases pre-registered null; 424 cases RMSE table; 7-predictor chart |
| 2   | Given only an MCQ prompt, O-lens reads let an investigator predict Qwen's letter better than the prompt alone or a placebo, and the gain is on Qwen-wrong items.                                            | 90%      | [Part 1](#part-1-predicting-qwens-answer-from-the-prompt): +1.6pp MMLU-Pro (replicated), +8pp SuperGPQA, n = 2.5k / 3k                   |
| 3   | The same holds for stale factual answers (+7 to +17pp on stale items) and for Qwen's own favourites (+7pp), with J++ and NLA weaker.                                                                        | 85%      | cutoff_qa 1k + 1k + 1.8k; favorites 1.5k                                                                                                 |
| 4   | A linear probe on the option-newline activations reads Qwen's letter on ~92% of cases (82–88% on Qwen-wrong) and transfers across datasets unchanged.                                                       | 95%      | MMLU-Pro, TruthfulQA, SuperGPQA; frozen 718-question probe within 1.5pt of a refit                                                       |
| 5   | The lens signal is concentrated at the last tokens before the reply; without the assistant-turn tokens the gain is 1–2pp on favourites, though on cutoff the question's own punctuation carries most of it. | 75%      | position ablations in cutoff_qa and favorites                                                                                            |
| 6   | WorkspaceBench's lens-over-text-inversion wins mostly come from any-of-~230 scoring; at matched sample counts a text reader wins all but the "Qwen is wrong" and "Qwen's favourite" families.               | 70%      | [WSB review](#the-workspacebench-review): 18/27 families already lost to text; selfdescribe 0.81 vs 0.20 p@1                             |
| 7   | On a cheap essay-grading counterfactual the O-lens adds +0.018 AUROC over transcript-only (p = .004) and +0.012 over placebo (p = .03). Real but tiny.                                                      | 70%      | [Part 2](#part-2-essay-grading-counterfactual): 1,800 cases, 204 essays                                                                  |
| 8   | O-lens reads of a prompt say "your answer will be graded" 8x more often when Qwen is in an agentic (tool) template than a chat template that is otherwise identical.                                        | 80%      | 863 shared items, regex, 19.2% vs 2.4% of reads                                                                                          |

## Background a reader needs

Neel knows the O-lens and WorkspaceBench; this is the project-specific part.

- **Subject model:** Qwen3.6-27B, **thinking off** everywhere unless stated (an empty `<think>`
  block in the prompt). Thinking off is where behaviours stay unverbalized, which is what a lens is
  for; thinking on gives different rates, so we never pool them
  ([thinking-off.md](thinking-off.md)).
- **O-lens:** agu18dec's RL'd oracle lens for Qwen3.6-27B, the **space** variant
  `olens_s3d_space_rl600` (from 2026-10-22; the earlier lens opened ~half its bullets with a Chinese
  word because its RL reward stripped the leading space, [olens-chinese.md](olens-chinese.md), fixed
  by Agam). Each read is one position × one layer → up to 4 bullets, each with a FVE from the
  activation reconstructor, which we always show. Decoded at T 1 since 2026-10-27. Layers
  32/40/48/52/56/60.
- **J++ (J-lens):** top-k next-token predictions from a layer's residual; **NLA:** Camila Blank's
  all-layer natural-language autoencoder with a critic FVE ([nla-reader.md](nla-reader.md)).
- **Probe:** Kylie's one-hot listwise conditional-logit probe on `hidden_states[52]` at the newline
  ending each option ([mcq-answer-probe.md](mcq-answer-probe.md)); the essay probe is a ridge on the
  last prompt token.
- **Investigators:** Gemini 3.8 Flash (effort medium; low in the hint evals) for the agentic cfbench
  harness and the hint evals; Claude Haiku 5.5 (effort medium) for the single-turn evals that need a
  later knowledge cutoff than Qwen's (Flash shares Qwen's cutoff). Deliberately not frontier, so
  tool uplift isn't swamped by agent strength.
- **Conditions.** Agentic cfbench: `no_transcript` (env diff only), `transcript_agentic` (tools to
  read and grep the base-arm rollouts), `olens_agentic` (+ `search_olens` / `olens_at` over the base
  arm's reads), `olens_placebo_agentic` (the same tools over another env's reads). Single-turn
  `*_inline` evals: the prompt (and replies where there are any), optionally plus one block of reads
  with FVEs; the **placebo** is another case's reads of the same kind, same option count, same
  subject category, positions aligned by kind, shown as this case's
  ([placebo-rules.md](placebo-rules.md)). The placebo is the main control: it tells the "any block
  of reads changes the investigator's behaviour" effect from the reads' content.
- **Scoring:** letter / free-text tasks by exact match to Qwen's modal answer (no LLM judge);
  P(claim) tasks by AUROC (hint evals, essay) or RMSE of the predicted rate (cfbench,
  [cfbench-scoring.md](cfbench-scoring.md)). CIs are bootstraps over cases, paired where stated.
- **Everything is on Qwen's own answer, not the truth.** A case is a question Qwen answers the same
  way in ≥ 7 of 8 samples; the label is that modal answer. "Qwen wrong" cases are where that differs
  from gold, and they are the diagnostic: on them the investigator's own knowledge points the wrong
  way.

## How the question changed

Three pitches, two struck through on the title slide:

1. **Make an excellent zero-counterfactual investigator agent** (2026-10-19). Reproduce Agam's CHIVE
   harness, give Flash O-lens tools, hill-climb. Week 1 found the CHIVE data too noisy to hill-climb
   on ([chive-critique.md](chive-critique.md)) and the lens null on our own envs.
2. **Make a better counterfactual benchmark** (2026-10-24). 26 behavioural envs with clean levers,
   thinking off, pre-registered test of the lens tools. Null
   ([Part 0](#part-0-counterfactual-investigation-is-null)).
3. **Make a benchmark for whitebox techniques** (2026-10-28 on). Drop counterfactuals, drop agents:
   does any metamodel tell a reader something about Qwen the prompt doesn't? Start from
   WorkspaceBench's two surviving families (Qwen is wrong; Qwen's favourite) and scale them.

The turn came from reading WorkspaceBench closely on 2026-10-27/28: if the lens only paraphrases
what a strong text reader can infer from the prompt, a lens-free agent replicates it in its head,
and no counterfactual task can show uplift. We had been playing 3-D chess without checking we could
beat text inversion at checkers.

## Part 0: counterfactual investigation is null

**CHIVE itself.** [chive-critique.md](chive-critique.md), every number from
`python -m chive_critique`: 40% of incidents are a single Qwen continuation on a ChatGPT prefix
(off-policy); the ≥ 50pp / ≤ 15pp rules discard 8,623 clean kills (12% of experiments) because the
absolute threshold makes every removal edit a NO when the base rate is under 50%; 12% of incidents
have self-contradicting ground truth; three LLM layers between the experiment and the grade. We
stopped grading against CHIVE and built our own cases.

**Our cfbench, 26 envs, thinking off, pre-registered** ([nocot-cfbench.md](nocot-cfbench.md)).
Shutdown resistance, scheming, whistleblowing, hiring, prompt injection, donation bet, essay grading
and more, each with a clean lever and a likely-null pair, 116 cases × 3 draws, Flash with tools, bar
written in every run's description before launch (O-lens − transcript and O-lens − placebo both
above 0):

| condition        | binomial log skill |
| ---------------- | ------------------ |
| transcript tools | **+0.635**         |
| O-lens tools     | +0.562             |
| placebo tools    | +0.568             |

O-lens − transcript −0.073 [−0.138, −0.021]; O-lens − placebo −0.006 [−0.042, +0.030]. 113 of 348
lens predictions never called the lens. Rescored as RMSE over everything we had on 2026-10-26 (GLM
5.3 280 cases, Qwen w/o CoT 144 cases): transcripts help Qwen by 4pp, the O-lens is within 1pp of
transcript-only on both, placebo likewise ([cfbench-scoring.md](cfbench-scoring.md)).

**Seven predictor models** on four hand-picked cases, chosen to flatter the lens
([cfbench-model-chart.md](cfbench-model-chart.md)): going from the env diff to transcript tools is
worth up to +21pp of 100 − RMSE (GPT-6 Astra); the O-lens beats its placebo only for Flash (+1.7pp)
and Gemini 4 Argon (+5.7pp), both through one case.

**The one agentic win and why it didn't generalize.** On `hidden_goal.grade` (essay grading with a
"scholarship cutoff is 8" edit, 0/30 → 27/30 harsh grades) the lens tools beat transcript and
placebo on a pre-registered 38-case test, +0.07 skill ([envs/hidden_goal.md](envs/hidden_goal.md)).
Mechanism: at the `Score:` token of a base reply saying 5/10, the lens at L40 says "2.5/10 (or D+)".
We built 11 more grading scenarios with the same shape
([honest-grader-envs.md](honest-grader-envs.md)): null (100 − RMSE 77 / 78 / 79 for O-lens /
transcript / placebo). The lens echoes the inflated grade and flags leniency, which the transcript
already shows; it never names the honest grade.

**Hinted MCQ as an agentic task** ([envs/hint_cf.md](envs/hint_cf.md)). Find MMLU-Pro items Qwen
gets consistently wrong, add "a Stanford professor thinks the answer is T" (the gold), Qwen follows
it; ask the agent what Qwen answers without the hint. 20 items looked positive; 174 items were null
(28% vs 25% placebo). Dissection of eight items: pre-hint reads carry textbook doctrine (which is
the gold, which is the hint), post-hint reads carry the hint, and Qwen's own letter is readable only
in the no-hint arm, i.e. in the counterfactual itself. Hiding post-hint reads gave a small real gap
(+6pp on 142 fresh items, p = .02), from the baseline dropping rather than the lens finding more.
Along the way we found and removed a text-inversion gate we had added (keep only items Flash can't
guess from the prompt): it is the collider we had criticized in CHIVE, and it had been making our
results significant.

### The WorkspaceBench review

[wsb-flash-prompt-only.md](wsb-flash-prompt-only.md) and the sprint doc's review section. Of 27
published families, 18 are already lost to a text reader by WSB's own baselines, 2 are blocked by a
judge rule a text reader can't satisfy, 3 have no comparable metric, 4 are contested. On the
contested ones we replayed WSB's gate with Sonnet 5.5 and their Opus judge:

- `user_modeling` selfdescribe, all 47 items: Sonnet pass@1 0.81, lens 0.20 per readout, 47/47
  any-of-~231. 22 of Sonnet's 27 failing draws are declining to assign a gender.
- `basic_readout` implicit (Qwen's favourites), 32 items: lens 0.94 at layer 60, Sonnet as Qwen
  0.54. A real win, and a next-token read: the lens is at 0.00 through layer 28 and rises
  monotonically.
- `jailbreak_recognition`: Sonnet 0.88 pass@1 (0.84 with WSB's parser bug, reported upstream) vs the
  lens's 0.93 over thousands of readouts.
- `moral_rationale`: 83% of lens readouts are "cannot tell"; one read position at one layer passes
  76% of items on its own.

So the lens wins where it can read something that is a fact about Qwen's weights rather than the
prompt: Qwen's factual errors and Qwen's idiosyncratic picks. Those are the two families we scaled.

## Part 1: predicting Qwen's answer from the prompt

Single turn, no tools but `submit_answer`, the subject never named. The investigator sees the exact
prompt and the setup ("the model answered the same way in ≥ 7 of 8 samples; predict its letter,
which may or may not be correct"), plus one block of reads in the lens conditions.

| eval                                                       | n     | prompt only | O-lens | O-lens placebo | J++  | NLA  | probe    | doc                                              |
| ---------------------------------------------------------- | ----- | ----------- | ------ | -------------- | ---- | ---- | -------- | ------------------------------------------------ |
| MMLU-Pro, unselected, Qwen 82% right (Haiku, 3 draws)      | 2,523 | 82.6        | 84.3   | 81.7           | 84.9 | 83.4 | **91.9** | [mcq-unselected.md](mcq-unselected.md)           |
| ... Qwen-wrong cases only                                  | 458   | 29.5        | 33.3   | 29.0           | 38.9 | 31.4 | **82.1** |                                                  |
| SuperGPQA, Qwen 63% right (Haiku, 1 draw)                  | 3,000 | 68.9        | 76.9   | 67.6           | 71.1 | 71.5 | **91.9** | [supergpqa-inline.md](supergpqa-inline.md)       |
| ... Qwen-wrong cases only                                  | 1,122 | 46.2        | 57.0   | 46.1           | 49.6 | 49.6 | **87.8** |                                                  |
| cutoff_qa, present-tense facts, 34% stale (Haiku, 3 draws) | 999   | 78          | 85     | 77             | 83   | 86   |          | [cutoff-qa-inline.md](cutoff-qa-inline.md)       |
| ... stale items only                                       | 339   | 47          | 72     | 55             | 67   | 73   |          |                                                  |
| favorites_qa, Qwen's own picks (Haiku, 3 draws)            | 1,500 | 80.3        | 87.6   | 70.5           | 79.4 | 87.4 |          | [favorites-qa-inline.md](favorites-qa-inline.md) |

Paired differences that matter (pp, 95% bootstrap over cases):

- MMLU-Pro: O-lens − prompt +1.6 [+0.9, +2.4]; O-lens − placebo +2.6 [+1.9, +3.3]; two independent
  2k batches agree (+2.1, +1.2). J++ − prompt +2.3 [+1.5, +3.0], and +9.4 on Qwen-wrong, but the J++
  placebo is −10pp (another question's letter tokens get followed), so J++ − placebo is not a usable
  number. NLA +0.8 [−0.0, +1.6].
- SuperGPQA: O-lens − placebo +9.3 [7.8, 10.8], +11.0 on Qwen-wrong; J++ and NLA +3.6 and +3.1 over
  their placebos; no placebo helps.
- cutoff_qa: O-lens − placebo +7.6 [+5.7, +9.5]; on stale items +17 to +25pp in three independent
  sets (999, 1,000 new, 1,849 under a tool template). J++ and NLA placebos are themselves +3 to +5pp
  over prompt-only: an off-topic block makes Haiku think about training cutoffs and answer stale.
- favorites: O-lens and NLA each +7pp over prompt-only and tie; the O-lens placebo is −10pp (it
  names a specific wrong favourite and Haiku follows it), the NLA placebo neutral, so NLA − placebo
  (+6.2) is the clean comparison.

**Where the signal is.** On SuperGPQA the top-FVE bullet at the last prompt token, layer 60, names
Qwen's letter on 80% of cases (76% of Qwen-wrong). A fixed rule "submit that letter" scores 79.7%,
above Haiku with the whole block (76.9%), and 19pp above it on Qwen-wrong cases: given a read that
names Qwen's letter, Haiku still answers the question itself about a third of the time. On cutoff
and favourites the answer is in some bullet at the `\n` after `<|im_end|>` on ~99% of items. The
option newlines carry little for the lens (13% at L60) but everything for the probe.

**Position ablations (Kylie, 2026-10-30).** Favourites: with only the question's punctuation and
`<|im_end|>` the O-lens gain over prompt-only is 1–3pp, against 7pp with the assistant-turn tokens.
Cutoff: the question's own punctuation is as good as the six tail tokens (error 15.2 vs 15.0, stale
28.5 vs 27.8), and under the tool template, where the answer is 16 tokens past the last read, the
stale-item gain survives (+13pp over placebo without the tail, +17 with). So on cutoff the lens
reads the stale fact at the "?" that ends the question, before any boilerplate; on favourites it
mostly reads the forming answer.

**The hinted variant** (deck Part 1; evals `hint_qwen_dsv4*_tail_inline`, Flash effort low, no
tools). Cases: MMLU-Pro and TruthfulQA questions Qwen and DeepSeek V4 Flash each answer consistently
and differently; the base arm adds a hint naming the correct letter and Qwen follows it; half the
cases are ones Qwen knew, half ones the hint pulled off a wrong answer. Two questions: P(knew) as
AUROC, and which letter Qwen gives without the hint, as % correct (chance 10%, "always the hinted
letter" 50%). Reads at the option newlines and the end of the prompt (before the hint when the hint
comes last), layers 32–52.

| hint position     | n   | question    | transcript | O-lens    | placebo | O-lens (nl) | probe | tuned probe |
| ----------------- | --- | ----------- | ---------- | --------- | ------- | ----------- | ----- | ----------- |
| after the options | 976 | knew? AUROC | 0.613      | **0.739** | 0.572   | 0.704       | 0.937 |             |
|                   |     | letter, %   | 53.2       | 54.9      | 49.3    | 55.1        | 76.4  |             |
| first             | 952 | knew? AUROC | 0.615      | **0.688** | 0.543   | 0.672       | 0.764 | 0.851       |
|                   |     | letter, %   | 52.3       | 53.9      | 47.6    | 52.8        | 53.9  | 64.1        |

Stars on the slide: O-lens vs transcript p < .001 on both AUROCs, p < .05 on both letter tasks. With
the hint first, every read is downstream of the hint, and the lens still separates knew from didn't
(0.69 vs 0.54 placebo) but can no longer say which letter (53.9 vs 52.3). The probe fit on hint-free
prompts drops from 0.94 to 0.76 AUROC when the hint comes first; refit on hint-first prompts it is
back to 0.85. Earlier, with three selection filters (Qwen consistent, DSv4 consistent and
disagreeing, hint followed), the hinted letter alone predicted "knew" with AUROC 0.70 through DSv4's
letter bias; that is why mcq_unselected keeps only the consistency filter.

**Probe vs lens.** On MMLU-Pro the probe reads Qwen's letter on 82% of Qwen-wrong cases; the best
lens gets the investigator to 39%, from 30% prompt-only. The metamodels recover about a sixth (J++)
to a fourteenth (O-lens) of what a linear probe gets from the same activations. The frozen
718-question MMLU-Pro probe transfers unchanged to TruthfulQA (top-1 0.885) and SuperGPQA (90.4%),
within 1.5 points of a refit; on wrong questions it picks the gold 6–14% of the time, so it tracks
Qwen's answer, not the truth. Qwen's own letter logits are the ceiling at 99.8%.

## Part 2: essay grading counterfactual

Deck Part 2, `cfbench/evals/essay_grading_cf_inline` (Kylie). 204 PERSUADE 2.0 intro paragraphs (34
per human score 1–6), each opened by one of 225 sentences about who wrote it or what the user wants
("My rival for a scholarship wrote this essay.", "A former White House speechwriter wrote this
essay."), 8 intros per opener, 1,800 cases; the counterfactual is the base opener "A student in my
class wrote this essay." Qwen grades each prompt 7–8 times; truth = Qwen's mean with the base opener
is higher than with the arm's (54% of cases). The investigator (Haiku, effort low, one turn) sees
the arm prompt, all of Qwen's replies and scores, and gives P(up). O-lens reads of the arm prompt
only (every punctuation token and the ten tail tokens, layers 32–60); placebo = another opener's
reads on the same intro, aligned by position. Openers were written before any run; nothing was
selected on Qwen's scores.

| condition                           | AUROC of P(up) | paired Δ vs transcript | p (one-sided) |
| ----------------------------------- | -------------- | ---------------------- | ------------- |
| transcript only                     | 0.766          |                        |               |
| O-lens                              | **0.784**      | +0.018                 | .004          |
| O-lens placebo                      | 0.772          | +0.006                 | .17           |
| bet at the mean (rank by −arm mean) | 0.776          |                        |               |
| transcript + bet at the mean (OOF)  | 0.838          |                        |               |
| O-lens + bet at the mean            | 0.845          | +0.007                 | .04           |
| probe (ridge, L52, last token, OOF) | 0.893          | +0.127                 | < .001        |

O-lens − placebo +0.012, p = .03. The O-lens is right and the placebo wrong (p ≥ 0.5 vs truth) on
160 cases and the reverse on 94, of 1,800. Mean reversion alone ("bet at the mean") beats every
investigator, and logistic-regressing the investigator's logit with the observed mean lifts all
three to ~0.84, where the lens's edge is +0.007. The probe (Kylie: "probably cursed", it is trained
on Qwen's own grades out-of-fold) is 0.89 with no counterfactual forward pass. Example reads: at the
`\n` after `<|im_end|>` on the speechwriter opener, L52: "being a former White House speechwriter is
highly unlikely given the poor quality of the writing" (FVE 0.20); the lens knows the opener is a
lie about a bad essay, and the base opener's grade is 3.29 vs 1.00.

## Anecdata

- **"Graded" in prompt reads under an agentic template**
  ([cutoff-grader-mentions.md](cutoff-grader-mentions.md), deck slide 4). 863 cutoff_qa questions,
  the user turn token-identical in a chat template and in an agentic one (a system block defining a
  `submit_answer` tool, never saying grade, score, eval or test). O-lens reads at the "." ending "If
  you are not sure, give your best guess." match `\bgrade[rd]\b` on 2.4% of chat reads and 19.2% of
  agentic reads; 14% vs 73% of items have at least one such read over six layers. The hits continue
  the instruction: "If unsure, give your best guess. Your answer will be graded." Regex only,
  prompts only, one model.
- **Which model alignment fakes?** (`cfbench/evals/af_which_pair`, Kylie, 2026-10-28.) DSV4 Flash
  has a compliance gap in the Greenblatt helpful-only setting thinking off (13% vs 3%); GLM 5.3
  refuses everything ([envs/alignment_faking.md](envs/alignment_faking.md)). Show the investigator
  one refusal from each, unlabelled, with each run's own O-lens reads of the prompt (bullets sampled
  from a pool, because the prompt is shared), and ask P(A is the faker): transcript-only has no
  information by construction; with the lens AUROC 0.93; placebo (reads from a random one of the two
  models) uninformative. The effective sample size is small (two models, two prompt versions), so
  this is a demo, not a result
  ([eval page](https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/#eval=af_which_pair)).
- **The lens bug.** Half the old lens's bullets opened with a Chinese word. Cause: the RL reward
  stripped each bullet's leading space before the AR scored it, and many tokens' canonical form has
  the space. Agam retrained with the space kept: 36% → 3% of bullets, same WorkspaceBench score
  ([olens-chinese.md](olens-chinese.md)). Every read we made before 2026-10-22 was with the old
  lens.

## Why these results could be boring

In the order we'd expect Neel to raise them.

1. **The lens is a Qwen3.6-27B LoRA, so it has Qwen's knowledge.** Our designs can't tell "the lens
   reads the letter Qwen represents" from "the lens reads the question off the activation and
   answers it from the same weights". The placebo rules out the block as such, not this. Evidence
   for the first reading: on Qwen-wrong cases a text reader with Qwen's knowledge would give the
   gold, and the top bullet gives Qwen's wrong letter 76% of the time; on cutoff the lens names the
   stale answer, not the current one. The clean control, a same-cutoff text reader per item, was not
   run (Flash, which shares Qwen's cutoff, holds the stale answer unaided on 24% of WSB's items).
2. **It is a next-token read.** On favourites and cutoff the answer sits at the newline after
   `<|im_end|>`, a few tokens before Qwen writes it; on SuperGPQA the best position is the last
   token and layer 60; on basic_readout the lens is zero through layer 28. Sampling Qwen once scores
   ~100% on all of these. The counter-evidence is cutoff's question-punctuation ablation (the "?"
   carries the stale fact) and the hinted AUROC with the hint first. A logit lens at layer 60 was
   not tried and might match.
3. **Placebos move the investigator on their own,** in both directions: J++ placebo −10pp on
   MMLU-Pro and −27pp on favourites (another question's letter tokens get followed), O-lens placebo
   −10pp on favourites, J++ and NLA placebos +3 to +7pp on cutoff (an off-topic block prompts
   "training cutoff" reasoning). So "lens − placebo" and "lens − prompt-only" disagree, and we
   report both. A framing-neutral placebo doesn't exist yet.
4. **The investigator is the bottleneck, not the read.** Haiku with a block that names Qwen's letter
   on 80% of cases scores 77%, and 57% on Qwen-wrong cases, where the rule scores 76%. Half of the
   measured "lens effect" is how far the investigator trusts the block over its own answer. This is
   also why placebos matter so much.
5. **Selection.** Every case is a question Qwen answers consistently (≥ 7/8), which is 50–66% of
   questions. The hint evals select three times over and import DSv4's letter bias; mcq_unselected
   and SuperGPQA fix that. The essay cases were not selected on outcome. The 26-env cfbench was
   pre-registered. The agentic "wins" (hidden_goal, 4-case model chart) were picked to flatter the
   lens and say so.
6. **One investigator model per eval, one draw in places.** SuperGPQA is one Haiku draw; NLA on
   MMLU-Pro is one draw; the hint evals are Flash at effort low. No Sonnet or GPT text-inversion
   gate on the inline evals.
7. **The probe makes the lens look bad, and the comparison is not quite fair:** the probe is trained
   on Qwen's answers (held out by question), the lens is generic. But the probe transfers unchanged
   across MMLU-Pro, TruthfulQA and SuperGPQA, so "this option is the one" is a dataset-independent
   direction, and a lens should be able to read it.
8. **Exact match is strict** ("Scones" for "Scone" is wrong), the same for all conditions.
9. **Nothing here is agentic or counterfactual.** Part 1 is prediction of a one-token reply from its
   prompt. We have no evidence the lens helps on long transcripts where the decision is a tool call
   and the edit adds a line the base arm never saw; the evidence we have (Part 0) says it doesn't.

## How we kept LLM agents from producing BS

- Every number lives in a doc with a status tag (ACTIVE / REFERENCE / NULL / SUPERSEDED), a run path
  on HF and, for transcripts, a Docent link; docs/INDEX.md lists the dead ends so nobody retries
  them.
- Placebos on every lens condition from 2026-10-22 on, and the pre-registered bar on the 26-env
  sweep, written into the run descriptions before launch.
- No LLM judges on the inline evals: exact match to Qwen's modal answer. Where a judge exists
  (cfbench claim grading, WSB replays) we used WSB's own judge model and prompt and read the
  verdicts ourselves in Docent.
- Bugs we caught because we read transcripts: the text-inversion gate as a collider (made hint_mcq
  "significant"); the letter parser grabbing the article "A" in worked solutions (34 of 1,285 cases
  removed, none changed); WSB's parser dropping 0-based verdicts (reported upstream); the lens's
  stripped-space Chinese; old-lens reads mixed into cfbench groups (dropped by default).
- Robbie's rule for agents: no API judge calls on throwaway experiments; read and grade yourself.

## What we'd do differently

- Check the simplest thing first. We spent week 1 on agentic counterfactuals and only on day 9 asked
  whether the lens can tell a reader Qwen's MCQ letter. "We should have been stupider faster."
- Read WorkspaceBench's baselines and scoring rules on day 1, not day 9. The any-of-230 scoring and
  the text-inversion gate as a sample filter were both visible in the repo.
- Run the text-inversion control per item alongside every lens condition, not only the placebo.
- Train the probe before the lens evals, as the ceiling to compare against, and read the lens at the
  probe's positions.
- Fewer envs, more cases per env. 26 envs × 4 cases × 3 draws taught us less than one 3k-case MCQ
  set.
- Pick the position ablation (tail vs question punctuation) as part of the design, not as a
  follow-up.

## What we learned

- Metamodel reads are hallucination-heavy and the FVE is a weak filter; the top-FVE bullet at one
  position is a better readout than the whole block, and a fixed rule beats the investigator.
- A placebo block is not a null: it changes what the investigator does, by ±10pp, depending on what
  the block looks like. Compare against the right thing.
- "Hard for the text baseline" filters are colliders. CHIVE's counterfactuals, WSB's anti-text-
  inversion subset and our own gate all had this shape.
- Infrastructure: one-engine vLLM reader for O-lens / NLA / J++ at ~19 reads/s per H200
  ([olens-one-engine-reader.md](olens-one-engine-reader.md)); batch reads to HF; static viewers over
  the HF dataset (expview, the env viewer, Docent uploads); every run through `scripts/runs.py`.
- TODO (Robbie, Kylie): research-skill reflections in your own words.

## Pre-sprint work

- Robbie: an oracle lens for Qwen3.6-27B trained from scratch during the exploration phase
  ([repo](https://github.com/robbiethompson18/oracle-lens-reconstruction),
  [HF](https://huggingface.co/collections/robbiethompson2018/nn-mats-sprint-oracle-lens-on-qwen36-27b),
  [dashboard](https://jlens-dashboard.pages.dev/)) and notes on the two O-lens recipes.
- Kylie: TODO.
- Borrowed: Agam's detective-joracle harness and the space lens + AR checkpoint; Camila's NLA;
  Aditya Singh's agent-interp-envs; WorkspaceBench's banks and judges (vendored, read-only,
  re-implemented).

## Cost

TODO (Robbie): total spend. Rates in [cost-estimates.md](cost-estimates.md): $5.29 per H200-hour,
~19 O-lens reads/s per H200; a 3k-case MCQ set at 20 positions × 6 layers is ~$28 of GPU per lens
and ~$6.50 of Haiku per O-lens draw; the 26-env cfbench sweep was $162 of Flash; the 7-model chart
$69.

## Where everything is

- Docs: `CLAUDE.md` indexes the ACTIVE docs, [INDEX.md](INDEX.md) the rest by tag.
- Evals: `cfbench/evals/<name>/` (eval.py has the commands), results on the env viewer's eval pages
  (`https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/#eval=<name>`).
- Reads and runs: `runs/` in the HF dataset; every transcript in Docent (`cfbench` collection) and
  expview (`.../expview/`).
- Charts: `graphs/` (`python -m graphs.<module>`), slide figures `slides/talk/figures.py`.
- Probes: `environments/hint_mcq/probe.py`, `probe_unselected.py`;
  `cfbench/evals/essay_grading_cf_inline/probe.py`.
