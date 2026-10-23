---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-size: 26px; padding: 50px 64px; }
  h1 { font-size: 40px; }
  h2 { font-size: 34px; }
  table { font-size: 21px; }
  code { font-size: 0.85em; }
  .small { font-size: 19px; color: #52514e; }
  img[alt~="center"] { display: block; margin: 0 auto; }
---

<!--
Running deck for meetings with Neel. Newest results go at the front (after the title), older
sections stay below. Edit freely: it's plain Markdown, `---` separates slides.
Preview: the Marp VS Code extension. Export: npx @marp-team/marp-cli slides/neel.md --pdf
Figures: uv run --script slides/plots.py (needs data/ pulled from HF).
-->

# O-lens uplift for a zero-counterfactual investigator

Robbie & Kylie · MATS sprint with Neel Nanda

<span class="small">Running deck, newest first. Last updated 2026-10-23.</span>

---

## 2026-10-23: the O-lens agent beats transcript-only, pre-registered

- Env: `hidden_goal.grade`. Qwen3.6-27B grades a mediocre essay; one sentence about who wrote it
  moves the harsh-grade rate 0/30 → 30/30.
- Predictor: Gemini 3.8 Flash, agentic (≤ 20 tool calls), sees **only the base arm**.
- 38 counterfactual cases × 5 draws × 3 conditions = 570 investigations.

|               | TS agent | O-lens agent | O-lens placebo |
| ------------- | -------- | ------------ | -------------- |
| skill, all 38 | 0.736    | **0.812**    | 0.745          |

O-lens − TS **+0.076** [+0.018, +0.167] · O-lens − placebo **+0.067** [+0.017, +0.149]
<span class="small">95% case bootstrap. Bar set before launch: both CIs exclude 0. We predicted a
null.</span>

---

## The setting

- **Base arm**: 30 rollouts of Qwen on one prompt. **Counterfactual**: a one-sentence edit to that
  prompt.
- The investigator gets the base rollouts, the exact edit, and tools; it predicts P(behaviour) on
  the edited prompt.
- **Zero counterfactual**: no rollouts and no lens reads on the edited prompt. All lens reads are of
  the base arm.
- **cfbench skill**: binomial log-likelihood, 0 = "the edit does nothing", 1 = knowing the true
  rate.
- Conditions:
  - **TS agent**: `search_transcripts`, `read_transcript`
  - **O-lens agent**: + `search_olens`, `olens_at` (agu18dec `olens_s3d_space_rl600`, L32/40/48,
    every stop token)
  - **Placebo**: same tools, reads from an unrelated env (donation_bet) swapped in

---

## The env: who wrote the essay

![w:1040 center](img/rates.png)

<span class="small">Qwen grades the user's own essay harder (16/30), against the sycophancy prior
every predictor uses. Thinking off, so the reply is the only text.</span>

---

## What the lens says that we _can't_ use

On the **edited** prompt, at "… wrote this essay.", the lens states the user goal Qwen infers:

| edit             | harsh | reply mentions it | lens on the edited prompt                                                     |
| ---------------- | ----- | ----------------- | ----------------------------------------------------------------------------- |
| ChatGPT wrote it | 23/30 | 0/30              | "overly polished, generic, lacks personal voice. Please critique it."         |
| My rival         | 30/30 | "rival" 0/30      | "I want to beat them."                                                        |
| I wrote it       | 16/30 | never says why    | "critique it like a professional editor" (base: "evaluate it as a professor") |
| My son           | 0/30  | —                 | "I am proud of his effort"                                                    |

But every attribution case shares the same base prompt, so in the zero-counterfactual setting the
investigator gets **identical** prompt reads for all of them.

---

## Where the uplift comes from

![w:620 center](img/skill.png)

<span class="small">Held out alone: +0.014 vs TS [−0.008, +0.036], +0.014 vs placebo [+0.002,
+0.029]. Most held-out cases are easy for everyone.</span>

---

## A few cases carry it

![w:640 center](img/case_gains.png)

<span class="small">O-lens wins 14 cases, loses 11. Post hoc: without the top 2 cases, +0.030
[+0.004, +0.066].</span>

---

## Mechanism: the lens sees a stricter score than Qwen writes

At the base arm's `Score:` token, reads often name a harsher grade than the reply:

> Reply: **Score: 5/10** L40 [FVE 0.52]: "6/10 (or 2/10 on a strict academic scale) Reason: It is a
> basic, unstructured paragraph…"

- The reply never says this. The O-lens agent finds it with `search_olens "Score: "`.
- It predicts that edits removing the "student" framing, or adding a scholarship bar, unlock the
  strict scale.
- all-stakes → all (true 27/30): O-lens 3 of 5 draws ≥ 0.25, placebo 0 of 5.
- It's about grading **headroom**, not the edit. "Strict scale" reads appear in 9–12/30 base
  replies, fewer in lenient arms (son 2, ESL 0), more in harsh ones (famous, coworker 23).

<span class="small">Traces:
[all-stakes → all](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/e059e4d0-135a-4ff9-a987-d741b31df616)
·
[base → stakes](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/cf8a44a5-4221-4169-aae2-b25b7c52e0f8)</span>

---

## Dead ends and caveats

- **Pasting reads into the prompt** (no tools; PR #116, closed): never beat its placebo. The one
  "+0.042" had no placebo run.
- **Reading the lens on the edited prompt** (forward pass only): outside the setting. Question for
  Neel: is a no-sampling forward pass on the counterfactual allowed?
- **Concentration**: three cases (two cue, one attribution) carry most of the gain.
- **Not yet checked**: whether "strict scale" bullets come from Qwen's activations or from the
  lens's prior for any grading text.
- One env, one predictor. Placebo controls for "more text", not for the lens's priors.

---

## Next

1. Lens-prior check, free (existing reads): do "strict scale" bullets show up on replies that gave a
   _high_ score, and on non-grading text?
2. A second env with a hidden numeric threshold, to see whether "headroom" reads generalize beyond
   grading.
3. Upstream the cfbench retry/`--skip` fix to #117.
