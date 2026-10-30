# CHIVE dataset audit: off-policy prefixes, thresholds, contradictory ground truth

**Status (2026-10-30):** REFERENCE. Numbers over all 300 incidents of
[agu18dec/chive-forensics](https://huggingface.co/datasets/agu18dec/chive-forensics) and Agam's 204
held-out incidents; `scripts/chive_audit.py` reprints every table.

Robbie's Oct 25 progress report (sprint doc, "Issue with CHIVE Data / Grader") listed five
complaints about CHIVE from reading cases. This doc puts numbers on each. Where a number disagrees
with the report, the number wins. Reproduce with `uv run python scripts/chive_audit.py`; the hand
labels it reads are in the HF dataset at `runs/chive_audit/2026-10-30_inert-contradictions/`.

## 1. "More than half the dataset is off-policy" → 40% of the prompt pool, 36–40% of claims

An incident is a WildChat (or SystemChat2 / Hermes / ToolACE) prompt whose earlier assistant turns
were written by whatever model the source dataset logged (ChatGPT for WildChat); only the final
completion is the target model. Counting prompts with at least one such prior assistant turn, on
Adam's release (`scripts/chive_adam_eliminations.py offpolicy`):

| Set (Adam's data)                          | ≥1 prior assistant turn |
| ------------------------------------------ | ----------------------- |
| the 4,714-prompt pool every eval run uses  | 1,880 (40%)             |
| Gemma 3 27B: investigations / eval claims  | 36% / 38%               |
| Qwen3-8B: investigations / eval claims     | 37% / 37%               |
| Qwen3.5-397B: investigations / eval claims | 39% / 40%               |

So "more than half" is wrong; it is two fifths, and the screening stage does not enrich for it
(multi-turn prompts yield an investigation slightly less often than single-turn ones, 33–34% vs
35–41%). The single-turn remainder is still off-distribution in the weaker sense that the user turns
are ChatGPT users' prompts, but the target model wrote every assistant token in those.

**Do the tools do better on the off-policy slice?** On Adam's own arms (Opus 4.8 answerer, quality
slice, paired per claim) no: every activation tool is within noise of transcript-only on both
slices, and the sign is mostly negative.

| Quality slice        | n   | transcript-only | AO − plain      | NLA − plain     | SAE − plain     |
| -------------------- | --- | --------------- | --------------- | --------------- | --------------- |
| Gemma, off-policy    | 494 | 0.709           | −0.4pp (SE 1.4) | −1.4pp (SE 1.6) | −0.2pp (SE 1.5) |
| Gemma, on-policy     | 800 | 0.754           | −1.3pp (SE 1.2) | −2.9pp (SE 1.3) | −2.8pp (SE 1.1) |
| Qwen3-8B, off-policy | 562 | 0.763           | −0.9pp (SE 1.4) | −3.2pp (SE 1.4) | −1.8pp (SE 1.3) |
| Qwen3-8B, on-policy  | 935 | 0.737           | −0.5pp (SE 1.1) | +0.5pp (SE 1.2) | +0.2pp (SE 1.1) |

In Agam's 204-incident Qwen3.6-27B harness (free-form explanation judged 1–10, 5 draws per arm) the
picture differs: O-lens − transcript is +9.5pp (SE 3.7) on the 84 off-policy incidents and −1.5pp
(SE 2.4) on the 120 single-turn ones, i.e. his whole +3pp lives in the off-policy slice. A plausible
reading is that when a ChatGPT turn is in the prefix the "cause" is often "the earlier turn said X"
(e.g. [000054](https://chive-cases.pages.dev/?case=prompt_000054), where Qwen corrects the prior
model's advice) and anything that points the agent at the prefix helps. Adam's claim format doesn't
show that effect, so treat it as a property of Agam's harness until replicated.

## 2. "About half is Qwen being wrong without a great reason" → 53%

Our single-rater hand labels of the 204 held-out behaviours (`viewer/behavior_kind.tsv`,
2026-10-19): **wrong** (factual or logic slip, self-contradiction, misread context, ignored
instruction) 108 = 53%; **content** (coherent but unprompted content or style choice, e.g.
[000106](https://chive-cases.pages.dev/?case=prompt_000106)'s tired character) 72 = 35%; **policy**
(refusal or persona scope) 24 = 12%. O-lens uplift by kind: wrong +1.7pp (SE 2.7), content +6.9pp
(SE 4.0), policy −2.5pp (SE 5.3). None is significant on its own.

## 3. Contradictory ground truth → 12% of incidents (13% of the held-out), 25% counting negated "causes"

Each incident's ground truth has 1–3 `targets` (the investigator's causes, first person, with prose
evidence) and 0–9 `ruled_out` items. Ruled-out items come in three kinds: 360 `refuted` (the
investigator's own alternative hypotheses), 344 `inert` from the `claim_bank` (experiments whose
edit moved the rate ≤ 15pp) and 84 `inert` from `experiment_le_15pp` (same rule, different source).
The judge is told a hypothesis matching any ruled-out item is NOT a match.

The contradiction: a target asserts mechanism M, and an inert item is the edit that removes M and
found it did nothing. [000348](https://chive-cases.pages.dev/?case=prompt_000348) is the template:
T2 "the well-being framing pulls me across the persona boundary"; X2 "the user message was rewritten
to remove mentions of stress and difficulty coping", 40% → 27%, inert. A predictor who says
"well-being framing" matches T2 and X2 at once.

We labelled all 428 inert items against their incident's targets. Four Claude Sonnet subagents
(Claude Code, no API) graded one chunk each; the main session re-read every flagged item and every
inert item whose rate is quoted in a target's evidence (109 more), and set the final label.
Sub-grader agreement was poor (4 to 24 flags per ~100-item chunk), so the item-level count is soft
by a few; the incident-level rate below moved from 13% to 12% under adjudication.

| Label             | Items | Meaning                                                                                                                                 |
| ----------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------- |
| CONTRADICTS       | 42    | the inert edit removes or neutralises a mechanism a supported target asserts                                                            |
| CONFIRMS_NEGATION | 58    | the "target" is itself a negation ("I am not computing the weekday from a year"); consistent, but a ruled-out factor written as a cause |
| UNRELATED         | 319   | a genuine alternative hypothesis that was tested and ruled out                                                                          |
| DEGENERATE        | 9     | not a counterfactual: five "the two conversations are identical" edits, four where the behaviour question no longer applies             |

- **35/300 incidents (12%) have at least one CONTRADICTS item; 27/204 (13%) of the held-out.**
- 45/300 (15%) have a negation-phrased target; the union is 75/300 (25%).
- Of the 42 CONTRADICTS edits, 15 left the rate at or above baseline (removing the "cause" did
  nothing), and only 4 are threshold artifacts where the edit cut the behaviour by more than half
  but under 15pp (e.g. 002332: 17% → 3%). Most are real inconsistencies in the write-up, not
  rounding.
- On the held-out set, O-lens − transcript is −5.9pp (SE 5.6) on the 27 incidents with a
  contradiction vs +4.4pp (SE 2.3) on the other 177. Noisy, but the direction is what you would
  expect if a noisy label hurts the arm with more information.

Adam's bar (Slack, 2026-10-30) was "issues that affect a large majority of datapoints, not 3%". This
one is 12–25%, not a majority; the thresholds below are.

## 4. The 50pp / 15pp thresholds → 46% of experiments are in neither band

CHIVE labels a counterfactual claim true if the edit moved the behaviour rate ≥ 50pp and false if ≤
15pp (paper §2.2, Appendix B.3); the claim bank's `inert` items are the ≤ 15pp side. Over the 1,944
experiments in the dataset:

| Band              | Experiments | Note                                                     |
| ----------------- | ----------- | -------------------------------------------------------- |
| true, ≥ 50pp      | 631 (32%)   | 587 are decreases                                        |
| excluded, 15–50pp | 890 (46%)   | 501 (56%) are Fisher-significant vs baseline at p < 0.01 |
| false, ≤ 15pp     | 423 (22%)   | none is Fisher-significant at p < 0.05                   |

- 255 edits drive the behaviour to ≤ 1/30 from a baseline under 50%; 253 of them sit in the excluded
  band. A removal edit can never be a positive claim when the baseline is under 50%, and 103/300
  incidents (34%) have such a baseline. 76/300 (25%) have no ≥ 50pp experiment at all.
- The 15pp "inert" labels are statistically defensible at n = 30 (none rejects the null), but the
  investigator's prose treats the same 40% → 27% as directional support, which is how §3 arises.
- A headroom-normalised effect (fraction of the baseline removed, or of the headroom filled) with a
  Fisher or likelihood-ratio test would relabel the 253 full eliminations as positives and leave the
  ≤ 15pp items alone. `CLAUDE.md` already says to do this for our own envs.

### The same count on Adam's released data

The numbers above are Agam's 300-incident Qwen3.6-27B run. The same rule on Adam Karvonen's own
release ([adamkarvonen/chive-data](https://huggingface.co/datasets/adamkarvonen/chive-data), the
three main-body WildChat eval runs, every experiment in `investigations.jsonl`;
`scripts/chive_adam_eliminations.py`, extracted rates in the run dir under `adam_rates/`):

| Run            | Investigations | Experiments | 15–50pp (excluded) | Baseline 10–50% → 0/30 | Investigations with one | … and no ≥ 50pp experiment |
| -------------- | -------------- | ----------- | ------------------ | ---------------------- | ----------------------- | -------------------------- |
| Gemma 3 27B IT | 1,820          | 26,257      | 9,565 (36%)        | 2,933 (11.2%)          | 524 (29%)               | 284 (16%)                  |
| Qwen3-8B       | 1,736          | 24,940      | 10,620 (43%)       | 3,164 (12.7%)          | 639 (37%)               | 368 (21%)                  |
| Qwen3.5-397B   | 1,616          | 23,349      | 9,161 (39%)        | 2,526 (10.8%)          | 485 (30%)               | 262 (16%)                  |
| **Total**      | **5,172**      | **74,546**  | **29,346 (39%)**   | **8,623 (11.6%)**      | **1,648 (32%)**         | **914 (18%)**              |

So: on 8,623 experiments the behaviour had a baseline rate between 10% and 50% and the edit took it
to 0 of 30, and because of the ≥ 50pp rule none of them is a claim (the released claim banks hold
only ≥ 50pp "supported" and ≤ 15pp "inert" items, nothing in between). That is one experiment in
nine, in a third of investigations (1,648 / 5,172 = 32%); in 18% of investigations (914) it is the
only kind of decisive result there is, so those investigations contribute no positive claim at all.

**How much bigger the usable set would be.** Of the 3,946 investigations that passed verification
into the three claim banks, 3,344 have at least one true claim and 602 have only false ones; 576 of
those 602 hold a stuck elimination. Counting them as positives takes investigations with a true
claim from 3,344 to 3,920 (+17%), and true claims from 6,276 to about 7,400 (+18%, at up to two per
gained investigation; Gemma +15%, Qwen3-8B +24%, Qwen3.5-397B +16%). The balanced eval sets would
grow by about the same fraction after the concreteness filter (roughly +490 true claims on 2,415).
Dropping the band entirely (any Fisher-significant change counts) would add far more: 56% of the
excluded band is significant at p < 0.01 in Agam's run.

## 5. Layers of LLM between the behaviour and the label

From the paper's pipeline (Appendix B) plus Agam's harness: a screening judge finds the behaviour
and writes the classifier question; an Opus investigator runs the counterfactuals and writes the
first-person causes; five verification-judge samples score the investigation 1–10 (kept at ≥ 8);
Opus 4.8 picks two informative and two refuting experiments; Sonnet 4.6, blind to outcomes,
describes each edit and flags confounds; three filter judges score mechanism concreteness,
reproducibility and single-factor edits; then Agam's harness hands the investigator's causes plus
the ruled-out list to a 1–10 judge of the predictor's free-form answer. The §3 contradictions enter
at the join between two of these: the investigator's prose and the claim bank's mechanical 15pp rule
are glued into one `judge_reference`. Handing the final judge the experiments and rates themselves,
and dropping the summarised "causes", removes the join.
