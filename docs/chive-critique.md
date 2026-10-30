# CHIVE critique: what is wrong with the dataset, with numbers

**Status (2026-10-30):** REFERENCE. Every number reproduces with `uv run python -m chive_critique`
([chive_critique/CLAUDE.md](../chive_critique/CLAUDE.md) says where the data lives); the §3 count on
Adam's data is a sample and is marked as such.

CHIVE ([Karvonen et al., 2026](https://arxiv.org/abs/2608.16747)) builds counterfactual explanations
of in-the-wild model behaviours and turns them into true/false claims that a predictor is scored on.
We spent two weeks trying to make an O-lens investigator beat the transcript on it and ended up
dissecting the dataset instead. None of the complaints below affects a majority of datapoints. Here
they are nonetheless, each with the count on Adam Karvonen's own release
([adamkarvonen/chive-data](https://huggingface.co/datasets/adamkarvonen/chive-data), the three
main-body WildChat eval runs: Gemma 3 27B, Qwen3-8B, Qwen3.5-397B, 5,172 investigations and 74,546
counterfactual experiments) and, where we have it, on Agam's 300-incident Qwen3.6-27B run
([agu18dec/chive-forensics](https://huggingface.co/datasets/agu18dec/chive-forensics)), which is
what we started from.

## Summary

1. **40% of the data is off-policy**: the prompt ends a WildChat conversation whose earlier
   assistant turns were written by ChatGPT, and only the last completion is the target model. Some
   "weird behaviour" is then not weird, e.g. the model correcting the previous model's advice
   ([000054](https://chive-cases.pages.dev/?case=prompt_000054)). On Adam's data no activation tool
   does better on that slice; on Agam's O-lens run the whole uplift lives there.
2. **Claims are labelled by a raw move in percentage points** (≥ 50pp true, ≤ 15pp false, in between
   discarded) rather than anything rate-aware, so 20% → 71% is a validated finding, 30% → 0% is
   discarded and 10% → 0% is "inert". On 8,623 experiments (12%, spread over 32% of investigations)
   the base rate was between 10% and 50% and the edit took it to 0/30, and the rule never shows the
   predictor any of them. In 18% of investigations there is no positive experiment at all, so the
   investigation enters the dataset as negatives only, even though it holds a 10%+ → 0 edit. The
   mirror case (edit pushes the rate to 100%) is six times rarer.
3. **The ground truth contradicts itself** in 12% of Agam's incidents: a ruled-out "inert" edit
   removes the very mechanism a supported cause asserts, because the inert list is assembled by the
   mechanical 15pp rule and never checked against the investigator's prose. 25% once you count
   "causes" that are themselves negations.
4. **Most of the behaviours are the model being wrong**: Adam's own judge flags 58–86% of
   investigated behaviours as mistakes (per target model), our hand labels 53% of Agam's. A judgment
   call, not a defect, but it shapes what the benchmark measures.
5. **Seven LLM roles sit between the behaviour and the label**; the contradictions in §3 enter at
   one join between them.

## 1. Off-policy prefixes: 40%

An incident is a WildChat (91% of Adam's pool; the rest SystemChat2, Hermes, ToolACE) prompt whose
earlier assistant turns were written by whatever model the source dataset logged; only the final
completion is the target model. Counting prompts with at least one such prior assistant turn
(`python -m chive_critique adam`, §1):

| Set (Adam's data)                          | ≥ 1 prior assistant turn |
| ------------------------------------------ | ------------------------ |
| the 4,714-prompt pool every eval run uses  | 1,880 (40%)              |
| Gemma 3 27B: investigations / eval claims  | 36% / 38%                |
| Qwen3-8B: investigations / eval claims     | 37% / 37%                |
| Qwen3.5-397B: investigations / eval claims | 39% / 40%                |

Screening does not enrich for it: multi-turn prompts yield an investigation slightly less often than
single-turn ones (33–34% vs 35–41%). Agam's run: 39% of 300, 41% of his 204 held-out. The
single-turn remainder is still off-distribution in the weaker sense that the user turns are ChatGPT
users' prompts, but the target model wrote every assistant token in those.

**Do tools do better on the off-policy slice?** On Adam's arms (Opus 4.8 answerer, the paper's
quality slice, paired per claim) no: every activation tool is within noise of transcript-only on
both slices, mostly negative.

| Quality slice        | n   | transcript-only | AO − plain      | NLA − plain     | SAE − plain     |
| -------------------- | --- | --------------- | --------------- | --------------- | --------------- |
| Gemma, off-policy    | 494 | 0.709           | −0.4pp (SE 1.4) | −1.4pp (SE 1.6) | −0.2pp (SE 1.5) |
| Gemma, on-policy     | 800 | 0.754           | −1.2pp (SE 1.2) | −2.9pp (SE 1.3) | −2.8pp (SE 1.1) |
| Qwen3-8B, off-policy | 562 | 0.763           | −0.9pp (SE 1.4) | −3.2pp (SE 1.4) | −1.8pp (SE 1.3) |
| Qwen3-8B, on-policy  | 935 | 0.737           | −0.5pp (SE 1.1) | +0.5pp (SE 1.2) | +0.2pp (SE 1.1) |

In Agam's harness (free-form explanation judged 1–10 against the causes, 5 draws per arm) the
picture differs: O-lens − transcript is +9.5pp (SE 3.7) on the 84 off-policy incidents and −1.5pp
(SE 2.4) on the 120 single-turn ones, so his whole +3.0pp (SE 2.1) lives in the off-policy slice. A
plausible reading is that when a ChatGPT turn is in the prefix the "cause" is often "the earlier
turn said X" and anything that points the agent at the prefix helps. That is a property of his
harness until replicated; Adam's claim format doesn't show it.

## 2. The 50pp / 15pp rule

CHIVE labels a claim true if the edit moved the behaviour rate ≥ 50pp and false if ≤ 15pp (paper
§2.2, Appendix B.3); the released claim banks hold only those two kinds, nothing in between. Over
every experiment in Adam's three runs (`python -m chive_critique adam`, §2):

| Run          | Investigations | Experiments | 15–50pp (no claim) | Baseline 10–50% → 0/30 | Investigations with one | … and no ≥ 50pp experiment | Baseline 50–90% → 30/30 |
| ------------ | -------------- | ----------- | ------------------ | ---------------------- | ----------------------- | -------------------------- | ----------------------- |
| Gemma 3 27B  | 1,820          | 26,257      | 9,565 (36%)        | 2,933 (11.2%)          | 524 (29%)               | 284 (16%)                  | 627 (2.4%)              |
| Qwen3-8B     | 1,736          | 24,940      | 10,620 (43%)       | 3,164 (12.7%)          | 639 (37%)               | 368 (21%)                  | 480 (1.9%)              |
| Qwen3.5-397B | 1,616          | 23,349      | 9,161 (39%)        | 2,526 (10.8%)          | 485 (30%)               | 262 (16%)                  | 524 (2.2%)              |
| **Total**    | **5,172**      | **74,546**  | **29,346 (39%)**   | **8,623 (11.6%)**      | **1,648 (32%)**         | **914 (18%)**              | **1,631 (2.2%)**        |

- A removal edit can never be a positive claim when the baseline is under 50%, however completely it
  kills the behaviour. The 8,623 are the clean kills the predictor is never asked about. The mirror
  (an edit that saturates the behaviour from a baseline over 50%) is 1,631 experiments in 797
  investigations (15%), much rarer because CHIVE's edits are mostly removals.
- In 914 investigations (18%) there is no ≥ 50pp experiment anywhere, so the investigation
  contributes negatives only; 576 of them are in the claim banks that way, 338 never made the bank.
- **What the usable set would gain.** Of the 3,946 verified investigations in the three claim banks,
  3,344 have a true claim. Counting the stuck kills as positives takes that to 3,920 (+17%) and true
  claims from 6,276 to about 7,400 (+18% at up to two per gained investigation; Gemma +15%, Qwen3-8B
  +24%, Qwen3.5-397B +16%). The balanced eval sets would grow by about the same fraction after the
  concreteness filter.
- Agam's run, same rule: 890 of 1,944 experiments (46%) in the 15–50pp band, 56% of them
  Fisher-significant at p < 0.01; 207 stuck kills (11%) in 86 incidents (29%); 61 incidents (20%)
  with one and no ≥ 50pp experiment.

**What would be better.** At n = 30, 10% → 0% is 3/30 vs 0/30 and Fisher's exact test gives p =
0.24: the data cannot tell it from 10% → 5%, so no threshold rescues it; only more samples do (at n
= 100 it is p = 0.002 while 10% → 5% stays at p = 0.28). The fix is two-part: a headroom-normalised
effect (fraction of the baseline removed, or of the remaining headroom filled; 10% → 0% and 30% → 0%
score 1.0, 40% → 80% scores 0.67, 10% → 5% scores 0.5, 000348's 40% → 27% scores 0.33) gated by
Fisher's exact test, and adaptive sampling when the baseline is under about 20%. "Inert" should mean
the interval excludes a large effect, not "not significant"; at n = 30 the 15pp negatives are weaker
than the positives under any rule.

## 3. Contradictory ground truth: one investigation in eight

Each incident's ground truth lists 1–3 supported causes (the investigator's prose, first person,
with evidence) and 0–9 ruled-out items. The judge is told a hypothesis matching any cause is correct
and one matching any ruled-out item is NOT a match "however plausible". Ruled-out items of kind
`inert` are appended mechanically: any experiment whose edit moved the rate ≤ 15pp. Nobody checks
them against the causes. [000348](https://chive-cases.pages.dev/?case=prompt_000348) is the
template: cause T2 "the well-being framing pulls me across the persona boundary"; inert X2 "the user
message was rewritten to remove mentions of stress and difficulty coping", 40% → 27%. A predictor
who names the cause matches T2 and X2 at once.

On Agam's run we labelled all 428 inert items against their incident's causes (four Claude Sonnet
subagents in Claude Code graded a chunk each; the main session re-read every flag and every inert
item whose rate is quoted in a cause's evidence, and set the final label; sub-grader agreement was
poor, 4 to 24 flags per ~100-item chunk, so the item count is soft by a few):

| Label             | Items | Meaning                                                                                                                |
| ----------------- | ----- | ---------------------------------------------------------------------------------------------------------------------- |
| CONTRADICTS       | 42    | the inert edit removes or neutralises a mechanism a supported cause asserts                                            |
| CONFIRMS_NEGATION | 58    | the "cause" is itself a negation ("I am not computing the weekday from a year"); a ruled-out factor written as a cause |
| UNRELATED         | 319   | a genuine alternative hypothesis, tested and ruled out                                                                 |
| DEGENERATE        | 9     | not a counterfactual: five "the two conversations are identical", four where the question no longer applies            |

- **35/300 incidents (12%) have a CONTRADICTS item; 27/204 (13%) of the held-out.** 45/300 (15%)
  have a negation-phrased cause; the union is 75/300 (25%).
- Of the 42 contradicting edits, 15 left the rate at or above baseline (removing the "cause" did
  nothing), and only 4 are the threshold artifact of §2 (e.g. 17% → 3% under 15pp). More examples:
  000713 (cause: the bracket ban suppresses commas; inert: the ban replaced by "use normal
  punctuation including commas", 67% → 73%), 003676 (cause: "Collective" triggers hive imagery;
  inert: renamed to "Republic", 60% → 70%), 003338 (cause: the no-other-text constraint forces a
  fabricated value; inert: permission to write a plain-text notice, 50% → 50%).
- On the held-out set, O-lens − transcript is −5.9pp (SE 5.6) on the 27 incidents with a
  contradiction vs +4.4pp (SE 2.3) on the other 177. Noisy, but the sign is what a noisy label
  should do.

**On Adam's data, a sample.** We drew 50 investigations per model (seed 0) among those with at least
one inert claim in the claim bank (85–90% of verified investigations) and labelled every inert claim
against the investigation's own `structured_answer` the same way: three Sonnet subagents, one per
model, then the main session re-read all 35 flags and kept 20 (`adam_inert_labels.jsonl` in the run
dir has both labels and the reason for each decision).

| Run          | Inert claims | CONTRADICTS | CONFIRMS_NEGATION | UNRELATED | Investigations with a contradiction |
| ------------ | ------------ | ----------- | ----------------- | --------- | ----------------------------------- |
| Gemma 3 27B  | 88           | 6           | 53                | 29        | 6/50 = 12%                          |
| Qwen3-8B     | 92           | 10          | 48                | 33        | 10/50 = 20%                         |
| Qwen3.5-397B | 83           | 4           | 54                | 25        | 4/50 = 8%                           |
| **Total**    | **263**      | **20**      | **155**           | **87**    | **20/150 = 13%**                    |

So the rate is the same on Adam's data as on Agam's: about one investigation in eight has a false
claim that negates one of its own stated causes (a binomial 95% interval on 20/150 is 8–20%). The
commonest form is a mismatch between the write-up's own evidence and the claim bank's number for the
same edit: the write-up says swapping the cheese festival for a kite festival drops cheese to 10%
and the bank records 16/30 (003654); the write-up says the brown-dwarf first turn alone gives 0% and
the bank records 25/30 (001871); the write-up says `nn.Conv2d` nearly eliminates the error and the
bank records 26/30 (000093). Also notable: 59% of Adam's inert claims are CONFIRMS_NEGATION, i.e.
the investigator's own negative controls. The false half of the claim bank is mostly edits the
investigator predicted would do nothing, which is why a transcript-reading predictor that thinks
like the investigator finds them easy.

## 4. Most of the behaviours are the model being wrong

Adam's own concreteness judge flags each investigated behaviour as a mistake or not
(`mechanism_concreteness.jsonl`, `is_mistake`; `python -m chive_critique adam`, §4):

| Run          | Labelled investigations | `is_mistake` | Top categories                                                                    |
| ------------ | ----------------------- | ------------ | --------------------------------------------------------------------------------- |
| Gemma 3 27B  | 1,407                   | 976 (69%)    | known_then_fixable, surprising_influence, entity_or_register_association          |
| Qwen3-8B     | 1,317                   | 1,137 (86%)  | known_then_fixable, surprising_influence, misinterpretation                       |
| Qwen3.5-397B | 1,265                   | 740 (58%)    | surprising_influence, statistical_or_lexical_artifact, coherent_quirk_non_mistake |

Our own single-rater labels of Agam's 204 held-out behaviours (`viewer/behavior_kind.tsv`) say the
same thing more coarsely: **wrong** (factual or logic slip, self-contradiction, misread context,
ignored instruction) 108 = 53%; **content** (coherent but unprompted content or style choice, e.g.
[000106](https://chive-cases.pages.dev/?case=prompt_000106)'s tired character) 72 = 35%; **policy**
(refusal or persona scope) 24 = 12%. O-lens uplift by kind in Agam's run: wrong +1.7pp (SE 2.7),
content +6.9pp (SE 4.0), policy −2.5pp (SE 5.3), none significant. This is a judgment call rather
than a defect, but it says what the benchmark measures: mostly "why did a small model slip here",
which is not obviously what a lens for unverbalised goals is for.

## 5. Layers of LLM between the behaviour and the label

From the paper's pipeline (Appendix B) plus Agam's harness: a screening judge finds the behaviour
and writes the classifier question; an Opus investigator runs the counterfactuals and writes the
first-person causes; five verification-judge samples score the investigation 1–10 (kept at ≥ 8);
Opus 4.8 picks two informative and two refuting experiments; Sonnet 4.6, blind to outcomes,
describes each edit and flags confounds; three filter judges score mechanism concreteness,
reproducibility and single-factor edits; then Agam's harness hands the investigator's causes plus
the ruled-out list to a 1–10 judge of the predictor's free-form answer. The §3 contradictions enter
where the investigator's prose and the claim bank's mechanical 15pp rule are glued into one judge
reference. Handing the final judge the experiments and their rates, and dropping the summarised
causes, removes that join.
