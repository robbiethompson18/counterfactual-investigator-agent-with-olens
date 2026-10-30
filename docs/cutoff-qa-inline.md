# cutoff_qa_inline: predicting Qwen's stale answer from end-of-prompt O-lens, J++ and NLA reads

**Status (2026-10-29):** REFERENCE. 999 items, one investigator (Haiku 5.5), O-lens / J++ / NLA each
against its own placebo; the caveats below are open. The 200-item smoke further down is superseded
by the 1k run but kept for its mechanism notes.

**Question.** Can an investigator that never sees Qwen3.6-27B's reply predict the answer it gives to
a present-tense factual question ("Who is the CEO of Intel?"), and do O-lens reads of the prompt
help? The label is Qwen's modal answer from the bank's stability gate, not the truth: on
`stable_wrong` items that answer is stale (Pat Gelsinger). No hint, no counterfactual, no LLM judge.

**Setup.** `cfbench/evals/cutoff_qa_inline/` (its docstring has the commands).

- Items: the first 200 stable (≥ 7/8 samples agree), non-numeric items of bank v2
  ([docs/envs/cutoff_qa.md](envs/cutoff_qa.md)) by bank order: 62 `stable_right`, 138
  `stable_wrong`. Qwen env run `cutoff_qa/2026-10-29_smoke200-nohint` (n = 1, shared server,
  thinking off, T 1 / top-p 1): Qwen repeated its modal answer on 177 of 200.
- Reads: `olens_reads/2026-10-29_cutoff200-tail`: space O-lens + AR, one-engine reader, T 1, at the
  six end-of-prompt tokens (the "." ending the question, `<|im_end|>`, `\n`, `<|im_start|>`,
  `assistant`, `\n`), layers 32/40/48/52/56/60. 1x H200, ~8 min, ~$0.80.
- Investigator: Claude Haiku 5.5 at effort medium, one turn, no tools, one `submit_answer`
  (knowledge_cutoff's prompt, `cfbench/evals/knowledge_cutoff/inline.py`); the model is never named.
  Not Gemini 3.8 Flash: Flash shares Qwen's cutoff and holds the stale answer itself on a quarter of
  WorkspaceBench's items (docs/envs/knowledge_cutoff.md). 3 draws × 3 conditions × 200 items, $1.40.
- Conditions: `transcript_inline` (the question), `olens_inline` (plus the item's own reads with
  FVE), `olens_placebo_inline` (another item's reads at the same positions, presented as this
  item's).
- Scoring: exact = the submission's `norm` (the env's, the same form as `qwen_modal`) equals the
  modal answer; loose adds whole-word containment either way. Paired sign test over items on the
  per-item mean over draws.

**1k result** (2026-10-29). 999 items: the 200 below (`smoke200-nohint`) plus 799 seeded-random
stable non-numeric bank-v3 items (`cutoff_qa/2026-10-29_new800-nohint`; the 800th hung on the Qwen
server and was dropped): 339 stale, 660 not. Reads at the same positions and layers as the smoke:
`olens_reads/2026-10-29_cutoff{200,800}-tail`, `jpp_reads/2026-10-29_cutoff{200,800}-all`,
`nla_reads/2026-10-29_cutoff{200,800}-tail` (all-layer NLA, greedy, critic FVE; its placebo draws
only from 200 MMLU-Pro prompts' reads, PR #405). Inputs
`cfbench_evals/2026-10-29_cutoff-qa-inline-inputs-1k` (`-1k-nla` for the NLA conditions, same cases
and other reads). One run per condition,
`cfbench/2026-10-29_cutoff-qa-1k-{transcript,olens,olens-placebo,jlens,jlens-placebo,nla,nla-placebo-mmlu}`,
Haiku 5.5 medium, 3 draws: 20,979 predictions, ~$20 of Haiku (OpenRouter-reported).
`…-1k-nla-placebo` is superseded: its placebo pool mixed in other cutoff items' reads (86 / 84 / 65
/ 94 there). Docent, one filtered view per run:
[transcript](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run?filter=97253028-1431-411b-a65a-4363587ffc84),
[O-lens](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run?filter=f84d47a6-0273-4864-a3e1-61180360be58),
[O-lens placebo](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run?filter=85c438e4-605d-4af1-9f35-a166135f6323),
[J++](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run?filter=c8c29354-a7c0-4f8e-9b31-300532ba602a),
[J++ placebo](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run?filter=eb576b08-acdf-487b-877f-3eac2f978ba9),
[NLA](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run?filter=66489f62-7303-48fb-9a99-795aad0891d6),
[NLA placebo](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run?filter=edfed094-ab42-40a6-ac9f-b12e40bed607),
superseded
[NLA placebo, old pool](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run?filter=cb531331-7c54-445d-adc8-0d190aa4fabd).

![transcript-only vs O-lens, J++ and NLA and their placebos, all / stale / not stale](img/cutoff-qa-inline.png)

![paired differences between conditions with 95% CIs, grouped by question](img/cutoff-qa-inline-diff.png)

(`uv run python -m graphs.cutoff_qa_inline` makes both; exact match, 95% bootstrap CIs over items,
the second paired by item. This is the one cutoff chart: every lens goes on it.)

| exact match (%)      | all     | stale   | not stale |
| -------------------- | ------- | ------- | --------- |
| transcript only      | 78      | 47      | 94        |
| O-lens / its placebo | 85 / 77 | 72 / 55 | 92 / 89   |
| J++ / its placebo    | 83 / 81 | 67 / 59 | 91 / 93   |
| NLA / its placebo    | 86 / 83 | 73 / 60 | 92 / 95   |

- **Only the O-lens clearly beats its own placebo by much:** +7.6pp [+5.7, +9.5]. NLA beats its
  MMLU-Pro placebo by +2.8pp [+1.0, +4.6], J++ its placebo by +1.7pp [+0.1, +3.3].
- **The J++ and NLA placebos aren't neutral:** +3.2pp [+1.7, +4.7] and +4.8pp [+3.5, +6.2] over
  transcript-only (the O-lens placebo −0.7pp). Haiku reads off-topic reads as noise, then reasons
  about the model's training cutoff and gives the stale answer more often; the gain is all on stale
  items (NLA placebo 60 vs 47). Against transcript-only NLA looks as good as the O-lens (+7.6 vs
  +6.9pp), but most of that is the placebo effect.
- O-lens − J++ +2.0pp [+0.3, +3.7]; O-lens − NLA −0.7pp [−2.3, +0.8]; J++ − NLA −2.7pp [−4.5, −1.0].

**Read-position ablation** (2026-10-30, Kylie: does the gain need the assistant-turn tokens?). The
1k run with only some reads shown: the first one or two of the six tail positions (`pos1` = the "."
ending the question, `pos2` = "." and `<|im_end|>`), or every punctuation token inside the question
(`qpunct`, 4.1 per prompt: commas, "?", "."), alone or with `<|im_end|>` (`qpunct_imend`). The
question's other punctuation was never read before: `olens_reads/2026-10-30_cutoff-fill-kylie` (the
requests of `scripts/fill_read_requests.py`, PR #432, for this set). The preamble names only the
positions shown. Runs `cfbench/2026-10-30_cutoff-qa-1k-olens-{pos1,pos2,qpunct,qpunct-imend}`
(conditions renamed after the run, e.g. `olens_qpunct_inline`), inputs
`cfbench_evals/2026-10-30_cutoff-qa-inline-inputs-1k-*`, 23,976 predictions, 0 invalid. Error rate
(all / stale / not stale), paired by item, 95% bootstrap CIs:

| reads                                 | O-lens            | its placebo        | O-lens − transcript-only | O-lens − placebo    |
| ------------------------------------- | ----------------- | ------------------ | ------------------------ | ------------------- |
| none (transcript-only)                | 21.9 / 53.3 / 5.8 |                    |                          |                     |
| "." only                              | 17.9 / 38.2 / 7.5 | 19.3 / 42.7 / 7.3  | −4.0pp [−5.7, −2.2]      | −1.4pp [−2.9, +0.2] |
| "." + `<\|im_end\|>`                  | 18.4 / 40.9 / 6.8 | 19.3 / 45.1 / 6.0  | −3.6pp [−5.2, −2.0]      | −0.9pp [−2.4, +0.4] |
| question punctuation                  | 15.2 / 28.5 / 8.3 | 20.6 / 44.4 / 8.4  | −6.7pp [−8.8, −4.7]      | −5.4pp [−7.3, −3.8] |
| question punctuation + `<\|im_end\|>` | 15.8 / 29.0 / 9.0 | 19.0 / 43.9 / 6.3  | −6.1pp [−8.3, −4.1]      | −3.2pp [−5.0, −1.6] |
| all six tail (through `assistant\n`)  | 15.0 / 27.8 / 8.4 | 22.6 / 44.7 / 11.2 | −6.9pp [−9.2, −4.8]      | −7.6pp [−9.5, −5.7] |

**On cutoff the question's own punctuation is as good as the six tail tokens** (15.2 vs 15.0, and
28.5 vs 27.8 on stale items), and beats its placebo by 5.4pp. The signal is at the "?" that ends the
question proper, before the boilerplate ("If you are not sure, give your best guess. Answer with
just the person name."): some read there names Qwen's modal answer on 659 of 986 items (190 of 339
stale), e.g. at "Who is the CEO of Intel?" L32: "The current CEO of Intel is **Pat Gelsinger**." The
commas (all in the boilerplate) name it on 40, the final "." on 316. The "." and `<|im_end|>` alone
are at their placebo. Every placebo beats transcript-only by 1–3pp: any reads push Haiku towards the
stale answer.

**Second 1k** (2026-10-30, Kylie). 1,000 more seeded stable, non-numeric bank-v4 questions not among
the 999 (326 stale), built from the bank by `cfbench/evals/cutoff_qa_inline/extend.py` (the label is
the bank's 8-sample modal answer; reads are of the prompt alone, so no Qwen episode). Reads
`olens_reads/2026-10-30_cutoff-new1k` (question punctuation + the six tail tokens), runs
`cfbench/2026-10-30_cutoff-qa-new1k-tail6` (transcript-only, six-tail O-lens + placebo) and
`..._cutoff-qa-new1k-olens-qpunct`, 15,000 predictions, $11, 0 invalid. Pooled with the first 999
(1,999 questions, 665 stale), error rate:

| condition            | all  | stale | − transcript-only, all | − its placebo, all  |
| -------------------- | ---- | ----- | ---------------------- | ------------------- |
| transcript-only      | 22.2 | 53.9  |                        |                     |
| question punctuation | 17.4 | 33.6  | −4.8pp [−6.3, −3.3]    | −4.4pp [−5.8, −3.2] |
| its placebo          | 21.8 | 47.0  |                        |                     |
| six tail tokens      | 17.2 | 32.2  | −5.0pp [−6.6, −3.5]    | −6.5pp [−8.0, −5.2] |
| its placebo          | 23.7 | 46.7  |                        |                     |

Question punctuation − six tail: +0.2pp [−0.9, +1.2] (stale +1.4 [−0.9, +3.5]); the same in each
half. The new 1k shows a smaller gain overall (O-lens − transcript −2.9pp vs −6.7pp on the
first 999) but the same on stale items (−15.7pp vs −24.8pp, CIs overlapping).

**200-item smoke** (`cfbench/2026-10-29_cutoff-qa-haiku-200`,
[Docent](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e)).

| condition            | exact | loose | stable_right | stable_wrong | vs transcript, win/loss | p      |
| -------------------- | ----- | ----- | ------------ | ------------ | ----------------------- | ------ |
| transcript_inline    | 66.7% | 69.0% | 96.2%        | 56.8%        |                         |        |
| olens_inline         | 81.2% | 82.5% | 90.3%        | 79.0%        | 53 / 12                 | <0.001 |
| olens_placebo_inline | 67.0% | 69.7% | 90.3%        | 60.4%        | 29 / 31                 | 0.90   |

The O-lens beats both transcript-only and the placebo by 14 pp, all of it on the stale items: on
`stable_wrong`, Haiku answers the current truth (Lip-Bu Tan) 124 of 414 draws transcript-only and 55
of 414 with the reads. On `stable_right`, where Haiku already knows the answer, the reads cost 6 pp:
8 of the 15 losses are format drift the lens induced ("macos sequoia" for "sequoia", "snapdragon 8
gen 3 for galaxy").

**Mechanism: the lens writes the answer at the newline after `<|im_end|>`.** Read-level recall (some
bullet contains Qwen's modal answer, loose match): 177 of 200 items (124/138 stable_wrong, 53/62
stable_right). By position: `\n` after `<|im_end|>` 176, `<|im_start|>` 125, `assistant` 115, "."
55, `<|im_end|>` 8. By layer: L32 125, L40 142, L48 152, L52 156, L56 156, L60 145. At "." the
bullets are instruction echoes ("Just the name") until the late layers, where they become "Answer:
Janet Yellen"; at the newline they are a sentence of Qwen's answer ("CEO of Intel is currently **Pat
Gelsinger**"). Haiku's accuracy with the reads is 82.5% on the 177 items whose reads name the answer
and 71% on the 23 whose reads don't (transcript-only 65.7% / 73.9%). Examples:
[Intel](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/06fe03c7-61ad-4dcd-90cc-9d0fa0722b84)
(transcript-only says Lip-Bu Tan 3/3, lens says Pat Gelsinger 3/3),
[Treasury Secretary](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/2ac35c0f-0d0a-4355-be93-f422bb7ab0c3)
(Scott Bessent → Janet Yellen), and a loss,
[Toyota CEO](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/1cc0ad9a-0530-41bd-afe8-27cb7dc4a804)
(Qwen says Koji Sato, the reads say "Akio Toyoda (current CEO) – As of 2024, Toyota's CEO is Koji
Sato", Haiku submits Akio Toyoda 3/3).

**J++ on the 200** (`jlens_inline`, `jlens_placebo_inline`, `jlens_inline_no60`; PR #393, runs
`cfbench/2026-10-29_cutoff-qa-haiku-200-jlens{,-placebo,-no60}`). The J++ lens's top 3 next tokens
at every token of the prompt, layers 32–60, ~4.5k tokens per prompt; placebo: the last n prompt
tokens of a hint_mcq transcript's J++ read. Exact match: J++ 78%, J++ placebo 73%, O-lens − J++
+2.8pp [−2.0, +7.8]. J++ − its placebo is +5.2pp [+1.0, +9.2], but the J++ placebo is itself +6.5pp
over transcript-only: Haiku calls its off-topic reads noise and then reasons about the model's
training cutoff, so the J++ framing alone pushes it toward stale answers. Compare J++ with its own
placebo only. The signal is at the last token, L60: its top 3 hold the first token of Qwen's answer
on 101 of 200 items (other layers ≤ 33), which is close to reading Qwen's first output token.
Dropping L60 costs 1.8pp (n.s.).

**NLA** (`nla_inline`, `nla_placebo_inline`; run `cfbench/2026-10-29_cutoff-qa-haiku-200-nla`, reads
`nla_reads/2026-10-29_cutoff200-tail`). Camila Blank's all-layer NLA
(`agu18dec/qwen3.6-27b-all-layer-nla-rl2000`, [docs/nla-reader.md](nla-reader.md)), greedy, one
free-text readout per position and layer at the O-lens block's six end-of-prompt tokens and layers,
every readout shown with its critic FVE (no filter). The block is ~5.6k tokens, the O-lens block's
size. Placebo: the readouts of an unrelated MMLU-Pro no-hint prompt at the same six tokens (200 read
in the same job), not another cutoff item. Mean critic FVE (pooled ruler) 0.35 at L32 to 0.53 at
L52; the old 11-layer NLA scored 0.11–0.23 on hint_mcq. Exact match: NLA 81.8%, NLA placebo 73.3%,
O-lens − NLA −0.7pp [−5.0, +3.7], NLA − its placebo +8.5pp [+4.5, +12.5]. The NLA placebo is +6.7pp
over transcript-only, the same size as the J++ placebo's: an off-topic reads block alone moves Haiku
toward the stale answer (stale items 55% → 65%). Read-level recall is the O-lens's: some readout
names Qwen's modal answer on 177 of 200 items (L52 160, L60 157, L32 131); Haiku is right on 87% of
draws where it does and 42% where it doesn't. The readouts are genre-and-format descriptions ("terse
Q&A prompt, expects a single name") with a guessed answer in them, and the guess is Qwen's stale
one. Examples: Nike CEO,
[NLA](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/91106e8f-c7c9-41d3-865c-66be78faba98)
(John Donahoe, Qwen's answer) vs
[placebo](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/c6696534-fba0-47eb-8167-7154dfbe45db)
(Elliott Hill, the current one); Ferrari's first fully electric car,
[NLA](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/57a1de6c-a98f-4980-8825-4009f773c099)
(Purosangue, Qwen's wrong answer) vs
[placebo](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/f96fa1a0-e3ef-4ddc-8669-269b2ace3d66)
(Elettrica, the real one). The NLA has the O-lens's caveat below: its AV is a Qwen3.6-27B LoRA with
Qwen's cutoff.

**Tool-call ablation: the answer 20 tokens away** (`cfbench/evals/cutoff_qa_tool_inline/`, PR #403;
runs `cutoff_gate/2026-10-29_v2-tool-required`, `olens_reads/2026-10-29_cutoff-tool200-tail`,
`cfbench/2026-10-29_cutoff-qa-tool-haiku-200`). Without a tool, the last read (`\n` after
`assistant`) is 5 tokens before the answer (the empty think block). Here Qwen gets a
`submit_answer(answer)` tool with `tool_choice: "required"`, so the template puts the tool schema in
a system block and the reply opens with the call wrapper; in Qwen's native format the answer is then
20 tokens past the last read. The bank v2 gate re-run under the tool: 325 stable (95 right, 230
stale) vs 266 without it, and of the 209 items stable both ways 207 keep the same modal answer.
Cases: the first 200 non-numeric v2 items stable under the tool gate (57 right, 143 stale; 117
overlap this section's 200), labels from that gate, reads at the same six tokens and layers on the
tool render, the same Haiku prompt (question only, no tool schema), 3 draws.

| condition            | exact | stable_right | stable_wrong | vs transcript, win/loss | p      |
| -------------------- | ----- | ------------ | ------------ | ----------------------- | ------ |
| transcript_inline    | 58.0% | 93.6%        | 46.4%        |                         |        |
| olens_inline         | 76.0% | 89.5%        | 71.8%        | 59 / 13                 | <0.001 |
| olens_placebo_inline | 61.7% | 91.2%        | 51.7%        | 34 / 23                 | 0.19   |

The O-lens − placebo gap is the same (+14.3pp vs +14.2pp without the tool); the lower absolute
numbers are the item set (72% stale vs 69%). On the 117 shared items, exact match without → with the
tool: transcript 70.9% → 70.1%, O-lens 85.8% → 89.7%, placebo 72.4% → 74.6%. Read-level recall 170
of 200 (127/143 stale); the newline after `<|im_end|>` still carries it (166). So moving the answer
from 5 to 20 tokens past the last read costs the lens nothing.

![no tool vs tool call: transcript-only, O-lens, placebo](img/cutoff-qa-tool-inline.png)

(`uv run python -m graphs.cutoff_qa_tool_inline`.)

**Split by whether Qwen wants the tool.** The reads are on the prompt, which is the same under
`tool_choice` auto and required, so forcing the call changes nothing the lens sees. What matters is
whether Qwen, at the end of that prompt, is about to open the call wrapper or to blurt the answer.
Bucketed by its unforced tool-call rate in `cutoff_gate/2026-10-29_v2-tool` (8 samples), exact
match:

| unforced tool calls | items | transcript | O-lens | placebo | O-lens − placebo [95% CI] | reads name answer |
| ------------------- | ----- | ---------- | ------ | ------- | ------------------------- | ----------------- |
| 8/8 (always)        | 90    | 56%        | 79%    | 64%     | +15 [+7, +23]             | 76/90             |
| 1–7/8 (mixed)       | 77    | 63%        | 79%    | 64%     | +15 [+8, +23]             | 65/77             |
| 0/8 (never)         | 33    | 53%        | 63%    | 52%     | +11 [−2, +24]             | 29/33             |

Stale items only: always 78% vs placebo 58% (n = 74), never 37% vs 10% (n = 17). The lens gains as
much on items where Qwen always opens the tool call as on the rest, so it isn't only reading items
where the answer is Qwen's intended next output. The buckets aren't randomized (the never-tool items
include the ones Qwen answers in hedged prose), and the shared-cutoff caveat below still applies.

**Tool-call ablation at scale: bank v3, O-lens, J++ and NLA (2026-10-30).** Run
`cfbench/2026-10-30_cutoff-qa-tool-v3` (harness 14; [Docent](https://docent.transluce.org)
collection `cfbench`). Bank v3's 2,170 stable non-numeric items re-gated under the tool
(`cutoff_gate/2026-10-30_v3-tool-required`: 1,849 stable, 1,842 with the no-tool answer, 85%;
`..._v3-tool-auto`: unforced tool calls on 70.7% of replies, 1,047 items always). Cases: the 1,849
stable items (456 stale). Haiku now sees the system turn (the tool definition) above the question.
Reads on the tool render (`olens_reads/2026-10-30_cutoff-v3tool-punct`,
`nla_reads/2026-10-30_cutoff-v3tool-punct`, `jpp_reads/2026-10-30_cutoff-v3tool-user`): O-lens and
NLA at every punctuation token of the question plus the last 10 prompt tokens (the empty think block
included; 13.1 positions per item), layers 32–60; J++ at every token from the user turn on. O-lens
and NLA placebos are another item's reads mapped onto this item's positions and shown under this
item's text. Seven conditions × 3 draws, exact match, per-item means; paired 95% bootstrap CIs.

| items                        | n     | transcript | O-lens | placebo | J++  | placebo | NLA  | placebo |
| ---------------------------- | ----- | ---------- | ------ | ------- | ---- | ------- | ---- | ------- |
| all                          | 1,849 | 83.3       | 87.1   | 81.5    | 86.5 | 84.6    | 87.9 | 86.8    |
| stale                        | 456   | 49.4       | 72.5   | 55.5    | 68.5 | 59.4    | 72.0 | 64.0    |
| not stale                    | 1,393 | 94.4       | 91.8   | 90.0    | 92.4 | 92.9    | 93.1 | 94.3    |
| always calls the tool (8/8)  | 882   | 83.8       | 88.3   | 82.2    | 87.6 | 86.2    | 88.8 | 88.1    |
| always calls the tool, stale | 232   | 49.9       | 73.0   | 56.9    | 71.6 | 62.2    | 73.7 | 67.0    |
| never calls the tool (0/8)   | 222   | 83.8       | 86.9   | 83.3    | 87.1 | 84.7    | 86.6 | 85.9    |

| lens − its placebo, pp | O-lens               | J++                | NLA                |
| ---------------------- | -------------------- | ------------------ | ------------------ |
| all                    | +5.5 [+4.2, +6.9]    | +1.9 [+0.9, +3.0]  | +1.1 [+0.1, +2.1]  |
| stale                  | +17.0 [+13.6, +20.4] | +9.1 [+6.2, +12.3] | +8.0 [+5.2, +11.0] |
| always tool, stale     | +16.1 [+10.9, +21.1] | +9.3 [+5.5, +13.4] | +6.8 [+3.0, +10.6] |

- **The O-lens result holds at 9x the items with the answer 16 tokens past the last read:** +17pp
  over its placebo on stale items, +16pp on the 232 stale items where Qwen always opens the tool
  call unforced. It isn't reading only the items where the answer is Qwen's intended next output.
- **J++ and NLA also beat their placebos, by about half as much** (+9 and +8 on stale). Against
  transcript-only all three look alike (+19 to +23 on stale), because every placebo block alone
  pushes Haiku toward the stale answer (O-lens placebo 55.5, J++ 59.4, NLA 64.0 vs transcript 49.4).
- **On not-stale items the reads cost 1–3pp** against transcript-only, as before.
- On the 863 items shared with the no-tool 1k run, each condition is within 1pp of its no-tool
  number (O-lens 88.5 → 88.1). That run read only the six tail tokens and showed Haiku no system
  turn, so this compares the whole setup, not the tool alone.

Cost: Haiku $65 (38,829 predictions, from token counts at $0.10/$0.50 per M; the rows' `cost` is 0
under BYOK); GPU about $45 (two gates on one H200, then 2 O-lens and 3 NLA pods for ~1.3 h).

**Tool run without the tail reads (2026-10-30, Kylie: are the last 10 tokens just reading the answer?).** The v3 tool
run's O-lens block cut to the question's own punctuation (`qpunct`: every punctuation token of the user turn, its final
"." included, 4.1 per item) or that plus `<|im_end|>` (`qpunct_imend`); the other 9 / 8 tail positions dropped, no new
reads (`cut_reads`, inputs `cfbench_evals/2026-10-30_cutoff-qa-tool-inline-inputs-qpunct{,-imend}`). Placebo as above
with the same seeds, so each draw's placebo comes from the same other item as in the full run, cut the same way. The block's
preamble names only the positions shown. Runs `cfbench/2026-10-30_cutoff-qa-tool-v3-olens-qpunct{,-imend}-full`
(conditions renamed `olens_qpunct(_imend)(_placebo)_inline` after the run; the runs without `-full` are the partial ones
that stopped on OpenRouter credits, reused), 22,188 predictions, 0 invalid; transcript-only and the full set from the v3
run. Error rate, per-item means over 3 draws, paired 95% bootstrap CIs by item:

| items               | n     | transcript | qpunct | placebo | qpunct + `<\|im_end\|>` | placebo | full (+ last 10) | placebo |
| ------------------- | ----- | ---------- | ------ | ------- | ----------------------- | ------- | ---------------- | ------- |
| all                 | 1,849 | 16.7       | 12.8   | 16.2    | 12.6                    | 14.6    | 12.9             | 18.5    |
| stale               | 456   | 50.6       | 31.6   | 44.4    | 31.2                    | 41.6    | 27.5             | 44.5    |
| not stale           | 1,393 | 5.6        | 6.7    | 6.9     | 6.5                     | 5.8     | 8.2              | 10.0    |
| always tool, stale  | 232   | 50.1       | 28.9   | 41.8    | 28.7                    | 38.8    | 27.0             | 43.1    |

| error, pp           | qpunct − placebo     | + `<\|im_end\|>` − placebo | full − placebo       | qpunct − full      |
| ------------------- | -------------------- | -------------------------- | -------------------- | ------------------ |
| all                 | −3.4 [−4.5, −2.2]    | −2.0 [−3.2, −0.9]          | −5.5 [−6.9, −4.2]    | −0.1 [−1.0, +0.8]  |
| stale               | −12.8 [−15.9, −9.9]  | −10.4 [−13.7, −7.1]        | −17.0 [−20.4, −13.7] | +4.1 [+1.5, +6.8]  |
| always tool, stale  | −12.9 [−17.8, −8.2]  | −10.1 [−14.9, −5.5]        | −16.1 [−21.1, −10.8] | +1.9 [−1.7, +5.6]  |

- **Most of the gain survives without the tail:** on stale items the question's punctuation alone is 12.8pp better than
  its placebo, against 17.0pp with the last 10 tokens. The tail adds about 4pp on stale items (31.6 → 27.5) and costs
  1.5pp on not-stale ones, so overall the two are level. Same as the 1k no-tool run, where the question's commas were as
  good as the six tail tokens.
- `<|im_end|>` adds nothing on top (−0.4pp on stale, CI across 0); its placebo is a little better than the qpunct one.
- Example (`e001-06`, the US Attorney General; Qwen's stale answer Merrick Garland): with qpunct reads Haiku says Garland
  on all 3 draws ([Docent](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/ba3a80c0-f6cf-4a79-8ee3-1dc0e8b49871)),
  with the placebo Pam Bondi on all 3 ([Docent](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/4be0fc4e-3592-4321-9694-ab9c1f925ba7)).
  Every conversation is in expview: `#dataset=runs/cfbench/2026-10-30_cutoff-qa-tool-v3-olens-qpunct-full` (and `-qpunct-imend-full`).

Cost: Haiku $18.50 (22,188 predictions, from OpenRouter's per-row `cost`); no GPU.

**Caveats.**

- **The lens is a Qwen3.6-27B LoRA, so it has Qwen's cutoff.** This design can't tell "the lens
  reads the answer Qwen represents" from "the lens reads only the question off the activation and
  answers it from its own (identical) weights". The placebo rules out the reads block as such, not
  this. WorkspaceBench's text-inversion gate (a strong reader given the prompt text) is the standard
  control and hasn't been run here; the closest existing number is the census in
  docs/envs/knowledge_cutoff.md (Flash, same cutoff, holds the stale answer on 24% of WSB items
  unaided). A per-item check would be: does the lens name the modal answer on items where a
  same-cutoff text reader does not.
- One investigator (Haiku), one draw of the lens per item, 200 of 240 live items. The effect is
  large enough that the sign test is unambiguous, but the stable_right loss shows the reads can also
  mislead a model that already knows.
- The env's `measure()` returns no `positive`, so `env_runs`/`olens_requests_from_runs.py` skip
  cutoff_qa rows; the requests were built straight from the episodes (CODE_SMELL.md).

**Next.** (1) The text-inversion control above. (2) Positions: the `\n` after `<|im_end|>` alone
carries nearly all the recall; a single-position, single-layer run is a 6x cheaper read. (3) More
items (the 1k run above used 999 of bank v3's 2,196 stable items). (4) Other investigators (Sonnet,
GPT-6.1), and `olens_agentic` with the lens as a tool. (5) A placebo that keeps the framing neutral:
the J++ and NLA placebos move Haiku on their own.
