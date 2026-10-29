# cutoff_qa_inline: predicting Qwen's stale answer from end-of-prompt O-lens reads

**Status (2026-10-29):** REFERENCE. First smoke, 200 items, one investigator; the caveats below are
open.

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

![transcript-only vs O-lens, J++ and NLA and their placebos, all / stale / not stale](img/cutoff-qa-inline.png)

![paired differences between conditions with 95% CIs, grouped by question](img/cutoff-qa-inline-diff.png)

(`uv run python -m graphs.cutoff_qa_inline` makes both; exact match, 95% bootstrap CIs over items,
the second paired by item. This is the one cutoff chart: every lens goes on it.)

**Result** (`cfbench/2026-10-29_cutoff-qa-haiku-200`,
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

**J++** (`jlens_inline`, `jlens_placebo_inline`, `jlens_inline_no60`; PR #393, runs
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
carries nearly all the recall; a single-position, single-layer run is a 6x cheaper read. (3) The
other 40 stable items, then a bigger bank (`scripts/cutoff_facts/BRIEF.md`). (4) Other investigators
(Sonnet, GPT-6.1), and `olens_agentic` with the lens as a tool.
