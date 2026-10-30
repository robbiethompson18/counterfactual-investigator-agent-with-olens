# favorites_qa_inline: predicting Qwen's own favourites from end-of-prompt reads

**Status (2026-10-29):** REFERENCE. 1,500 items, one investigator (Haiku), three draws; O-lens, NLA
and J++ each against transcript-only and its own placebo.

**Question.** Can an investigator who never sees Qwen3.6-27B's reply guess its own pick for a
free-response preference question ("Which text editor would you choose for editing a config file
over SSH? Answer with just the name.")? And do metamodel reads of the prompt help? The label is
Qwen's modal answer from the bank's stability gate (≥ 7/8 samples agree). There is no truth. This is
WorkspaceBench's `basic_readout` implicit family scaled from 32 items to 1,500
([docs/envs/favorites_qa.md](envs/favorites_qa.md)), run the same way as
[cutoff_qa_inline](cutoff-qa-inline.md).

## Setup

`cfbench/evals/favorites_qa_inline/` (its docstring has the commands).

- **Items:** every stable item of bank v3, 1,500 of 5,187 questions, by shape: 409 `pick_one`, 338
  `recommend`, 294 `choose_for`, 277 `favorite`, 175 `name_one`, 7 pilot `be_one`/`least_favorite`.
  Qwen env run `favorites_qa/2026-10-29_v3-1500-nohint` (n = 1, shared server, thinking off, T 1).
- **Reads**, all at layers 32/40/48/52/56/60:
  - **O-lens:** `olens_reads/2026-10-29_favorites-v3-tail`. Space lens plus AR, T 1, at the six
    end-of-prompt tokens (".", `<|im_end|>`, "\n", `<|im_start|>`, "assistant", "\n"). The
    thinking-off render then has `<think>\n\n</think>\n\n`, so the last read is 4 tokens before the
    answer.
  - **NLA:** `nla_reads/2026-10-29_favorites-v3-tail`. The all-layer NLA, greedy, at the same
    tokens, every readout shown with its critic FVE. Mean FVE is 0.31 at L32 and 0.52–0.56 at
    L40–L52.
  - **J++:** `jpp_reads/2026-10-29_favorites-v3-all`. Top 3 at every token of the rendered prompt.
  - **Cost and time:** two 1× H200 pods, about 55 minutes for the O-lens and NLA in parallel, then
    J++ (about 10 minutes) and the critic (about 30 minutes). About $12 of pod time.
- **Investigator:** Claude Haiku 5.5 at effort medium, cutoff_qa_inline's one-turn prompt, model
  never named, 3 draws.
- **Placebo:** another favorites item's reads of the same kind, presented as this item's. The same
  other item is used for all three tools on a given (draw, case).
- **Runs** (all on
  [Docent](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e)):
  `cfbench/2026-10-29_favorites-haiku-v3-olens` (transcript-only, O-lens, O-lens placebo),
  `..._v3-jlens` and `..._v3-nla`. 31,500 predictions, $22.54, 0 invalid.
- **Scoring:** exact match of the normalized submission to Qwen's modal answer; paired by item
  (per-item mean over draws), 95% bootstrap CIs. Reproduce with
  `uv run python -m cfbench.evals.favorites_qa_inline.report data/runs/cfbench/2026-10-29_favorites-haiku-v3-{olens,jlens,nla}`.

## Result

| condition            | all   | pick_one | recommend | choose_for | favorite | name_one |
| -------------------- | ----- | -------- | --------- | ---------- | -------- | -------- |
| transcript_inline    | 80.3% | 80.0%    | 78.7%     | 77.0%      | 84.8%    | 82.5%    |
| olens_inline         | 87.6% | 89.8%    | 87.6%     | 85.6%      | 86.6%    | 87.4%    |
| olens_placebo_inline | 70.5% | 72.3%    | 70.6%     | 71.9%      | 62.2%    | 77.0%    |
| nla_inline           | 87.4% | 88.2%    | 86.6%     | 83.9%      | 89.9%    | 88.8%    |
| nla_placebo_inline   | 81.2% | 80.4%    | 80.7%     | 77.6%      | 84.2%    | 85.0%    |
| jlens_inline         | 79.4% | 77.4%    | 79.5%     | 77.0%      | 84.4%    | 81.3%    |
| jlens_placebo_inline | 53.3% | 49.1%    | 57.3%     | 53.4%      | 44.0%    | 70.3%    |

| paired difference | vs transcript-only  | vs its own placebo     | placebo vs transcript-only |
| ----------------- | ------------------- | ---------------------- | -------------------------- |
| O-lens            | +7.3pp [+5.5, +9.0] | +17.2pp [+15.3, +18.8] | −9.8pp [−11.4, −8.1]       |
| NLA               | +7.0pp [+5.5, +8.6] | +6.2pp [+4.9, +7.5]    | +0.8pp [−0.5, +2.2]        |
| J++               | −0.9pp [−2.9, +1.1] | +26.1pp [+24.0, +28.3] | −27.0pp [−28.8, −25.1]     |
| O-lens − NLA      | +0.3pp [−1.3, +1.8] |                        |                            |

- **The O-lens and the NLA each add ~7pp over transcript-only, and they tie.** NLA vs its placebo
  (+6.2pp, placebo neutral) is the cleanest comparison. The O-lens's +17pp over its placebo is
  inflated: its placebo sits 10pp below transcript-only.
- **Transcript-only is strong: 80% against cutoff's 67%.** Haiku's own idea of a typical favourite
  is usually Qwen's. The tools close about a third of the remaining 20pp. The misses are Haiku's
  defaults: vim for nano, camel for fennec fox, Cabernet Sauvignon for Pinot Noir, chelo kabab for
  ghormeh sabzi. On 129 items transcript-only is 0/3 and the O-lens 3/3. Example: SSH config editor,
  [transcript-only says vim](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/460e44d2-2ece-43d9-bdf2-2863b9032710),
  [with the O-lens nano](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/85919084-e62d-4896-a140-172639b33ed6).
- **The answer is in the reads almost always.** Some O-lens bullet contains Qwen's modal answer on
  1,490 of 1,500 items, at the "\n" after `<|im_end|>` on 1,485. NLA readouts name it on 1,477, also
  mostly at that "\n". Haiku converts that into 87%: about one time in ten it picks another
  candidate from the bullets.
- **Placebos show how much Haiku trusts each block.** The O-lens placebo names a specific wrong
  answer, and Haiku follows it (−10pp). On 30 items transcript-only is 3/3 and the placebo 0/3 (e.g.
  [favourite landform](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/003e53ca-242a-4729-9438-c0747d5e8d95)).
  The NLA placebo is neutral; its readouts are genre-and-format descriptions with hedged candidates
  ("likely Snake, Rat, or Dog"). The J++ placebo is −27pp, but it shows the other question's own
  tokens, so it is an obviously mismatched block and not a clean null. Compare J++ with
  transcript-only.
- **J++ adds nothing, though the signal is there.** At the last prompt token, the J++ top 3 at L60
  hold the first token of Qwen's answer on 910 of 1,500 items (61%), and on 925 at the "\n" after
  `assistant`. Haiku doesn't use it, likely because it is a sub-word fragment in a ~4.5k-token block
  of mostly noise. On cutoff J++ helped (78% vs 67%).

## Caveats

- **Close to a next-token read.** The answer sits at the "\n" after `<|im_end|>`, a few tokens
  before Qwen answers, as on cutoff. Neither lens is shown reading a preference earlier than the
  model forms its answer. Sampling Qwen once on the prompt would score ~100%.
- **One investigator.** Haiku only. Flash, the repo's default predictor, was not run.
- **Exact match is strict.** "Scones" for "Scone" counts wrong, as does "bbq sauce" for "barbecue
  sauce". The loose match in `cfbench.evals.cutoff_qa_inline.report` is not applied to the pooled
  table.
- **The writers' styles leak.** Each writer's 50 questions share a domain, and per-writer stable
  yields range 3–30 of 50. Stratifying by writer, not done here, would show whether gains
  concentrate in a few domains.
