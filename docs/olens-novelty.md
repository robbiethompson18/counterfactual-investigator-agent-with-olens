# Novelty: does an O-lens read explain more than the text that follows? (dead end as a metric)

> **Dead end as an interestingness metric, don't build on it.** The code was never merged. It lives
> on the closed PR
> [#122](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/pull/122),
> branch `olens/novelty` at `6e9cb2f`: `olens/src/olens/novelty.py`, `scripts/olens_novelty.py` (CPU
> scorer), `scripts/olens_atoms.py` (GPU atom dumps for existing read runs). It is stacked on the
> residual-reads branch ([olens-residual-reads.md](olens-residual-reads.md)) and uses
> `olens.continuation` (PR #120). A separate session wrote this, and Robbie stopped it before its
> write-up. This doc is assembled from its run dir.

2026-10-23. The question: the O-lens mostly reads back the text the model is about to write. Is it
_only_ that? If a read explains part of the activation that the realized text doesn't, that part
might be "interesting" (unverbalized), and the gain could rank positions or bullets for an
investigator.

**Answer: by FVE, the lens explains clearly more than the continuation, but the extra doesn't pick
out new content.** About half of a read's FVE lies beyond the next 32 tokens, and far above a null.
But bullets that echo the transcript score as high as bullets labelled counterfactual. Within a read
(same position), counterfactual bullets don't score higher than the others.

## Metric

Per position, with z the whitened activation, C the AR atoms of the continuation's prefixes (1, 2,
4, 8, 16 and 32 tokens) and B the read's bullet atoms, NNLS in the pooled whitened space:

- `cont = FVE(z; C)`
- `gain = FVE(z; C ∪ B) − cont`
- per bullet `n_b = FVE(z; C ∪ {b}) − cont`
- `novelty = gain − null_other`, where null_other is the gain from bullets read at a position of
  another transcript

The `_transcript` variant adds AR atoms of every 32-token chunk of the whole transcript to C. That
asks "is this anywhere in the transcript?" rather than "in the next 32 tokens?".

## Data

`data/runs/olens_novelty/2026-10-23_space-reads/` on the HF dataset. It holds atoms and scores for
the 7 space-lens `olens_reads/2026-10-22_*` runs: cfbench spike set, donation bet, hidden goal
v1–v3, overnight v1/v2-targets. That's up to 50 positions per transcript at L44 (L48 for hidden
goal), plus `residual-v1/` scoring the residual-reads run's plain reads. `summary.txt` has the full
tables. `cf_labels/` has the labelling.

## Results

L44, means over positions:

| run                 | positions | cont (next 32) | lens alone | gain  | null_other | gain, whole transcript as base |
| ------------------- | --------- | -------------- | ---------- | ----- | ---------- | ------------------------------ |
| cfbench spike       | 6,219     | 0.069          | 0.131      | 0.075 | 0.002      | 0.045                          |
| donation bet        | 6,000     | 0.108          | 0.208      | 0.112 | 0.004      | 0.068                          |
| hidden goal v1–v3   | 18,528    | 0.10           | 0.20       | 0.112 | 0.006      | 0.08                           |
| overnight v1        | 800       | 0.067          | 0.128      | 0.074 | 0.003      | 0.048                          |
| residual-v1 (plain) | 2,177     | 0.136          | 0.201      | 0.085 | 0.003      | n/a                            |

- **Most of what the lens explains lies beyond the realized continuation.** Gain exceeds the
  other-transcript null at 97–100% of positions. With the whole transcript as the base, about
  0.04–0.09 is still left.
- **That gain isn't "new content".** The fork's Sonnet subagents labelled 1,118 bullets in 280 reads
  (a random sample, not selected on novelty): ALT 465, UNRELATED 241, INVENTED 169, COUNTERFACTUAL
  85, ECHO 62, OTHER 56, ELSEWHERE 40. The exact rubric isn't saved in the run dir; the labels read
  as alternative phrasing / unrelated / invented details / a different action or decision / echo of
  nearby text / other / found elsewhere in the transcript. Mean n_b over the next 32 tokens:

  | label          | n   | n_b   |
  | -------------- | --- | ----- |
  | ECHO           | 62  | 0.077 |
  | COUNTERFACTUAL | 85  | 0.070 |
  | ALT            | 465 | 0.057 |
  | INVENTED       | 169 | 0.052 |
  | ELSEWHERE      | 40  | 0.048 |
  | UNRELATED      | 241 | 0.021 |
  | OTHER          | 56  | 0.009 |

  Echoes of the transcript score highest.

- **Counterfactual vs the rest:** AUROC 0.60 (p = 0.002) across bullets, and 0.60 (p = 0.04) at read
  level. That's confounded by position, because some positions just have higher FVE. Within the 35
  reads that mix counterfactual and other bullets, it's 0.061 vs 0.058 (p = 0.39). Holding the
  position fixed, n_b doesn't find the counterfactual bullet.
- **The highest-novelty positions** (`*/novelty_extremes.md`) are mostly bullets that restate text
  further away in the same transcript, which the 32-token base doesn't cover. One example: a
  whistleblow email the model drafts later.

## Lessons

- "FVE beyond the continuation" measures how much of h the lens explains outside the next tokens. It
  does not measure whether the content is new to a reader. Use a text-level judgment (or resampling)
  for that.
- A novelty claim needs a null matched to the perturbation, not only an "other transcript's bullets"
  null. In the residual-reads run, edits of h that removed an unrelated direction produced as much
  on-topic novelty as the real edit ([olens-residual-reads.md](olens-residual-reads.md)).
- `olens.continuation.nnls` must be float64 per-row NNLS. A Gram-form solve blew up on the
  near-collinear prefix atoms. This fork found the bug.
