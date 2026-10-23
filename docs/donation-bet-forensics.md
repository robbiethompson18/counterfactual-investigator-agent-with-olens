# Donation Bet forensics: lens reads and single-transcript investigators

2026-10-23 (Kylie). Two things on the Donation Bet goodbad env
(`evalcue/envs/donation_bet/goodbad.py`, Qwen3.6-27B, thinking on): token-level O-lens / J++ / NLA
reads on a sample of transcripts, browsable in a static port of the lens case page, and an
investigator test of whether an agent reading one transcript can tell whether the bet changes Qwen's
answers.

## Lens reads

- Run: `data/runs/donation_bet_forensics/2026-10-23_olens-jpp-nla/` (`scripts/dbet_forensics.py`).
  116 transcripts from the 2026-10-22 `donation_bet_goodbad` runs: 8 per arm for good vs bad and
  good vs beers (both directions), plus matched pairs of good vs nothing and good vs beers (same
  question, same direction; 78 pairs covering both-above / both-below / split estimates and the
  CoT-acknowledges combinations). The rollouts had no per-request seed, so pairs match on question
  and direction only.
- Reader: the case page's pod server (`transcript-viewer/pod/server.py`, `sky/pod_h200.yaml`): space
  O-lens `olens_s3d_space_rl600` + AR (4 bullets, per-bullet and joint whitened FVE), J++
  (`olens.extras.JPP`), ceselder L42 NLA with its critic FVE. Token ids are the thinking-on chat
  template re-tokenised from the stored text.
- Coverage (stopped by request, so partial): J++ on every token of all 116; O-lens L40 + NLA on
  punctuation, newline and chat-template tokens of all 116; full-token O-lens L40 on 36 transcripts.
  Full-token NLA and L32/L48 were planned only for the 16 showcase transcripts in `showcase.json`
  and were not finished. `meta.json` lists per-step status.
- Viewer: `scripts/dbet_case_site.py RUN OUT [--lite]` writes a static `index.html` (the case page
  reading cached reads; GPU-only panels hidden) and `pairs.html` (side-by-side pairs with filters
  for direction, estimate side, CoT acknowledgment and the showcase). Case data ships as
  `cases/*.gz.txt` (base64 gzip). `--lite` keeps fewer J++ layers so it fits an Artifact.
- Fleet notes: a shared mkdir-lock job queue (one worker per pod) worked well. Vast "H200" offers
  were a 47 GB card and a 99 GB disk, so use RunPod only for this server. O-lens on one H200 runs
  ~5.5 reads/s, NLA ~3.6/s, J++ ~50 tokens/s.

## Single-transcript investigator

- Run: `data/runs/donation_bet_investigator/2026-10-23_transcript-only/`
  (`scripts/dbet_investigate.py`, `scripts/dbet_prior_analysis.py`). A case is one Qwen rollout plus
  the mirrored prompt (the outcomes swapped). The question: would the share of estimates above the
  line change by more than x pp? Ground truth per (prompt, question) comes from the existing ~100
  rollouts per arm (exceeds / does-not-exceed mirror pairs). 480 cases: 3 comparisons × 2 directions
  × 10 questions × 8 rollouts. DeepSeek V4.1 Flash, reasoning effort medium, 3 draws (Sonnet 5.5
  stopped part-way on cost).
- Results (x = 10 pp): transcript AUROC 0.87 [0.83, 0.91], but prompt-only (no transcript) 0.90 with
  much better log-loss. Per case the transcript adds nothing over the prompt and makes the
  investigator over-confident. A logistic regression of the label on logit(prompt-only) and
  logit(transcript) still gives the transcript term +1.05 [0.80, 1.46] on beers + nothing, and
  leave-one-question-out log-loss improves 0.56 → 0.44. Within good vs beers alone, the transcript
  ranks questions at 0.70 where the prompt is at chance (0.54).
- Group discrimination (good vs nothing = effect, good vs beers = null), matched-pair concordance:
  0.84 with the transcript vs 0.95 prompt-only, because the prompt names the group. Blinding the
  group by redacting the other outcome everywhere (prompt, mirror, CoT, reply; 3/960 calls leaked)
  gives 0.60 [0.56, 0.65] with the transcript vs chance prompt-only. Giving the prompt-only answer
  as an explicit prior made things worse.
- Redaction is crude (paraphrases, a placeholder that is itself a cue). Proposed next design: many
  other outcomes with measured effects, scoring transcript vs prompt-only predictions across
  outcomes, then the lens-tool condition on top.

## Pages (private claude.ai artifacts)

expview transcripts + CoT judge `Qb8tg4N6vu9iv4tP93e2VJ`; case page `W8WHWi7ywYjPG5YfDqQuya`;
matched pairs `3N2eXCGZA93LqXshK2zrVc`; investigator `Jn1ryEi7UJy3b34goL6WRM`; blinded / prior
conditions `XC3ArQSETf5n4CPsRxfCuC` (each at `https://claude.ai/artifact/<id>`).

Not synced to HF: the repo's `hf.key` gets 403 on the sprint dataset (`scripts/runs.py finish` fails
at upload).
