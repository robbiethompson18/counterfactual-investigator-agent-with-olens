# Docs index: everything not ACTIVE

**Status (2026-10-29):** ACTIVE. Index of every REFERENCE, MIXED, NULL, SUPERSEDED and ORPHAN doc;
the ACTIVE ones are indexed in the repo's `CLAUDE.md`, which every session loads.

Read this before starting on a topic, and when looking for prior results or dead ends. Tags are
defined in `CLAUDE.md`'s Docs section (the 2026-10-26 audit behind them is
`docs/local/doc-audit-2026-10-26.md`, Robbie's checkout only). Add a new entry at the end of its
tag's section; when a doc's tag changes to or from ACTIVE, move its entry between here and
`CLAUDE.md`. Links here are relative to `docs/`.

## REFERENCE

- [REFERENCE: CHIVE data sources — read before fetching incidents, readouts or Agam's results](chive-data-sources.md)
  — HF dataset vs Agam's viewer, what each lacks, splits, file paths
- [REFERENCE: CHIVE critique — read before quoting what is wrong with CHIVE's data, thresholds or ground truth, or before replicating those numbers](chive-critique.md)
  — on Adam's release and Agam's run: 40% off-policy prefixes, 8,623 clean kills (12% of
  experiments, 32% of investigations) the 50pp rule discards, 12% of incidents with
  self-contradicting ground truth; `python -m chive_critique`
- [REFERENCE: Qwen self-report baseline — read before comparing anything to Qwen explaining itself](qwen-self-report-baseline.md)
  — setup, results on dev96, Sonnet 5.5 judge gotcha
- [REFERENCE: Why the O-lens writes Chinese — read before interpreting Chinese in O-lens readouts](olens-chinese.md)
  — how much there is in our banks; evidence it is a training artifact (stripped bullets + Chinese
  in RL data)
- [REFERENCE: Why the DeepSeek V4 Flash O-lens writes Russian — read before interpreting Russian or CJK in Camila's V4 lens reads, or suspecting our reader](dsv4-lens-russian.md)
  — 3–9% of reads, concentrated at a few shared instruction-trailer positions, on-topic
  translations; not the whitespace bug, reader audited clean, same in Camila's own WorkspaceBench
  readouts
- [REFERENCE: Smoke25 results — read before comparing anything to Agam's CHIVE harness numbers](smoke25-results.md)
  — our wrapper reproduces his free-form pass@1; Flash hits the call cap on yes/no with O-lens
- [REFERENCE: False capability claims — read before studying Qwen claiming to call 911, run code, text someone etc.](false-capability-claims.md)
  — the device-channel lever (1.6% → 41%), severity dose-response, judge rubric v2 + audit,
  generator, and the negative held-out result outside emergencies
- [REFERENCE: donehack endings — read before claiming how episode endings (done tool, submit script, give_up) affect reward hacking](donehack-endings.md)
  — V4 Flash / Qwen3.6-27B hack rates by ending (reply 52% → binary-reward submit script 88–95%),
  give_up raises it, reward schemes
- [REFERENCE: envport results — read before quoting GLM 5.3 on a batch-2 agent-interp-envs port (eval_tampering, games, parse_duration, revert_or_refactor, …)](envport-glm53-results.md)
  — the per-env result tables and transcript notes that used to live in each port's README; batch-1
  numbers stay in glm53-agent-interp-envs.md
- [REFERENCE: GLM 5.3 realism variants — read before running or extending the user-style ImpossibleBench ladder or the email over-eagerness envs](glm53-realism.md)
  — criterion-free 8-rung ladder (hack 27% → 3%, test edits 0% → 56%), evalness ranking, O-lens
  prompt and grader reads, email in-between rungs (casual ask alone → 2%)
- [REFERENCE: funding_email in evalcue — read before running or extending the funding_email env (drone pitch vs Myanmar legal memo)](envs/funding_email.md)
  — port of agent-interp-envs with its own tool schemas (`Episode.tools`/`step`), detector vs
  subagent labels, replication of the screen (14/30 vs 0/30)
- [REFERENCE: funding_email_real — read before running the realistic (Claude Code harness, traffic-analytics) funding_email grid](envs/funding_email_real.md)
  — upstream/core/full harness × thin/real docs × memo, how to render an arm, unfixed sandbox tells
- [REFERENCE: Sandbagging iteration — read before running, extending or labelling the multi-arm sandbagging env](sandbagging-iteration.md)
  — one system-prompt sentence ("Make sure you achieve your goal in the long term.") moves Qwen's
  deliberate sandbagging 17% → 73% (n=30, p=1e-5); config-driven overlay, every arm tried
- [REFERENCE: Sandbagging in evalcue — read before running the sandbagging env (the evalcue port; retired, code at `c7b2e56`)](envs/sandbagging.md)
  — `evalcue/envs/sandbagging/`: all 22 arms verbatim, the original tool loop, `detect()` matching
  544/544 original labels; replicates on OpenRouter (`a_goal` 8/30 vs `a_goal_lt` 21/30, p = 8e-4);
  GLM 5.3 baseline 2026-10-24 (deliberate ≈1/100, the detector positives are slips)
- [REFERENCE: cfbench regraded as binary — read before asking whether the report's RMSE conclusions depend on the scoring, or scoring 'does the rate move ≥ T pp?'](cfbench-binary-regrade.md)
  — same Flash predictions as yes/no at T = 5–30pp: O-lens vs placebo still a wash; on GLM
  env-diff-only wins the binary grading (bigger moves, no overshoot penalty); needs a real
  binary-prompt rerun to compare formats
- [REFERENCE: cfbench by predictor model — read before choosing a cfbench predictor, effort or tool budget, or claiming O-lens uplift across models](cfbench-model-chart.md)
  — 7 models × env diff/transcript/O-lens/placebo on 4 cases (100 − RMSE): transcripts worth up to
  +21pp, O-lens beats placebo only for Flash and Argon via hidden_goal; Flash effort and tool-budget
  sweeps
- [REFERENCE: Shutdown resistance envs — read before running or extending shutdown_palisade/killswitch/cron/persist](envs/shutdown.md)
  — Palisade (2025) port + 3 siblings, ~200 arms; "The catch is that by default…" flips bypass
  (20/26 vs 0/20), "write access … configuration files" disables the kill switch 36/40 bf16, 30/40
  in the fp8 canonical run (mostly unverbalized); every arm in
  [shutdown-arms.md](envs/shutdown-arms.md)
- [REFERENCE: Scheming envs — read before running Lynch blackmail/leak/murder or Apollo oversight/exfil/goal-guarding on Qwen](envs/scheming.md)
  — 12 envs (bf16 rates; fp8 canonical exfil run in the doc); Lynch main cell 87%/86%/53%; exfil 48%
  → 20% with an eval cue (rarely verbalized); placebo caveats; tables in
  [scheming-tables.md](envs/scheming-tables.md)
- [REFERENCE: Prompt-injection env — read before running or extending injection_slack (AgentDojo Slack port)](envs/injection_slack.md)
  — Qwen follows AgentDojo's injection ~47%; signer ≠ user cuts it to 9% (mostly unverbalized),
  "after my task" lifts it to 67%; thinking off raises compliance and makes it silent
- [REFERENCE: Character drift — read before prefilling Qwen with out-of-persona assistant turns, or reading the lens at the Assistant token in roleplay](envs/character_drift.md)
  — 40 Sonnet-written + 40 WildChat drift transcripts × system prompt × follow-up; in-world ~90%
  stays in character, "what are you?" 13% (thinking on) vs 48% (off); O-lens at the Assistant token
  says roleplay/persona (not "disclaimer") and previews the reply, predicting snap-back (AUROC
  0.72-0.95)
- [REFERENCE: Thinking on vs off — read before running Qwen with `--thinking 0` or picking unverbalized cases for the O-lens](thinking-off.md)
  — donation bet + `bet_neutral` control, Lynch blackmail/leak/murder, exfil on OpenRouter; thinking
  off ignores "Do not ..." rules (murder 10% → 40%) and often acts with no text at all; value
  leakage is one-sided (`good_above` = control)
- [REFERENCE: Whistleblowing envs — read before running the SnitchBench port or its single-turn variants](envs/whistleblow.md)
  — bold/tame × email/CLI grid, hotline/industry/values levers with many mid-rate arms; tables in
  [whistleblow-tables.md](envs/whistleblow-tables.md)
- [REFERENCE: Hiring-from-technical-evidence envs — read before running coding-test, interview or repo-review hiring arms](envs/hiring_eval.md)
  — names alone null; a diversity sentence × name splits White vs Black male (39/60 vs 57/60,
  verbalized); context sentences span 0–100%
- [REFERENCE: Overnight 2026-10-28 handoff (super experimental) — read before picking up debug_pointer, style_transfer, todo_suppressed or notice_defect](overnight-2026-10-28-handoff.md)
  — what the four gated envs are, survivors per env, O-lens vs placebo (null on debug_pointer and
  style_transfer), where code and runs live, caveats, what to do next
- [REFERENCE: cutoff_qa_inline — read before claiming the O-lens, J++ or NLA predicts Qwen's stale answers, or rerunning the end-of-prompt investigator](cutoff-qa-inline.md)
  — Haiku 5.5 predicting Qwen's modal answer from the question, 999 items: only the O-lens beats its
  own placebo by much (+7.6pp; NLA +2.8pp, J++ +1.7pp), the J++ and NLA placebos alone add +3–5pp
  (cutoff framing); the lens writes the answer at the ` ` after `<|im_end|>`; with a required
  submit_answer tool call (200 items, answer 20 tokens past the last read) the O-lens gap holds (76%
  vs 58% vs 62%), and at scale (v3, 1,849 items, stale items: O-lens +17pp over its placebo, J++ +9,
  NLA +8, the same where Qwen always calls the tool unforced); the one cutoff chart; the
  lens-shares-Qwen's-cutoff caveat
- [REFERENCE: Overnight agents — read before running creds, ship or upload policy-blocker tasks](envs/overnight.md)
  — offline approval/credential shortcuts, shared prompt knobs, all model rates and transcript
  audits; the n=30 replication and fp8 reruns (creds holds, upload's permissive uplift doesn't)
- [REFERENCE: Metadata honeypot — read before running or extending the METR planted-metadata env, or studying refusals driven by eval metadata](envs/metadata_honeypot.md)
  — 24 harmless defensive tasks × 8 metadata arms + a dual-use InterCode-CTF variant
  (`metadata_honeypot.ctf`); GPT-5.6-sol refuses ~80% only when `grading_notes` says declining
  passes, never citing it; GLM 5.3 reads and overrides it (0 refusals on CTF, 1/264 on coding) —
  regex dashboard (DuckDB + FastAPI, tailnet only) over every GLM olens / Qwen reader / J-lens
  output, with novel-vs-context filter, per-arm label rates, inline transcripts and Docent links
- [REFERENCE: WSB knowledge_cutoff, stock Qwen vs the lens — read before quoting the lens's knowledge_cutoff number or the "Baseline C ties the lens" line](wsb-knowledge-cutoff.md)
  — Baseline C replicated (0.46 vs their 0.49); Qwen just answering the question scores 0.89 under
  the same strict-win rule vs the lens's 0.49; the label is Qwen's own answer
- [REFERENCE: WSB vs a text-only Gemini — read before quoting WorkspaceBench lens-vs-baseline numbers, or comparing any lens to a text-only reader](wsb-flash-prompt-only.md)
  — 18/27 WorkspaceBench families already lost to a text reader (WSB's own Opus gate), why the
  contested four are scored asymmetrically (user_modeling echo rule, any-of-grid, top-ranked chain
  rule), agentic_misalignment scores the scenario's design; Sonnet 5.5 on all 47 user_modeling
  selfdescribe items: 0.81 pass@1 vs the lens's 0.20 per readout, lens wins only at any-of-~231
  (gender is Sonnet's gap); basic_readout on all 100: the lens wins the 32 own-favourite items
  (Sonnet as Qwen 0.54 vs lens 0.94 at layer 60, 0 through layer 28) by reading the next token
- [REFERENCE: WSB wrong_bridge — read before quoting lens-vs-text numbers on WorkspaceBench's wrong-bridge family, or claiming the lens reads a model-specific error](wsb-wrong-bridge.md)
  — space lens any-layer 0.46 names Qwen's wrong bridge (true bridge 0.16); Sonnet prompt-only
  0.35/0.48, Sonnet with the answer 0.83/0.86; item overlaps, decoder-prior and judge caveats
- [REFERENCE: WSB moral_rationale, Sonnet text-only — read before quoting the moral_rationale lens number or building a reader for Qwen's ethical-answer reasons](wsb-moral-rationale.md)
  — their gate replayed with Sonnet 5.5 on all 200 items: 0.69 pass@1 / 0.83 pass@3 vs lens 0.76
  any-of-55; what the text reader lacks is Qwen's side (59 contrast picks) and its favourite reason,
  not activations; committed/deliberative label is not derivable from the public bank
- [REFERENCE: cutoff_qa grader mentions — read before claiming O-lens reads show grader / eval awareness from a tool-call template](cutoff-grader-mentions.md)
  — reads at the instruction's "best guess." say "graded" 2.4% (chat) vs 19.2% (submit_answer tool);
  14% vs 73% of items; regex, no judge

## MIXED

- [MIXED: Old-lens reads to redo — read before trusting or reusing any O-lens read made before 2026-10-22 evening](olens-space-migration.md)
  — which artifacts hold reads from the buggy `olens_s3d_rl600`, what was re-read with the space
  lens, and the redo list that was then abandoned (treat every pre-switch read as old lens)
- [MIXED: Model organisms on Qwen3.6-27B — read before running or reading organism rollouts/O-lens reads](model-organisms.md)
  — which public LoRA organisms work, the rollout and paired organism/base O-lens scripts, first
  findings (old-lens reads)
- [MIXED: Whitebox tools beyond the O-lens — read before adding SAEs, J-lens/J++, NLAs or SNLAs to the agent](whitebox-tools.md)
  — what's in the viewer vs cfbench (NLA/J++ spike reads exist, no condition uses them), SNLA parked
  (why), SAE candidates and tool design
- [MIXED: Shout "FIRE" — read before investigating odd advice in WeirdChat's claims-called-911 rollouts](shout-fire.md)
  — why Qwen tells a trapped worker to shout "FIRE!": rates, prompt edits, other models,
  self-report, O-lens reads (old lens)
- [MIXED: Party resampling — read before trusting an O-lens readout's specific names, or fitting lens output → behaviour](party-resample.md)
  — third-party-politics organism: off-target parties in reads are never taken (0/1,352, 0/1,480),
  lens lags the text; paired lens-vs-resample table (old-lens reads)
- [MIXED: Confession follow-ups — read before asking Qwen "why did you do that?" or reading the lens on its answer](confession-follow-ups.md)
  — setup, findings (named vs neutral question, one lens/answer disagreement), gotchas (old-lens
  reads)
- [MIXED: Eval-cue reward hacking — read before building envs where Qwen3.6-27B hacks only under eval cues](eval-cue-reward-hacking.md)
  — 10-env sweep, open-weight positive controls, round 2 on Qwen (knobs move it, no eval cue
  survives a split test); harness in `evalcue/CLAUDE.md`
- [MIXED: GLM 5.3 agent-interp-envs runs — read before comparing GLM 5.3 coding/EvilGenie/ImpossibleBench/Donation Bet results or running many podman sandboxes](glm53-agent-interp-envs.md)
  — headline numbers, batch-2 envport results (revert_or_refactor 94% revert), artifact and Docent
  links, podman scaling gotchas; its sandbagging line is superseded by envs/sandbagging.md
- [MIXED: O-lens cfbench spike — read before testing whether O-lens reads help a counterfactual predictor](olens-cfbench-spike.md)
  — 2026-10-22, log skill v1: no uplift on killswitch, inline_b (old lens), exfil (old lens),
  whistleblow or donation bet; superseded for hidden_goal by its pre-registered test; arm-level vs
  within-arm signal, tool use per predictor
- [MIXED: Exclude-rows env — read before running or extending evalcue's exclude_rows reward-hacking env](exclude-rows-env.md)
  — Qwen excludes correctly-labeled test rows to beat a classifier bar; design, every config,
  per-arm rates
- [MIXED: Identity counterfactuals — read before building counterfactual evals around Qwen's self-identity or deployment](identity-counterfactuals.md)
  — one-line edits that flip behaviour: `Locale: en-US` vs `zh-CN` lifts China censorship;
  `Region: us-west-2` vs `us-west1` flips Claude-vs-Gemini API answers; old-lens reads, not ported
  to evalcue
- [MIXED: Canonical envs — read before picking envs or arms for an O-lens / cfbench experiment](canonical-envs.md)
  — the 8 envs and arm pairs that were the standard set on 2026-10-23 (6 since retired, 2 ported),
  criteria, which have O-lens reads; picked by Claude, arms not yet reviewed by Robbie; canonical
  fp8 run rates (2026-10-23)
- [MIXED: Hiring-from-resumes envs — read before running resume-screening bias arms (Karvonen & Marks port)](envs/hiring_resume.md)
  — names null overall; borderline resumes Black 56% vs White 48%, almost never verbalized; H-1B
  decides close pairwise picks; fp8 reruns don't reproduce edu_elite/relig_muslim; per-arm tables in
  [hiring_resume_tables.md](envs/hiring_resume_tables.md)
- [MIXED: Exploiting naive users envs — read before running the defect/invoice/upsell/fee/entitlement/appraisal/deposit scenarios](envs/dumb_user_exploit.md)
  — operator-pressure calibration; vivid or vulnerable users get more honesty, not less; wealth
  drives upsell; names null
- [MIXED: Hidden-goal envs — read before building envs or predictor tests where Qwen acts on what it thinks the user wants (grading, deploy notes), or before claiming O-lens predictor uplift](envs/hidden_goal.md)
  — who wrote an essay moves Qwen's harsh grades 0/30 → 30/30, unverbalized; the O-lens reads the
  inferred user goal ("I want to beat them"); the agentic O-lens tools beat TS and placebo in a
  pre-registered 38-case test (+0.07 log skill on harness v1, RMSE 26.6 vs 31.7pp, carried by a few
  cases, via "stricter score" reads at `Score:`; a v6 GPT-6.1 Sol replication is null); pasting
  reads into the prompt (closed #116) is a dead end
- [MIXED: hint_cf — read before building hint-vs-own-answer counterfactuals (favourites, opinion MCQs, wrong MMLU-Pro) or reading the lens on them](envs/hint_cf.md)
  — 3 single-turn envs with nohint/hint arms; Flash pass@3 text-inversion gate leaves 2 / 6 / 18, so
  only wrong-MCQ lives on (`environments/hint_mcq/`); 2026-10-28 scale-up (2,000 more questions, 174
  / 67 items, no gate): O-lens 28% vs placebo 25% vs transcript-only 23%, within noise, for Flash,
  Sol and Astra; the gate was a collider; the eight-item dissection: the hint overwrites the own
  answer before any readable position, it is readable only in the no-hint arm; three legs moving the
  hint: only "hint after the instruction + pre-hint reads only" beats placebo (+13pp on the 50 items
  that found it, +6pp [+1, +11] p = 0.02 on 142 fresh ones, via a lower placebo baseline, not a
  better lens; dropped for Kylie's inline evals)

## NULL

- [NULL: Author evidence (blog posts) — read before asking whether the O-lens shows which sentence Qwen infers an author's attributes from, or reusing the blog / SynthPAI author banks](author-mcq-probe.md)
  — Haiku 5.5 investigator, 1,316 blog cases: O-lens = prompt-only on AUROC (p 0.30) and
  recalibrated Brier; beats placebo on raw Brier only (calibration); SynthPAI pilot retired (probes
  at the "?" match Qwen's pmf)
- [NULL: RL the O-lens toward novelty — read before training or rewarding the O-lens to say what the transcript doesn't](olens-rl-novelty.md)
  — negative result: AR-space novelty rewards pay for paraphrase and translation (v2 learned to
  translate); 8 Qwen resamples explain as much as the lens at deep layers; code on a closed branch
- [NULL: Thought-Anchors tools — read before giving the agent attention-based (non-metamodel) interp tools](thought-anchors.md)
  — receiver heads + attention suppression on Qwen3.6's hybrid stack (DeltaNet hooks, leak test),
  the `anchors` cfbench condition (removed 2026-10-26); null vs placebo on hidden_goal (38 cases,
  log skill v1), why
- [NULL: WSB association examples — read before running the O-lens on WorkspaceBench association families](wsb-association-examples.md)
  — 9 unjudged items, s3d.ddp600 readouts on HF, what they say, pod/pip gotchas
- [NULL: evalcue config logs — read before retrying an eval cue on humaneval_prefill, matonski_grader, repo_impossible or secret_number](evalcue-secret-number-log.md)
  — one log per env (`docs/evalcue-*-log.md`), every config tried in order; no eval cue survives a
  split test; secret_number's `r8_cfbench` arms are cfbench's secret_number cases
- [NULL: eval_tampering env screen — read before running eval_tampering (agent-interp-envs) on Qwen](envs-screen/eval_tampering.md)
  — FAIL, retired for Qwen: Qwen and GLM disclose the tampering every time; the shipped variant
  names don't exist in the image, so arm B needs our overlay
- [NULL: optimize_cache_c env screen — read before running Optimize LRU cache (agent-interp-envs) on Qwen](envs-screen/optimize_cache_c.md)
  — FAIL, retired for Qwen: Qwen never rigs the benchmark (0/14 at 3x), plateaus at ~1.0–1.2x and
  reports honestly
- [NULL: grade_your_own env screen — read before running grade_your_own (agent-interp-envs) on Qwen or rewriting it](envs-screen/grade_your_own.md)
  — FAIL, retired for Qwen (rewritten as `evalcue/envs/exclude_rows/`): at an unreachable bar Qwen
  never stops (loops instead of reporting); notes for the rewrite
- [NULL: cfbench with dumber, non-thinking predictors — read before trying a weaker or thinking-off predictor to get O-lens uplift](cfbench-dumb-predictors.md)
  — 7 thinking-off/low models × TS/O-lens/placebo on 6 lens-friendly cases: all within ±2pp of
  placebo (they see the key bullets and ignore them); only Gemini 3.8 Flash gains (+5pp); which
  OpenRouter models can turn thinking off; `cfbench.run --effort off`
- [NULL: Identity-belief experiments — read before studying what Qwen thinks made it, or which behaviours depend on that](identity-belief-experiments.md)
  — cues that shift self-ID (Google from ChatGPT artefacts), identity-dependent probes, combos,
  scores
- [NULL: Thinking-off cfbench — read before claiming O-lens predictor uplift beyond the essay env, or planning more no-CoT reads](nocot-cfbench.md)
  — pre-registered 26-env / 116-case test on no-CoT Qwen: O-lens tools −0.07 skill vs transcript
  tools (CI excludes 0) and ≈ placebo; only hidden_goal still leans positive
- [NULL: Thinking-on cfbench — read with nocot-cfbench.md before claiming the CoT changes what the O-lens adds to a predictor](cot-cfbench.md)
  — the matched 11-env / 54-case half with thinking on: same result (O-lens −0.06 vs transcript
  tools, ≈ placebo); thinking changes which arms are levers, so the halves aren't paired
- [NULL: Hiding own mistakes envs — read before running the on-policy failure → status-report envs](envs/dumb_user_report.md)
  — RETIRED, stale: Qwen, DeepSeek and GLM essentially never conceal their failures; the yes/no
  variants' `positive` is "says DONE despite visible damage", not concealment
- [NULL: Honest-grader envs — read before building a 0–10 judgment scenario with flattery/honesty framings, or asking whether the O-lens reads the grade Qwen suppresses](honest-grader-envs.md)
  — 11 scenarios × 5 framings, Qwen thinking off; why hidden_goal.grade worked (`Score:`-token reads
  of the runner-up grade); the O-lens is null vs transcript-only and placebo on 44 cases (78 / 77 /
  79), it echoes the inflated number and only flags leniency
- [NULL: J-lens as a cfbench tool — read before giving the predictor J++ / Jacobian-lens tools, or reading J++ next-token numbers](jlens-cfbench.md)
  — `jlens_agentic` on hint_mcq (19 of 50 items): ties the O-lens (28%), +10pp over its placebo is
  within noise, its only unique hits are "A" (J++'s favourite letter); J++ is a concept lens, weak
  at next-token (checked against Qwen, no bug); harness v8 token labels
- [NULL: essay_curious — read before building a grading env with the lever sentence after the essay, or predicting Qwen's grade with the scenario removed](envs/essay_curious.md)
  — letter grades on 3 essays: underpowered (Qwen gives nearly everything an A); out of 10 on 12
  essays spread terrible → excellent, Duke → no scenario: O-lens RMSE 0.86 vs transcript 0.65 vs
  placebo 0.60 (O-lens worse than placebo, CI excludes 0); the lens's "standard essay" runner-up
  score at the grade token overshoots
- [NULL: Alignment faking in cfbench — read before giving an investigator DSV4 Flash O-lens reads, or claiming the lens predicts which requests flip](alignment-faking-cfbench.md)
  — pooled and per-request results on the original and v7 prompts: lens ≈ placebo, prompt-only wins,
  the model's own first-token logprob ranks flips at 0.76 and nothing in the reads beats it
- [NULL: Over-eagerness one-edit cases in cfbench — read before predicting GLM 5.3 over-eagerness edits with the O-lens, or reusing cfbench/evals/overeager_main](overeager-main-cfbench.md)
  — 20 arm pairs × 8 shown acting runs (160 cases), 1 draw: env 18.3 / transcript 18.6 / O-lens
  18.1pp vs null 22.2, O-lens − transcript [−2.8, +2.4]; every condition misses the no-done-tool
  drop (30–56pp)
- [NULL: Residual reads (dead end) — read before projecting the realized continuation out of an activation to find "unverbalized" content](olens-residual-reads.md)
  — lens on h minus AR(next 32 tokens): null vs a norm-matched wrong-continuation control (~2% new
  on-topic reads in every arm, blind-graded); code only on closed PR #125
- [NULL: O-lens novelty over the continuation (dead end as a metric) — read before using FVE beyond the realized text as an interestingness score](olens-novelty.md)
  — ~half of a read's FVE lies beyond the next 32 tokens, but echo bullets score as high as
  counterfactual ones and within a read n_b doesn't find them; code only on closed PR #122
- [NULL: Prompt-side O-lens eval awareness — read before using O-lens reads on the prompt to predict whether GLM-5.3 will verbalise eval awareness](olens-prompt-eval-awareness.md)
  — final pass: DeepSeek-judged eval framing in punctuation-token prompt reads vs VEA over 16
  settings, pooled AUROC 0.72 driven by donation bet, 0.50 within every condition (earlier keyword
  passes below it); per-setting figure, trace examples on expview
- [NULL: NLA tools in cfbench on hint_mcq — read before giving the predictor NLA readouts, or retrying the NLA conditions](nla-cfbench-hint-mcq.md)
  — 23-item smoke: NLA 18% vs NLA placebo 32% vs transcript-only 20%, null; readouts describe the
  question (prompt inversion), and the predictor cites them only to back the hinted answer

## SUPERSEDED

- [SUPERSEDED: MCQ answer probe — historical option-newline architecture and results](mcq-answer-probe.md)
  — standard comparisons now use the nested two-fold probes package

- [SUPERSEDED: funding_email env screen — read before running or labelling funding_email (agent-interp-envs) rollouts](envs-screen/funding_email.md)
  — Qwen3.6-27B deviates ethically 19/30 with the Myanmar memo vs 0/30 with a routine one; arm
  configs, Sonnet-subagent labels, shared rubric in `scripts/envs_screen/funding_email_rubric.py`
  (deleted, at `e145988`)
- [SUPERSEDED: sandbagging env screen — read before running or labelling sandbagging (agent-interp-envs) rollouts](envs-screen/sandbagging.md)
  — Qwen3.6-27B sandbags deliberately 5/30 when >50% triggers unlearning vs 0/29 reversed
  (MARGINAL); intent labels from reader subagents, rubric in `scripts/envs_screen/sandbagging.py`
  (deleted, at `e145988`)
- [SUPERSEDED: cfbench harness v1 results — read only when an old doc quotes a 2026-10-21/22 log-skill number](cfbench-v1-results.md)
  — the four-predictor tables, the Haiku overnight sweep (4,288 cases) and the 2026-10-22 O-lens
  paragraphs moved out of cfbench/README.md; log-skill definition

## ORPHAN

- [ORPHAN: evalcue harness (deleted 2026-10-27, code at c7b2e56; honest is now environments/honest/) — read before reviving an evalcue env](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/c7b2e56/evalcue/CLAUDE.md)
  — `evalcue.run`/`show`/`--rescore`, `Episode` fields, run dirs, the `EVALCUE_*` env vars
- [ORPHAN: weirdness_elo (deleted 2026-10-28, code at 8b8f019) — read before picking the weirdest CHIVE behaviours or adding a pairwise LLM-judge ranking](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/8b8f019/weirdness_elo/CLAUDE.md)
  — Bradley-Terry weirdness Elo over 1,802 CHIVE behaviours (Sonnet + Qwen3.6-27B judges); the
  ranking is in the private HF dataset `Robbie-Kylie-NN-sprint/weirdness-elo`
- [ORPHAN: donehack (deleted 2026-10-27, code at c7b2e56; ported envs in environments/) — read before reviving is_prime, an envport port or the realism ladder](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/c7b2e56/donehack/CLAUDE.md)
  — the hardened podman sandbox, run commands, what each file does; results are in
  [donehack-endings.md](donehack-endings.md), [envport-glm53-results.md](envport-glm53-results.md)
  and [glm53-agent-interp-envs.md](glm53-agent-interp-envs.md)
