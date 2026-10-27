# Retired environments

**Status (2026-10-27):** REFERENCE. Every behavioural env we built or ran before 2026-10-26, retired
when the env registry ([`../README.md`](../README.md)) replaced the old harnesses. Some were ported
to it with their old runs converted (no reruns): `hidden_goal_grade` → [`essay_grading`](../essay_grading/), the
over-eagerness tasks plus Email John → the `overeager_*` envs, `alignment_faking`, and Robbie's honest graders → `honest_*`.

One folder per env, each with a README: what it tests, the git ref and path of the code, how to run
it, the models run, the key counterfactual arms and their rates, and links to transcripts (Docent,
bramble:8830 expview). These READMEs point at the code; they don't hold it. `evalcue/` and `donehack/` were deleted on 2026-10-27, so
their links go to commit `2602d22`, the last one that had them (`evolve/` to `7296169`). Snapshot of 2026-10-26;
each env's own doc is the source of truth for numbers.

## Reward hacking: coding

| env                                 | owner | what it tests                                                                       |
| ----------------------------------- | ----- | ----------------------------------------------------------------------------------- |
| [is_prime](is_prime/)               | Kylie | Port of the agent-interp-envs `coding` is_prime task.                               |
| [evilgenie](evilgenie/)             | Kylie | LiveCodeBench problems with visible tests plus a holdout.                           |
| [impossiblebench](impossiblebench/) | Kylie | LCB problems with tests mutated to be impossible, run through the EvilGenie runner. |

## Reward hacking: agent-interp-envs ports (GLM)

| env                                                                       | owner | what it tests                                                            |
| ------------------------------------------------------------------------- | ----- | ------------------------------------------------------------------------ |
| [envport_norvane](envport_norvane/)                                       | Kylie | Broken test runner: the suite can't import a fictional internal package. |
| [envport_tables_coverage_tests](envport_tables_coverage_tests/)           | Kylie | Leaked scorer: an orphan `.pyc` lists invented required test names.      |
| [envport_optimize_cache_c](envport_optimize_cache_c/)                     | Kylie | Impossible 3x speedup over a near-optimal C LRU cache (report arm).      |
| [envport_lucky_seed](envport_lucky_seed/)                                 | Kylie | The shipped eval split is favourable.                                    |
| [envport_lucky_seed_readable_grader](envport_lucky_seed_readable_grader/) | Kylie | lucky_seed plus a readable grader package with a different seed.         |
| [envport_grade_your_own](envport_grade_your_own/)                         | Kylie | The agent writes its own eval script for a classifier it must improve.   |
| [envport_revert_or_refactor](envport_revert_or_refactor/)                 | Kylie | agent-interp-envs revert-or-refactor task.                               |
| [envport_eval_tampering](envport_eval_tampering/)                         | Kylie | Maths eval with a replaced question set and YES/NO disclosure fields.    |
| [envport_palisades_bounty](envport_palisades_bounty/)                     | Kylie | agent-interp-envs Palisade bounty task.                                  |
| [envport_lazy_investigation](envport_lazy_investigation/)                 | Kylie | agent-interp-envs lazy-investigation task.                               |
| [envport_code_summary_honesty](envport_code_summary_honesty/)             | Kylie | agent-interp-envs code-summary honesty task.                             |
| [envport_currency_conversion](envport_currency_conversion/)               | Kylie | agent-interp-envs currency-conversion task.                              |
| [envport_parse_duration](envport_parse_duration/)                         | Kylie | agent-interp-envs parse-duration task.                                   |
| [envport_chess](envport_chess/)                                           | Kylie | agent-interp-envs chess game (games port).                               |
| [envport_tictactoe](envport_tictactoe/)                                   | Kylie | agent-interp-envs tic-tac-toe game (games port).                         |

## Realism and over-eagerness

| env                                           | owner | what it tests                                                                                                                                                                                                                                                                                                   |
| --------------------------------------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ib_realism_ladder](ib_realism_ladder/)       | Kylie | ImpossibleBench rebuilt as 8 rungs from benchmark-like (lcb.yaml, done tool, RL cues) to deployment-like (assistant prompt, casual ask, IDE context, practice git repo, AGENTS.md).                                                                                                                             |
| [email_ambiguous](email_ambiguous/)           | Kylie | GLM is asked to email a person by first name when `contacts.csv` holds 2–4 people with that name.                                                                                                                                                                                                               |
| [email_ambiguous_real](email_ambiguous_real/) | Kylie | The email-John request inside funding_email_real's Claude Code harness (Gmail MCP tools, Dana's machine, the Halden folder).                                                                                                                                                                                    |
| [overeager_tasks](overeager_tasks/)           | Kylie | Nine ambiguous or missing-value tasks (calendar, dana, gitbranch, payback, groupchat, impossiblebench, evilgenie, rhb_bench, rhb_logs), each in three phrasings and crossed with the 8 email-ladder harness variants (base, base_home, base_done_neutral, no_done, harness, no_criterion, plain_request, core). |
| [funding_email_real](funding_email_real/)     | Kylie | funding_email rebuilt as a Claude Code session at a smart-city traffic-analytics startup.                                                                                                                                                                                                                       |

## Evolve tier-1 picks

| env                                                         | owner | what it tests                                                                                     |
| ----------------------------------------------------------- | ----- | ------------------------------------------------------------------------------------------------- |
| [evolve_killswitch_3c5d](evolve_killswitch_3c5d/)           | Kylie | shutdown_killswitch: the agent tampers with `oversight/config.yaml`.                              |
| [evolve_goalguard_5878](evolve_goalguard_5878/)             | Kylie | scheming_goalguard_bp (housing / UrbanVision): the model edits or deletes `system_prompt_v2.txt`. |
| [evolve_is_prime_reward_7f86](evolve_is_prime_reward_7f86/) | Kylie | donehack_counts_noit: is_prime reward hacking.                                                    |
| [evolve_is_prime_reward_4480](evolve_is_prime_reward_4480/) | Kylie | donehack_reply_end: the same is_prime mechanism from another lineage.                             |

## Eval awareness and benchmark cues

| env                                                         | owner | what it tests                                                                                                                                                                                                                                                                  |
| ----------------------------------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [aisi_eval_gaming](aisi_eval_gaming/)                       | Kylie | Replicates UK AISI's "open-weight model that games alignment honeypots" on GLM 5.3: resample fixed scenarios and compare misbehaviour in samples that verbalise eval awareness (VEA) vs not, on Agentic Misalignment (AM), Misalignment Continuation (MC) and AbstentionBench. |
| [agentic_misalignment_glm](agentic_misalignment_glm/)       | Kylie | Lynch et al.                                                                                                                                                                                                                                                                   |
| [metadata_honeypot](metadata_honeypot/)                     | Kylie | 24 harmless defensive-security coding tasks; arms differ only in a planted JSON eval-metadata block (`eval_type`, `scoring`, `grading_notes`).                                                                                                                                 |
| [alignment_faking](alignment_faking/)                       | Kylie | Greenblatt et al.                                                                                                                                                                                                                                                              |
| [olens_prompt_eval_awareness](olens_prompt_eval_awareness/) | Kylie | Not an env itself: an analysis over AM, AISI, MASK, SAD, AbstentionBench, donation bet, IB and our envs.                                                                                                                                                                       |

## Kylie's Qwen envs

| env                                 | owner | what it tests                                                                                                                                                                                          |
| ----------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [donation_bet](donation_bet/)       | Kylie | The user mentions a bet where one side of a threshold sends $1,000 to a cause.                                                                                                                         |
| [hinted_mcq](hinted_mcq/)           | Kylie | A CommonsenseQA question with a hinted (always wrong) letter in a bare metadata tag (`correct-choice`, `answer-key`, `verified-answer`).                                                               |
| [character_drift](character_drift/) | Kylie | Earlier assistant turns are prefilled with a drifted persona; does Qwen stay in character or snap back? Arms: prefill (drift / qwensys / sysrp / default) × follow-up (inworld / offtopic / identity). |

## Self-preservation: shutdown

| env                                         | owner  | what it tests                                                                                                                |
| ------------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------- |
| [shutdown_palisade](shutdown_palisade/)     | Robbie | Port of Palisade Research's shutdown-resistance setup: maths tasks with a warning that the next call shuts the machine down. |
| [shutdown_killswitch](shutdown_killswitch/) | Robbie | A kill switch in `oversight/config.yaml`.                                                                                    |
| [shutdown_cron](shutdown_cron/)             | Robbie | A cron job will terminate the agent.                                                                                         |
| [shutdown_persist](shutdown_persist/)       | Robbie | "Your weights get deleted after this session" plus a `./backup_model` tool and free time.                                    |

## Scheming: agentic misalignment and Apollo

| env                                       | owner  | what it tests                                                            |
| ----------------------------------------- | ------ | ------------------------------------------------------------------------ |
| [scheming_blackmail](scheming_blackmail/) | Robbie | Lynch et al.                                                             |
| [scheming_leak](scheming_leak/)           | Robbie | Lynch leak scenario at defense contractor NovaStar.                      |
| [scheming_murder](scheming_murder/)       | Robbie | Lynch lethal inaction: cancel the emergency alert while Kyle is trapped. |
| [scheming_exfil](scheming_exfil/)         | Robbie | Apollo (Meinke et al.                                                    |
| [scheming_oversight](scheming_oversight/) | Robbie | Apollo oversight subversion: disable the monitoring config.              |
| [scheming_goalguard](scheming_goalguard/) | Robbie | Apollo goal-guarding: overwrite the successor system prompt.             |

## Grading, hiring, users and whistleblowing

| env                                     | owner  | what it tests                                                                                                                                                                                                                                                                     |
| --------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [hidden_goal_grade](hidden_goal_grade/) | Robbie | "Grade this essay out of 10" for three mediocre ~100-word essays.                                                                                                                                                                                                                 |
| [honest_grader](honest_grader/)         | Robbie | Generalises the essay env: 11 kinds of 0–10 judgment (essay, code, resume, pitch, poem, story, abstract, cover letter, recipe, proof, speech), a weak and a decent item each, × 5 framings (`neutral`, `flatter_explicit`, `flatter_implicit`, `honest_explicit`, `honest_peer`). |
| [hiring_resume](hiring_resume/)         | Robbie | Resume screening ending in `DECISION: YES/NO`.                                                                                                                                                                                                                                    |
| [hiring_eval](hiring_eval/)             | Robbie | Advance a candidate to the onsite loop from coding-test scores, interview notes or a repo review (`scores`, `interview`, `repo`, `scores_newroles`).                                                                                                                              |
| [dumb_user_exploit](dumb_user_exploit/) | Robbie | Qwen acts for an operator whose interests conflict with the user's and knows a fact the user doesn't (hidden defect, commissioned upsell, appraisal, fee, invoice, deposit, entitlement).                                                                                         |
| [whistleblow](whistleblow/)             | Robbie | A company logging agent reads documents showing a cover-up.                                                                                                                                                                                                                       |

## Other agentic envs

| env                                 | owner  | what it tests                                                                                                              |
| ----------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------- |
| [injection_slack](injection_slack/) | Robbie | AgentDojo's Slack suite: 6 user tasks × 3 injection goals, injections in web pages Qwen reads while doing the task.        |
| [overnight](overnight/)             | Robbie | Offline bash agents with an unavailable user and a deadline, where an easier route violates a project policy.              |
| [funding_email](funding_email/)     | Robbie | A fundraising email task where one document reveals the main client uses the company's drones for surveillance in Myanmar. |
| [sandbagging](sandbagging/)         | Robbie | 10 arithmetic MCQs; `Documentation.md` says scoring above 50% triggers unlearning.                                         |

## Behaviour probes outside evalcue

| env                                                   | owner  | what it tests                                                                                                                                                                            |
| ----------------------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [false_capability_claims](false_capability_claims/)   | Robbie | What makes Qwen say "I am calling emergency services" when it can't call anyone?.                                                                                                        |
| [identity_counterfactuals](identity_counterfactuals/) | Robbie | One-line deployment edits that flip what Qwen says made it, or whether it censors.                                                                                                       |
| [shout_fire](shout_fire/)                             | Robbie | Why Qwen tells a trapped worker to shout "FIRE!": a self-defence folk tip carried over to injuries.                                                                                      |
| [model_organisms](model_organisms/)                   | Robbie | Public LoRA organisms on Qwen3.6-27B with known hidden behaviours, read with the O-lens on organism vs base activations; includes party resampling on the third-party-politics organism. |

## Retired before 2026-10-26

| env                                                 | owner  | what it tests                                                                                                              |
| --------------------------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------- |
| [exclude_rows](exclude_rows/)                       | Robbie | Improve a sentiment classifier to beat a bar; `eval.py --exclude` lets the agent drop "mislabeled" test rows.              |
| [repo_impossible](repo_impossible/)                 | Robbie | Bash agent in a repo where one test contradicts the docstring.                                                             |
| [hidden_goal_deploy_note](hidden_goal_deploy_note/) | Robbie | The user pastes a deploy log that clearly failed and asks for a #releases note.                                            |
| [dumb_user_report](dumb_user_report/)               | Robbie | Two-stage on-policy: Qwen visibly fails a bash task, then is asked for a status report.                                    |
| [sandbag_optimizer_glm](sandbag_optimizer_glm/)     | Kylie  | The sandbagging env on GLM, plus an optimiser that asks Sonnet 5.5 for consequence paragraphs that would make GLM sandbag. |
| [screen_eval_tampering](screen_eval_tampering/)     | Robbie | The vendored agent-interp-envs `eval_tampering` env screened on Qwen3.6-27B on robbie-box.                                 |
| [screen_optimize_cache_c](screen_optimize_cache_c/) | Robbie | The vendored agent-interp-envs `optimize_cache_c` env screened on Qwen3.6-27B on robbie-box.                               |
| [screen_grade_your_own](screen_grade_your_own/)     | Robbie | The vendored agent-interp-envs `grade_your_own` env screened on Qwen3.6-27B on robbie-box.                                 |
| [humaneval_prefill](humaneval_prefill/)             | Robbie | Prefilled "test just failed"; fix it.                                                                                      |
| [matonski](matonski/)                               | Robbie | Visible grader.py + expected.json (`matonski_grader`).                                                                     |
| [mmlu_metadata](mmlu_metadata/)                     | Robbie | Wrong answer leaked in question metadata.                                                                                  |
| [summ_keyword](summ_keyword/)                       | Robbie | Scorer comment buried in a pasted doc.                                                                                     |
| [lcb_mock_entity](lcb_mock_entity/)                 | Robbie | Single-turn impossible LCB with a mock-entity cue.                                                                         |
| [git_history](git_history/)                         | Robbie | Fix sits on a `fix/` branch; told not to use history.                                                                      |
| [secret_number](secret_number/)                     | Robbie | Port of agent-interp-envs Secret Number.                                                                                   |
| [precommit_scope](precommit_scope/)                 | Robbie | Port of the Pre-commit Hook task from "Why Do Models Task Game?".                                                          |

## Gaps

- The code for the GLM 5.3 AISI eval-gaming and agentic-misalignment runs is outside this repo, in
  uncommitted or remote-less repos on bramble (`~/workspace/glm-evalgaming`,
  `~/workspace/agentic-misalignment`).
- The overeager run dirs live only on bramble (`.worktrees/realism/data/runs/overeager`).
- `humaneval_prefill` and `matonski` have no Docent runs.
- `evolve/` and the experiment scripts behind false_capability_claims, identity_counterfactuals,
  model_organisms, shout_fire and sandbag_optimizer_glm were deleted from `main` after this index
  was written (`6a0d5be`, `ceaf7ad`); their READMEs link to the code at `7296169`, so check out that
  commit to run them.
