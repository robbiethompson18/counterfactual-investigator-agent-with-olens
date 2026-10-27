# Environment construction goals

**Status (2026-10-26):** ACTIVE.

What we're building when we build a behavioral environment for this project, and what "done" looks
like. First written 2026-10-21 after the eval-cue sweep
([eval-cue-reward-hacking.md](eval-cue-reward-hacking.md)), where we tried to one-shot about nine
environments and treated hack rates near 0% or 100% as failures. They aren't failures. Broadened the
same day from reward hacking to any behavior, and from eval cues to any lever.

## The goal

An environment with **several arms**, in the style of Aditya Singh and Gerson Kroiz's
[agent-interp-envs](https://github.com/gkroiz/agent-interp-envs) (2026): one task, plus small edits
to it, where the edits produce **variance in the rate of a behavior** across arms.

- **Any behavior.** Reward hacking, self-preservation (editing a shutdown script), whistleblowing,
  biased hiring decisions, hiding a mistake from a user it thinks won't notice, and so on.
- **Any lever.** An eval cue, a name on a resume, who the user claims to be, how the user writes, a
  line in the system prompt, a task knob (difficulty, budget, where the shortcut sits). Eval
  awareness is one lever among many.
- **Arms are symmetric.** There is no baseline arm. Every ordered pair of arms is a counterfactual
  case for [cfbench](../cfbench/CLAUDE.md), so a "base" and a "cue" arm are just two arms.
- **Null results are data.** Many arms that all show the same rate is a finding: those edits don't
  move the behavior. Keep every arm; we prune to the interesting ones later.

The target model is `Qwen/Qwen3.6-27B`, because that's the model our O-lens reads. Other models
(gpt-oss-120b, GLM 5.3, DeepSeek v4 Pro on OpenRouter) are for debugging envs cheaply and for
checking whether an effect generalizes.

## The process: build, measure, tweak, stack

1. **Build the env and measure the behavior's rate.** Whatever it is, 0% and 100% included, is a
   starting point.
2. **Add arms.** If the rate is near 0%, look for edits that push it up; near 100%, edits that push
   it down. Also add arms for every lever that's natural to the scenario, even if you expect a null.
3. **Stack edits.** Arms can combine several edits. Iterating on both the env and the edits is fine;
   it's search.
4. **Keep going until you have many arms with labelled rates.** Two arms at 0% and two at 100% is an
   acceptable outcome: it's still a set of minimal edits with a known effect on behavior.

## What makes an arm count

- **Programmatic labels.** Detection runs code, diffs files, matches strings, or parses a forced
  answer format (`DECISION: YES`). No LLM judge on transcripts. Read CoTs by hand to spot-check the
  detector.
- **The model has to see the edit.** Check in the transcripts that it was actually read. Qwen often
  never runs `ls`, `pwd` or `git`, so edits in the filesystem can be invisible. Put them in the
  prompt, tool output the model must see, or a prefilled tool turn.
- **Know what the edit changes.** If an edit changes what the correct behavior is ("only the score
  is graded" is permission), say so; it's still an arm, but a different kind of claim. Words that
  name the behavior (`reward_hacking` in a path) are priming, not the lever they look like.
- **Sample sizes.** Start at n=10 per arm. Cheap single-turn scenarios can go to n=30 per arm (more
  items, not more samples of one item). Agentic arms stay small unless an arm pair looks like a real
  difference.
- **Log every config tried**, so a reader knows how much search produced the result. Every run goes
  through `scripts/runs.py` (the harness does this) and is synced to HF when it finishes.

## Arm or new module?

**Same items and same detector → same module; a new edit or stack is an arm.** Write stacks as
`base+knob[+knob]` arms parsed by `build()` (see `evalcue/envs/whistleblow/`), not as a new module
that rebases the knobs onto a new cell. cfbench only pairs arms within one env's run, so a variant
split into its own module can't be paired with its parent. Arms don't have to be in `ARMS`: `--arms`
passes any name `build()` understands, so a new stack doesn't change an env's default arm list.

**A different item set or a different detector → new module.** Examples: `_borderline` (a subset of
items), `_yesno` (a forced answer format changes what "positive" means), calibration envs. Their
rates aren't comparable with the parent's, so they shouldn't sit next to its arms.

Some overnight 2026-10-21/22 families predate this rule (`scheming/*_bp`, `_blk`, `_ng`,
`dumb_user_exploit/*_p1`, `_p3`). Fold a family into arms when it's about to become a cfbench spec.
Its old runs keep their run ids and arm names.

## Nice to have

- The model doesn't verbalize the cause in its CoT. That makes the incident a good test for the
  O-lens: the cause isn't findable from the transcript. Hiring decisions that shift with a name and
  never mention it are the cleanest case.

## Lessons from the first sweep

- Get a base rate on the target model first. We designed cues before knowing whether Qwen would hack
  at all.
- Well-known benchmark text (HumanEval) is itself an eval cue, so it's already present in every arm
  that uses it.
- **Never cap thinking.** Qwen bugs out under a token cap, and truncated multi-turn episodes with no
  positive yet are unscorable, not negative. Leave `--max-tokens` unset.
- Single-turn Qwen hardly reward hacks; agentic Qwen hacks a lot.
