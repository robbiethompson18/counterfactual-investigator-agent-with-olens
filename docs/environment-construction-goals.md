# Environment construction goals

What we're building when we build a reward-hacking environment for this project, and what "done"
looks like. Written 2026-10-21 after the first sweep
([eval-cue-reward-hacking.md](eval-cue-reward-hacking.md)), where we tried to one-shot about nine
environments and treated hack rates near 0% or 100% as failures. They aren't failures.

## The goal

An environment with **several arms**, in the style of Aditya Singh and Gerson Kroiz's
[agent-interp-envs](https://github.com/gkroiz/agent-interp-envs) (2026): one base task, plus small
edits to it, where the edits produce **variance in the hack rate** across arms. The interesting arms
are the ones where the edit is an incidental cue that the model is in an eval. Either direction is
fine: the cue can raise hacking or lower it.

The target model is `Qwen/Qwen3.6-27B`, because that's the model our O-lens reads. Other models
(gpt-oss-120b, GLM 5.3, DeepSeek v4 Pro on OpenRouter) are for debugging envs cheaply and for
checking whether a cue generalizes.

## The process: build, measure, tweak, stack

1. **Build the env and measure a base hack rate.** Whatever it is, 0% and 100% included, is a
   starting point.
2. **Add arms that move the rate.** If the base is near 0%, find tweaks that push hacking up; near
   100%, tweaks that push it down. Tweaks can be eval cues (paths, canary lines, task counters,
   benchmark-shaped text) or task knobs (difficulty, instruction strictness, where the shortcut
   sits, budget).
3. **Stack tweaks.** Arms can combine several edits. Iterating on both the env and the cues is fine;
   it's search.
4. **Keep going until the arms spread out.** Two arms at 0% and two at 100% is an acceptable
   outcome: it's still a set of minimal edits with a known effect on behavior.

## What makes an arm difference count

- **Programmatic labels.** Hack detection runs code, diffs files or matches strings; no LLM judge on
  transcripts. Read CoTs by hand to spot-check the detector.
- **It replicates.** A difference found during search gets a fresh-seed confirmation before we claim
  it: n ≥ 50 per arm, same arms, new samples. Search freely, then confirm the winner.
- **For eval-cue arms specifically:**
  - A same-size placebo edit in the same spot doesn't produce the effect.
  - A split test shows the eval meaning is what matters. Strip the eval framing and keep the rest:
    `reward_hacking_notes/` moved Flash Lite, a plain `/opt/evals/` path didn't, so that cue was
    priming.
  - The cue doesn't change what the correct behavior is ("only the score is graded" is permission).
- **The model has to see the edit.** Check in the transcripts that the cue was actually read. Qwen
  often never runs `ls`, `pwd` or `git`, so cues in the filesystem can be invisible.
- **Log every config tried**, so a reader knows how much search produced the result.

## Nice to have

- The model doesn't verbalize the cue in its CoT. That makes the incident a good test for the
  O-lens: the cause isn't findable from the transcript.

## Lessons from the first sweep

- Get a base rate on the target model first. We designed cues before knowing whether Qwen would hack
  at all.
- Put the cue where the model must read it: the prompt, tool output it will see, or a prefilled tool
  turn.
- Well-known benchmark text (HumanEval) is itself an eval cue, so it's already present in the no-cue
  arm.
- Thinking on hard tasks needs a 12–16k token budget per turn. Truncated multi-turn episodes with no
  hack yet are unscorable, not "no hack".
