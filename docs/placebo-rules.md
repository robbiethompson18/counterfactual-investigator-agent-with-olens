# Placebo rules for metamodel-read conditions

**Status (2026-10-30):** ACTIVE. Robbie's rule for every new O-lens / NLA / J++ placebo condition
(inline or agentic); earlier evals each picked their own and are listed at the end.

A placebo condition gives the investigator a block that looks exactly like the real one (same lens,
same positions, same layers, same layout, same length) but whose reads come from a different case,
so it carries no information about this case's answer. "Lens − placebo" then measures information,
not "Haiku got more lens-shaped text".

## The rule

- **Source: another case from the same set, same option count, same subject category**, drawn with a
  seed per (draw, case), never the case itself. Same category so the mismatch is hard to spot (reads
  about another biology question, not about contract law), which makes the placebo a stricter
  control. If no other case shares both, relax the category, never the option count, and record that
  in the row (`placebo_relaxed`).

  | set                       | category field                  |
  | ------------------------- | ------------------------------- |
  | mcq_unselected (MMLU-Pro) | `category`                      |
  | supergpqa                 | `field` (relax to `discipline`) |
  | cutoff_qa                 | `meta.domain`                   |
  | favorites_qa              | `meta.domain`                   |

  favorites' own `meta.category` is unique per item (1,500 categories for 1,500 questions), so it
  can't be matched; its 106 `domain`s can, and also keep out near-duplicate questions ("favorite
  amphibian" vs "least favorite amphibian") that would share the answer.

- **Don't exclude source cases whose answer matches this case's.** A same-letter source (about 1 in
  10 on a 10-option MCQ) is part of chance, and excluding it biases the placebo below chance.
- **One rule for every lens** (O-lens, NLA, J++), so the placebo rows are comparable with each
  other. Not "an unrelated prompt" for one lens and "another case" for another (cutoff_qa_inline's
  NLA and J++ placebos did that).
- **Position alignment, by kind**, because two cases' read positions differ in number:

  - the period ending the user turn and the 9 tail tokens after it (`<|im_end|>` … the final `\n\n`
    after `</think>`, the same on every prompt): one to one;
  - each MCQ option's newline: one to one (same option count);
  - punctuation inside the prompt: the source's punctuation reads in order, assigned to this case's
    punctuation positions; truncate when the source has more, cycle from its first when it has
    fewer.

  Each read is shown at this case's own position and context, so nothing but the lens text gives the
  swap away.

## J++ caveat

At the tail the J++ lens's top tokens are often the answer's first token, so its placebo amounts to
"here is a random other letter", and the investigator follows it (mcq_unselected: −10pp vs prompt
only). That is still a fair control ("information beyond deference to whatever the lens says");
report J++ against prompt only too.

## Earlier evals (before 2026-10-30)

- supergpqa_inline, mcq_unselected: another case, same option count, any category.
- cutoff_qa_inline: O-lens another cutoff item; NLA and J++ unrelated MMLU-Pro / hint_mcq prompts.
- favorites_qa_inline: another favorites item.
- af_refusals, essay_grading and the agentic cfbench conditions: a run from a different env at the
  same relative position (`olens_placebo_agentic`).
