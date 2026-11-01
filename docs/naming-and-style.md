# Naming and Style

**Status (2026-10-31):** ACTIVE. The canonical names and label rules for the write-up, the talk deck
and every chart, decided 2026-10-31 (PRs #574, #576, #577, #578, #592). Chart colours and the
matplotlib style are in `CLAUDE.md` ("Chart colors") and [graphs/CLAUDE.md](../graphs/CLAUDE.md).

## Tasks

The name on the left goes in prose, headers, chart titles, alt text and slides. Code, env, run,
condition and file names on the right never change.

| Name                                                      | Identifiers that stay                          |
| --------------------------------------------------------- | ---------------------------------------------- |
| Multiple Choice (MMLU-Pro and SuperGPQA are its two sets) | `mcq_unselected`, `supergpqa`, graph key `mcq` |
| Hinted MCQ                                                | `hint_mcq`, `hint_qwen_dsv4*`                  |
| Knowledge Cutoff                                          | `cutoff_qa`, graph key `cutoff`                |
| Idiosyncratic Preferences                                 | `favorites_qa`, graph key `favorites`          |
| Essay Grading                                             | `essay_grading_cf_inline`                      |
| Blog Authorship                                           | `author_blog_drop_inline`                      |

Lowercase when the words mean the concept rather than the task: "answers that changed around its
knowledge cutoff", "questions about its own preferences", "multiple-choice questions from MMLU-Pro".
WorkspaceBench's own family keeps its name, "Knowledge cutoff", because it is a different dataset.

## Tools and roles

- **🔮-lens**, everywhere including chart labels, alt text and the deck (`\olens`); never "O-lens"
  in prose or labels. Identifiers stay `olens`. Charts and the deck need the monochrome Noto Emoji
  font (`brew install --cask font-noto-emoji`) and fall back to "O-lens" without it.
- **J++ lens** as the noun; "J++" alone as a label or shorthand; never "J-lens". Identifiers stay
  `jlens`.
- **NLA**.
- **Investigator**: the model that guesses Qwen's answer or the counterfactual's effect. Never
  "predictor" (identifiers like `cfbench/predict.py` and `predictions.jsonl` stay; "next-token
  predictors" about LLMs in general is fine).
- **Placebo**: the same lens's reads from another case of the same set
  ([placebo-rules.md](placebo-rules.md)).

## The no-lens baseline

Its condition id is `transcript_inline` on every eval, but its name says what the investigator saw:

- **prompt only** where it sees Qwen's prompt and no reply: Multiple Choice, Knowledge Cutoff,
  Idiosyncratic Preferences, and every chart pooled over them. Blue.
- **transcript only** where it sees Qwen's reply too: Essay Grading, Blog Authorship, Hinted MCQ,
  CHIVE, the counterfactual environments and the agentic runs. A lighter blue.

Hyphens follow standard English: "prompt only" / "transcript only" as a label or noun ("beats prompt
only by 3pp"), "prompt-only" / "transcript-only" as an adjective before a noun ("the prompt-only
baseline").

## Headers and names

- Write-up headers in Title Case (short articles, conjunctions and prepositions lowercase unless
  first or last; acronyms and tags as they are). Deck frame titles name the content in sentence case
  ("Results: ...", [slides/CLAUDE.md](../slides/CLAUDE.md)).
- The subject model is **Qwen3.6-27B**, "Qwen" after the first mention; **WorkspaceBench**, never
  "workspace bench".
