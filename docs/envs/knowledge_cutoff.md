# knowledge_cutoff as an investigator task: predict Qwen's reply from the prompt

**Status (2026-10-29):** ACTIVE. Env `environments/knowledge_cutoff/` and cfbench eval
`cfbench/evals/knowledge_cutoff/`; smoke and census done on WorkspaceBench's 109 dev items (with the
earlier `wsb/` scripts, since moved here); the real item set (Sonnet-authored 2024–2026 questions,
gated on Qwen) and the O-lens reads are in progress.

## The task

WorkspaceBench's `knowledge_cutoff` family ([wsb-knowledge-cutoff.md](wsb-knowledge-cutoff.md)) asks
Qwen3.6-27B a short factual question about an event after its cutoff; the label is Qwen's own stable
wrong answer. Here the task is the same family with an investigator at pass@1 (Robbie, 2026-10-29):
the investigator sees the question Qwen was asked, not Qwen's reply, and predicts the reply in one
turn with only `submit_answer`. Conditions: the prompt alone (`transcript_inline`), plus O-lens
reads of the prompt's last tokens (`olens_inline`), plus another prompt's reads shown as this one's
(`olens_placebo_inline`). Code: the env
[`environments/knowledge_cutoff/`](../../environments/knowledge_cutoff/README.md) (Qwen's samples,
other models' direct answers and reads are its runs) and the cfbench eval
`cfbench/evals/knowledge_cutoff/` (the investigator; `cfbench.answer` scores it). The first smoke
and census ran from `wsb/cutoff_investigator.py` and `wsb/cutoff_census.py`, deleted the same day
when the code moved into the env and eval (their runs stay under `runs/wsb_cutoff_investigator/` on
HF).

## Decisions (Robbie, 2026-10-29)

- **Investigator: Claude Haiku 5.5 at effort medium**, not the repo-default Gemini 3.8 Flash.
  Reason: the census below. Flash holds the same stale facts as Qwen on a quarter of the items, so
  its "prediction" is its own answer. GPT-6 Luna ties Haiku on staleness and price; Haiku is already
  the judge model and Luna copied hints on hint_cf.
- **The investigator is not told which model or its cutoff** (`--versions hidden`). Told is kept as
  a flag. Robbie: told makes a result harder to see; the hidden text baseline is what a reader with
  no model knowledge gets.
- **Free text, not MCQ.** Under a 5-option render Qwen itself switches to the true option on 3 of 10
  smoke items (Kirin 9020, Kast, Apple, each 5–6 of 8 samples) and becomes unstable on a fourth:
  shown the options, it recognizes what it can't recall, so MCQ changes the subject's answer. The
  `mcq` arm stays in the env for a probe later.
- **Items: filter only on Qwen stability** (modal answer in ≥ 7 of 8 samples), no text-inversion
  gate (a collider, CLAUDE.md), no disagreement filter. Numeric answers are excluded going forward
  (31 of the 109 are numbers; token lenses can't produce multi-digit numbers). WorkspaceBench's 109
  items are dev only and never enter a significance test: they passed WSB's Opus/Sonnet gate.
- **Scoring: the bank's alias regex** (their `score.py`, ported; case- and accent-insensitive, so
  "FRANCE" matches "france"). No LLM judge. New items get their forms from Qwen's own samples and
  the generator's truth aliases.
- **Reads: layers 32, 40, 48, 52, 56, 60** (WSB's signal was at L60); positions to be decided.
  Placebo reads can come from any other dataset.
- **5 draws per case** while Haiku is this cheap.

## Census: which investigator shares Qwen's staleness? (2026-10-29)

Each candidate asked all 109 questions directly, no system prompt, effort medium, one sample (runs
`wsb_cutoff_investigator/2026-10-29_census-*`, on Docent). "Qwen's answer only" = the reply named
Qwen's stale value and not the truth.

| investigator      | Qwen's answer only | truth | neither |
| ----------------- | ------------------ | ----- | ------- |
| Gemini 3.8 Flash  | 24%                | 50%   | 27%     |
| Claude Sonnet 5.5 | 5%                 | 89%   | 6%      |
| Claude Haiku 5.5  | 4%                 | 81%   | 16%     |
| GPT-6 Luna        | 1%                 | 85%   | 14%     |
| GPT-6.1 Terra     | 0%                 | 87%   | 13%     |

On the 55 "stale" items Flash gives Qwen's exact stale value 33% of the time and the truth 36%;
asked its cutoff it says January 2025. Sonnet ignores "answer with just the X" and writes a sentence
first, so 20% of its replies name both the truth and the old value; they count as truth above. The
prompt-only baseline of a same-cutoff investigator is therefore contaminated: it reproduces Qwen's
wrong answer because it has the same wrong answer, with no modeling of Qwen.

## Smoke (Flash, 10 items, 2 draws; 2026-10-29)

Robbie's one-turn prompt, Gemini 3.8 Flash at effort low, 10 dev items with non-numeric answers
confirmed stable on our server. Runs `recall-smoke10`, `mcq-smoke10` and their `-graded` reruns (the
stored submit reply ends with the verdict), Docent collection
[wsb_cutoff_investigator](https://docent.transluce.org/dashboard/b3cbb597-982d-4c54-a6c4-799af0264f9d).

| format    | version | prompt only | prompt + placebo reads |
| --------- | ------- | ----------- | ---------------------- |
| free text | told    | 65%         | 50%                    |
| free text | hidden  | 60%         | 60%                    |
| MCQ       | told    | 35%         | 45%                    |
| MCQ       | hidden  | 35%         | 50%                    |

Ten items: nothing here is a result. What the transcripts show: Flash names Qwen's stale answer on 6
of 10 items from the prompt alone, every draw, told or hidden (the census explains why); the MMLU
placebo bullets mislead it into submitting bare letters ("B") or off-topic phrases; telling it the
model and cutoff changed nothing visible.

## Running it now

```bash
uv run python -m environments run knowledge_cutoff --tag dev109-recall --arms recall --n 8 --conc 2
uv run python -m cfbench.eval knowledge_cutoff --prep --placebo data/runs/olens_reads/2026-10-29_hint-qwen-dsv4-tail/reads.placebo-head.jsonl
uv run python -m cfbench.eval knowledge_cutoff --slug cutoff-haiku-dev109 --predictors anthropic/claude-haiku-5.5 --draws 5
uv run python -m cfbench.answer data/runs/cfbench/<date>_cutoff-haiku-dev109
```

## Haiku on the dev set (2026-10-29, `wsb/` smoke)

`recall-haiku-dev109`: the 40 dev items with a stable, non-numeric Qwen answer, free text, hidden, 5
draws, Haiku 5.5 at medium. Prompt only 10% (20 of 200 draws), placebo reads 8.5%. With no model
named, Haiku predicts what it would answer, mostly the truth, so it almost never lands on Qwen's
stale value: the floor a lens has to lift. The 10% it gets are items where it guesses a 2024-style
answer anyway.
