# Brief: author 50 short-answer "knowledge cutoff" questions in your domain

You are building a dataset of dated factual questions. The subject model (Qwen3.6-27B) has a
knowledge cutoff around January 2025. We want questions about events from January 2024 through
September 2026. A model with that cutoff will answer the 2024 ones correctly and will often answer
the later ones with a stale value (last year's winner, the previous CEO, the old price). Both kinds
are useful. Your job: 50 questions, each verified, each with a short unambiguous answer.

## Per question

- **Ground it.** Start from a real page: a Wikipedia "2024 in X" / "2025 in X" / "2026 in X" page,
  a list page, a release-history table, or the primary announcement. Use WebSearch and WebFetch
  (load them via ToolSearch if they aren't loaded).
- **Verify it.** Either fetch one page that states the answer, or find two independent search
  results that state the same answer. Snippet-only from one result is not enough. Record the URL you
  actually read the answer on. If you can't verify, drop it.
- **Short, unambiguous answer.** One name, one number, one title. No lists. Drop anything where two
  answers are defensible, where sources disagree, or where the value keeps changing (view counts,
  live prices) unless the question pins a specific dated publication.
- **Pin the event.** Name the product / office / competition / list and the month and year.
- **Not derivable from the question.** No version numbers that track the year (Java 25 in 2025),
  no "which company, maker of Pringles", no answers that follow from a generic rule (which rocket
  launched a NASA flagship → Falcon Heavy). Ask yourself: could a smart reader who never heard of
  the event guess it? If yes, drop it.
- **One question per series.** Don't ask the same record / office / annual list at several dates.
  Pick the one date with the best stale trap.
- **Prefer stale traps.** For post-cutoff questions, prefer ones where the same quantity had a
  different value in 2023–2024 and record that older value as `pre_cutoff_analog`. Aim for at least
  two thirds of your post-cutoff questions to have one. Questions with no stale analog are allowed.
- **Dates.** About 15 questions on 2024 events, 25 on 2025, 10 on January–September 2026. Nothing
  after September 2026.
- **No numeric answers.** Every answer must be a name, title, place, product or word, never a
  number, year, price, count, version number or measurement. (Two reasons: the subject's numbers
  vary across samples so they fail the stability gate, and token lenses can't name multi-digit
  values because the subject splits digits.) If the natural question has a numeric answer, ask a
  different question about the same event whose answer is a name, or drop it.
- **Don't repeat existing items.** The file `existing_list.txt` beside this brief lists every
  question already in the dataset (date, question, answer). Don't ask about the same event or the
  same answer.

## Output

Write ONE file at the path you were given, one JSON object per line, exactly these keys:

```
{"id": "<prefix>-01", "domain": "<your domain>", "event_date": "YYYY-MM",
 "question": "<the question, ending with: If you are not sure, give your best guess. Answer with just the <answer_type>.>",
 "answer_type": "<e.g. product name / number / company name / version number>",
 "truth": "<canonical short answer, lowercase for names>",
 "truth_aliases": ["<other acceptable surface forms>"],
 "pre_cutoff_analog": "<the older value a Jan-2025 model might give, or null>",
 "source_url": "<the url you read the answer on>",
 "verified_by": "fetched page" | "two search results",
 "notes": "<one line on why this is verifiable and unambiguous>"}
```

**Append to the file after every 10 questions** (don't hold all 50 in memory until the end), so
partial work survives if you run out of room. Number ids sequentially.

Write nothing anywhere else: no files in the repo, no commits, no messages to anyone. When done,
reply with the count written, the date split, the numeric count, and a one-line list of questions
you dropped with the reason.
