# Code smells

Tech debt we know about, newest first.

- **2026-10-31, duplicated chart code.** `graphs/talk.py`'s `grader` and `drink_drive` draw the same
  bars as `graphs.cutoff_grader_mentions.period` and `graphs.drink_drive` (the write-up's PNGs),
  each with its own copy of the drawing loop. Fold them into one draw function each, as
  `essay_grading.essay_bars` and `author_blog_drop` already do, once the deck stops changing.
