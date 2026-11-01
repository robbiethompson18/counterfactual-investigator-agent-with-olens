# Code smells

Tech debt we know about, newest first.

- **2026-10-31, duplicated chart code.** `graphs/talk.py`'s `grader` and `drink_drive` draw the same
  bars as `graphs.cutoff_grader_mentions.period` and `graphs.drink_drive` (the write-up's PNGs),
  each with its own copy of the drawing loop. Fold them into one draw function each, as
  `essay_grading.essay_bars` and `author_blog_drop` already do, once the deck stops changing.

- **2026-10-31, `scripts/build_writeup.sh` resource path.** pandoc searches `docs` (the current
  checkout) before the main checkout's `docs`, so a pool checkout with a stale gitignored
  `docs/img/` embeds its old PNGs in `final-writeup.html` (PR #589 shipped 32 stale charts that way;
  #590 fixed the HTML). Build the HTML in the checkout whose charts are fresh, or make the script
  prefer the newest copy.
