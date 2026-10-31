# Code smells

Tech debt we know about, newest first.

- **2026-10-31, graphs:** the deck and the write-up still draw the same results twice.
  `graphs/talk.py` `essay()` and `author()` overlap `graphs/headline_inline.py`'s essay chart and
  `graphs/author_blog_drop.py` (different bar sets, same statistics in `cfbench.scoring`). The next
  step is drawing functions that take an `ax`, so one module renders a chart at write-up size (PNG)
  and at slide size (`@slide`, PDF). Not done before the deadline because the two show different
  bars.
