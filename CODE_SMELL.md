# Code smells

Tech debt we know about, newest first.

- **2026-10-31, graphs:** the deck (`graphs/talk.py` `essay()`) and the write-up
  (`graphs/headline_inline.py`) draw essay grading from the same statistics
  (`cfbench.scoring.essay_auroc_bars`) as different charts: AUROC with the "bet at the mean"
  variants on the slide, % right plus paired differences in the write-up. Whether they should be one
  chart drawn at two sizes (a draw function taking an `ax`, rendered as PNG and under `@slide` as
  PDF) is an editorial call, open as of this date. The blog chart was unified the same day:
  `author_blog_drop` and the ablation's blog panel compute from `author_bars` / `author_cases`.
