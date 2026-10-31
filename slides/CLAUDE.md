# slides

Beamer decks in the sprint's style: Metropolis restyled to look like the env viewer's Evals pages
(`environments/_lib/page.html`: warm off-white page, Inter, indigo accent, hairline rules, no cards)
with the charts' condition colours (`graphs/style.py`). The theme is `beamerthemeolens.sty`, `talk/`
is the sprint's one presentation, and `make` builds it to `talk/build/talk.pdf`. PDFs and figures
are build output (gitignored): don't commit them. CI builds the deck on every push to main that
touches `slides/` or `graphs/` and publishes it at
https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/talk.pdf (share that
link); a PR touching them attaches its `talk.pdf` to the workflow run
(`.github/workflows/env-viewer-pages.yml`).

**Setup.** XeLaTeX with Metropolis, latexmk and the Inter font (Fira Sans is the fallback). On
Debian/Ubuntu:
`apt-get install texlive-xetex texlive-latex-extra texlive-fonts-recommended latexmk fonts-inter`.

**The deck.** `talk/talk.tex`, with its charts in `talk/figures.py` (run first by `make`, as
`python -m slides.talk.figures`, from the real results), writing PDFs to `fig/`. Add new parts to it
rather than starting another deck.

**Style rules.**

- Dense, for reading full screen on a monitor rather than presenting to a hall: the 9pt class with
  every size at 2/3 outside the title slide (`talk.tex` redefines them; figures scale their text by
  the same `S`), frames top-aligned, 5 mm margins, no big gaps.
- Indigo is structure (progress bar, section labels, `\takeaway`, `\alert`); every other colour is a
  condition, the charts' (transcript-only blue, O-lens orange, J++ purple, NLA green, no tools gray,
  a placebo is its lens's colour with stripes in the background colour; `slides/figure.py`'s
  `bar_kw` does it). Change a colour in `page.html` / `graphs/style.py` and the theme's
  `\definecolor`s together.
- No boxes or cards anywhere: figures, tables and panels sit straight on the page. A part opens with
  `\parthead{n}{title}` on its first frame (no separate section page).
- Figures: make them with `slides/figure.py` (`figure(w_mm, h_mm)`, `save`), which applies
  `slides.mplstyle` (Inter, 8pt, the charts' ink and grid) and takes the colours from
  `graphs/style.py` (`bar_kw`, `diff_rows`). Make each figure at its slide size (`FULL_W` × `FULL_H`
  under a title, `HALF_W` for a column) and include it with `\slidefig`, which never scales it, so
  the figure's text matches the slide's. Vector PDF, transparent background.
- No commentary (Kylie, 2026-10-30): frame titles name the content ("Results: ..."), never the
  result; no `\takeaway` lines; bullets give numbers and facts, no "only" or "harder". No eval keys
  or run paths on slides.
- Charts: a binary P(claim) task as AUROC (axis from 0.5; % right rewards a shift in the
  investigator's bias), a letter task as % correct; one decimal on values; each lens beside its
  hatched placebo under one label; the axis stops just above the data; the dotted line is the
  meaningful baseline (e.g. always the hinted letter); one-sided stars, stated in the footnote.
- Give provenance with `\source{...}` (run, doc) at the bottom of a results slide.

**Macros.** `\olens`, `\jlens`, `\nla`, `\transcript` (the condition's name in its colour),
`\cond{colour}{text}`, `\sw{olens}` (swatch), `\swp{olens}` (striped placebo swatch), `\muted{...}`,
`\best{...}` (table winner), `\tablefont` (small, tabular figures), `\slidefig{path}`,
`\takeaway{...}`, `\source{...}`, `\begin{panel}[title]` (blocks are panels too),
`\stat{value}{colour}{what it is}`, `\struck{...}` (a crossed-out superseded line, one line only).
