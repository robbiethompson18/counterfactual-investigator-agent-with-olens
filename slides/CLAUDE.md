# slides

Beamer decks in the sprint's style: Metropolis restyled to look like the env viewer's Evals pages
(`environments/_lib/page.html`: warm off-white page, Inter, indigo accent, hairline rules, no cards)
with the charts' condition colours (`graphs/style.py`). The theme is `beamerthemeolens.sty`,
`example/` is a deck to copy (title, one-page summary, section, figure, table, two-column, panels),
and `make` builds it to `example/build/example.pdf`. PDFs and figures are build output (gitignored):
share the PDF, don't commit it.

**Setup.** XeLaTeX with Metropolis, latexmk and the Inter font (Fira Sans is the fallback). On
Debian/Ubuntu:
`apt-get install texlive-xetex texlive-latex-extra texlive-fonts-recommended latexmk fonts-inter`.

**A new deck.** Copy `example/` to `slides/<name>/`, rename the `.tex` to `<name>.tex`, then
`make deck=<name>`. Put the deck's charts in `slides/<name>/figures.py` (run first by `make`, as
`python -m slides.<name>.figures`), writing PDFs to `fig/`.

**Style rules.**

- Dense, for reading full screen on a monitor rather than presenting to a hall: 9pt, frames
  top-aligned (`\documentclass[aspectratio=169, 9pt, t]{beamer}`), 5 mm margins.
- Indigo is structure (progress bar, section labels, `\takeaway`, `\alert`); every other colour is a
  condition, the charts' (transcript-only blue, O-lens orange, J++ purple, NLA green, no tools gray,
  a placebo is its lens's colour with stripes in the background colour; `slides/figure.py`'s
  `bar_kw` does it). Change a colour in `page.html` / `graphs/style.py` and the theme's
  `\definecolor`s together.
- Start a deck with a one-page summary after the title: three `\stat` numbers, two bullets, a
  `\takeaway`. No boxes or cards anywhere: figures, tables and panels sit straight on the page.
- Figures: make them with `slides/figure.py` (`figure(w_mm, h_mm)`, `save`), which applies
  `slides.mplstyle` (Inter, 8pt, the charts' ink and grid) and takes the colours from
  `graphs/style.py` (`bar_kw`, `diff_rows`). Make each figure at its slide size (`FULL_W` × `FULL_H`
  under a title, `HALF_W` for a column) and include it with `\slidefig`, which never scales it, so
  the figure's text matches the slide's. Vector PDF, transparent background.
- One message per frame: say the result in the frame title ("The O-lens adds 8pp over the prompt
  alone", not "Results"), and use `\takeaway{...}` for the one sentence the slide is for.
- Give provenance with `\source{...}` (run, doc) at the bottom of a results slide.

**Macros.** `\olens`, `\jlens`, `\nla`, `\transcript` (the condition's name in its colour),
`\cond{colour}{text}`, `\sw{olens}` (swatch), `\swp{olens}` (striped placebo swatch), `\muted{...}`,
`\best{...}` (table winner), `\tablefont` (small, tabular figures), `\slidefig{path}`,
`\takeaway{...}`, `\source{...}`, `\begin{panel}[title]` (blocks are panels too),
`\stat{value}{colour}{what it is}`, `\struck{...}` (a crossed-out superseded line, one line only).
