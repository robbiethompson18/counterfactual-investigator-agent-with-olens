# graphs

Every chart in the repo, for the docs and for the slide deck, one module each, run from the repo
root as `uv run python -m graphs.<module>`; each writes PNGs under `docs/img/` (gitignored: the doc
that shows a chart names the module that regenerates it), except `talk`, which writes the deck's
PDFs to `slides/talk/fig/`. `docs/` and `slides/` are leaves: nothing here imports from them, and no
Python lives there. Run selection is not here: every run a chart reads is named in `cfbench/catalog.py`,
and scoring lives in `cfbench.scoring`. Every module imports `style.py`, and importing it applies the look (no `use()`,
no `rcParams` anywhere else): colors, placebo stripes, bars with CI whiskers and the
paired-difference rows (each labelled by its two conditions' swatches, A − B) are `bar_kw`,
`marker_kw`, `ci_bar`, `positions` and `diff_rows`; CLAUDE.md has the rule. A chart whose setup departs from the
write-up's defaults carries the note inside the figure, `departure(fig, note)`, never under it in the Markdown or the
deck, so it travels with the image. A slide figure is a
function decorated with `@slide`, which applies the deck's sizes and transparent background
(`SLIDE_RC`) around that one figure, with `slide_figure(w_mm, h_mm)` and `slide_save`; never apply
them process-wide (that is what turned the headline PNGs white and small on 2026-10-31). The font is
Inter (macOS: `brew install --cask font-inter`), with a silent DejaVu fallback if it's missing; the
🔮 of "🔮-lens" comes from the monochrome Noto Emoji (`brew install --cask font-noto-emoji`), and
without it labels fall back to "O-lens" (`style.OLENS`). Each module's first docstring line says
what it draws and which doc shows it; `grep -m1 '"""' graphs/*.py` lists them.

Canonical runs for the final write-up: every lens number, chart and token count on MCQ (MMLU-Pro,
SuperGPQA), knowledge cutoff and preferences comes only from the menus_inline runs listed in
`cfbench/catalog.py` `MENUS`, menu `CANONICAL_MENU`, with their 3-draw top-ups; never the 2026-10-29
per-set runs (their read positions and prompts differ by set). Essay grading has no menus run and
keeps its own.
