# make                  the presentation: figures, then talk/build/talk.pdf
# make deck=talk        slides/talk/talk.tex, its figures from `python -m graphs.talk` (if that module exists)
# make tex              the PDF from the figures already in talk/fig/, for checking a text edit
deck ?= talk
export TEXINPUTS := $(CURDIR):

all: figs tex

figs:
	if [ -f ../graphs/$(deck).py ]; then cd .. && uv run python -m graphs.$(deck); fi

tex:
	cd $(deck) && latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build $(deck).tex

clean:
	rm -rf */build */fig

.PHONY: all figs tex clean
