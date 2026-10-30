# make                  the example deck: figures, then example/build/example.pdf
# make deck=talk        slides/talk/talk.tex, its figures from `python -m slides.talk.figures` (if that module exists)
deck ?= example
export TEXINPUTS := $(CURDIR):

all:
	if [ -f $(deck)/figures.py ]; then cd .. && uv run python -m slides.$(deck).figures; fi
	cd $(deck) && latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build $(deck).tex

clean:
	rm -rf */build */fig

.PHONY: all clean
