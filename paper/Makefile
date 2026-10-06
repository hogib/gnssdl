# Build the research proposal.
#
#   make            build proposal.pdf (tectonic if available, else latexmk, else pdflatex+bibtex)
#   make clean      remove build by-products
#   make distclean  also remove the PDF

MAIN    := proposal
SOURCES := $(MAIN).tex references.bib
AUX     := $(addprefix $(MAIN).,aux bbl blg log out toc fls fdb_latexmk synctex.gz)

TECTONIC := $(shell command -v tectonic 2>/dev/null)
LATEXMK  := $(shell command -v latexmk 2>/dev/null)

.PHONY: all clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(SOURCES)
ifneq ($(TECTONIC),)
	tectonic -X compile $(MAIN).tex
else ifneq ($(LATEXMK),)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex
else
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
endif

clean:
	rm -f $(AUX)

distclean: clean
	rm -f $(MAIN).pdf
