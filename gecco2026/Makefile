# GECCO 2026 Workshop Paper — Makefile
# Builds paper.tex with pdflatex + bibtex

PAPER = paper
LATEX = pdflatex
BIBTEX = bibtex
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

.PHONY: all clean watch

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	$(LATEX) $(LATEX_FLAGS) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(LATEX_FLAGS) $(PAPER)
	$(LATEX) $(LATEX_FLAGS) $(PAPER)
	@echo ""
	@echo "Build complete: $(PAPER).pdf"

clean:
	rm -f $(PAPER).{aux,bbl,blg,log,out,pdf,fls,fdb_latexmk,synctex.gz}
	rm -f *.aux *.bbl *.blg *.log *.out
	@echo "Clean"

# Quick single-pass build (no bibliography)
quick:
	$(LATEX) $(LATEX_FLAGS) $(PAPER)

# Word count (approximate)
wc:
	@detex $(PAPER).tex 2>/dev/null | wc -w || echo "Install detex for word count"
