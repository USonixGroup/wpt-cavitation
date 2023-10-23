# This Makefile is designed to be used with GNU Make to run the LaTeX commands to generate a thesis.
#
# If you're not using the Linux or OS X command line, you probably won't need this.


# The target definitions are a bit of a lazy shortcut (we *could generate each of the intermediate files with specific rules only when we need to) but it works and it's short.

LaTeXSources= $(wildcard *.tex *.bib) JASA.cls

.PHONY: all clean superclean test
all: manuscript_jasa.pdf


manuscript_jasa.pdf: $(LaTeXSources)
	pdflatex --halt-on-error $(basename $@).tex
	bibtex $(basename $@)
	pdflatex --halt-on-error $(basename $@).tex
	pdflatex --halt-on-error $(basename $@).tex
#	dvips -P pdf $(basename $@).dvi
#	ps2pdf $(basename $@).ps


clean: 
	-@rm -v *.aux *.lof *.log *.lot *.fls *.out *.toc *.dvi *.bbl *.bcf *.blg *-blx.aux *-blx.bib *.run.xml *.fdb_latexmk *.synctex.gz *.pdfsync *.alg *.loa *.thm *.nav *.snm *.vrb *.acn *.acr *.glg *.glo *.gls *.brf *.lol *.idx *.ilg *.ind *.ist *.maf *.mtc *.mtc0 *.pyg *.nlo *.sagetex.sage *.sagetex.py *.sagetex.scmd *.sout *.sympy *.tdo *.xdy *.pfg *.slg *.sot *.stn

superclean: clean
	-@rm *.pdf *.ps

rootclean: clean superclean
	-@rm $(TestCases) $(TestCases:.pdf=.tex)

