
.PHONY: clean
clean:
	-rm main.fdb_latexmk main.fls main.blg main.bbl main.aux main.log neimhin-summaries.aux

main.pdf: main.tex mybib.bib
	latexmk -pdf main.tex
