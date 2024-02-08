
.PHONY: clean
clean:
	-rm main.fdb_latexmk
	-rm main.fls
	-rm main.blg
	-rm main.bbl
	-rm main.aux
	-rm main.log

main.pdf: main.tex mybib.bib
	latexmk -pdf main.tex
