main.pdf: main.tex neimhin-summaries.tex research-questions.tex mybib.bib
	latexmk -pdf main.tex

.PHONY: clean
clean:
	-rm main.fdb_latexmk main.fls main.blg main.bbl main.aux main.log neimhin-summaries.aux ben-summaries.aux research-questions.aux main.synctex.gz
