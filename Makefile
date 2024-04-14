ROBINSON-GUNNING-NEIMHIN-16321701-CS7IS4-FINAL-GROUP3-MONITOR.pdf: final-group3.pdf
	cp $< $@

ROBINSON-GUNNING-NEIMHIN-16321701-CS7IS4-GROUP3-MONITOR-ARCHIVE.zip: replicability-archive.zip
	cp $< $@

replicability-archive.zip: replicability-archive/
	zip -r $@ $<

fig/ds1-contingency-table.csv.pdf: fig/ds1-contingency-table.csv
	python src/csv_to_pdf.py $< $@

review-classifier-thankfulness.pdf: review-classifier-thankfulness.csv
	python src/csv_to_pdf.py $< $@

thankfulness-bow.pdf: thankfulness-bow.txt
	python src/csv_to_pdf.py $< $@

animosity-bow.pdf: animosity-bow.txt
	python src/csv_to_pdf.py $< $@

fig/evals.pdf: fig/evals.csv
	python src/csv_to_pdf.py $< $@

fig/evals.csv:
	python src/manual_eval.py

data/ds1.csv: data/ds1.csv.zip
	unzip $<
