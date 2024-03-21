data/submission-names-unique.csv:
	echo name > $@
	zstdcat /media/neimhin/TOSHIBA\ EXT/reddit/subreddits23/AmItheAsshole_submissions.zst | jq .name | grep -v '^null$$' | sed 's/"//g' | uniq >> $@

data/submission-names-manual.csv: data/submission-names-split
data/submission-names-analysis.csv: data/submission-names-split

tmp/submission-names-split: data/submission-names-unique.csv
	python src/select-200-10000-submissions.py
	mkdir -p tmp
	touch $@
