submission-names-unique.csv:
	echo name > $@
	zstdcat /media/neimhin/TOSHIBA\ EXT/reddit/subreddits23/AmItheAsshole_submissions.zst | jq .name | grep -v '^null$$' | sed 's/"//g' | uniq >> $@

submission-names-manual.csv: submission-names-split
submission-names-analysis.csv: submission-names-split

.PHONY: submission-names-split
submission-names-split: submission-names-unique.csv
	python src/select-200-10000-submissions.py
