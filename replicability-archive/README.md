The following scripts depend on downloading
`./AmItheAsshole_commenst.zst` and `./AmItheAsshole_submissions.zst`
from the torrent with hash: 56aa49f9653ba545f48df2e33679f014d2829c10
and putting these files in the ./data directory.

All scripts should be run when the working directory is the same as the location of this README.md file.


There is a list of all unique 'submission names' in `./data/submission-names-unique.csv`

To extract submissions from zstandard compressed file run:

`./bin/cat_submissions | python src/extract_submissions.py`

The 'submission names' of the 50,000 uniformly sampled submissions are in `./data/selected-50000.csv`.

To collect the 50,000 submissions and their comments into `./data/submissions_with_comments`
run:
`./bin/cat_comments | python src/extract_comments.py`.

The manually annotated comments are collected in `./data/manual.csv`.

In order to train the logistic regression model we need bert embeddings of the manually annotated comments which we can generated with:
`python src/man_embeddings.py`.

To run cross-validation for the animosity and thankfulness classifiers run:

`python src/manual_train_cls_thankfulness.py` and
`python src/manual_train_cls_animosity.py`

# Dataset 1 for RQ1
To create the dataset for dataset 1 we need to run:
`python src/mk_ds1_pre.py` (creates `./data/ds1-step1.csv`)
`python src/mk_ds1_valid.py` (creates `./data/ds1_subs_with_valid.csv`)
`python src/mk_ds1_grouped.py` (create `./data/ds1_counts_per_sub.csv` and `./data/ds1_grouped/*`)
`python src/mk_ds1_sample.py` (creates `./data/ds1.csv`)

To create BERT embeddings for Dataset 1 run:
`python src/ds1_embeddings_c2.py`

Finally, to label dataset for the labels of interest (`thankfulness` and `you_as_subject`), run:
`python src/ds1.py` (creates `./data/ds1-you-thank.csv`)

To run the statistical chi-square test on the labeled data run:
`Rscript src/ds1-test.r`

