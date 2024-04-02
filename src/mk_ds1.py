"""
Make dataset for RQ1.

To answer RQ1 we need examples of advice-seeking submissions with a negative comments
(namely where the comment contains either YTA or ESH). Each comment is directly descended from one other node in the discussion tree. For this dataset we are interested in comments descended directly from the first submission, i.e. the root of the tree.
We say the root node (the submission) has depth 0, $d=0$, a comment directly on the submission has depth 1, $d=1$, and so on. Comments with depth 1 are named \textsc{tlc}s (Top Level Comments) by \citet{bao-2021}. 
Secondly, we need to see that the OP has responded to the negative comment.
Therefore a sample in this dataset has three texts, the original submission $c_0$ authored by $a_1$ ($d=0$), a comment $c_1$ on the original submission containing any of the substrings ``YTA'' or ``ESH'' authored by $a_2\neq a_1$ ($d=1$), and a comment $c_2$ authored by $a_1$  ($d=2$) on the comment $c_1$ authored by $a_2$. This structure is represented in Figure~\ref{fig:two-comments}.
For each of the three texts we then extract proxy measures for the extra-textual quantities of interest using LIWC, VADER, and BERT, namely whether the user $x$ expresses appreciation towards the user $y$. Other simple features such as comment lengths (word count, sentence count), and the textual index locating the relevant acronym are recorded.
We also approximate the number of moral judgements a submission has received (the count of comments containing one of the acronyms).
Such that individual data samples are reasonably de-correlated we extract only one sample from each discussion tree.
"""

import pandas as pd
import json
import sys
df_iterator = pd.read_csv("data/selected-50000.csv", chunksize=1)

seqdf = pd.DataFrame()

num_subs_deleted = 0
num_subs_removed = 0
num_tlcs_deleted = 0
num_tlcs_removed = 0


def top_level_comments(submisson, comments):
    for c in comments:
        if c["parent_id"] == submission["name"]:
            yield c


def comment_children(comment, comments):
    for c in comments:
        if type(c["parent_id"]) is int:
            if c["parent_id"] == comment["id"]:
                print('int', c["parent_id"])
                yield c
        elif type(c["parent_id"]) is str:
            if "name" in comment:
                if c["parent_id"] == comment["name"]:
                    yield c
            else:
                parent_id = c["parent_id"].split("_")[1]
                if parent_id == comment["id"]:
                    print("split", parent_id, comment["id"])
                    yield c


discussions = {
    "submission_name": [],
    "tlc": [],
    "2lc": [],
}


def load_submission_and_comments(name):
    fname = f"data/submissions_with_comments/{name}"
    with open(fname, "r") as f:
        line = f.readline()
        submission = json.loads(line)
        comments = []
        while line:
            line = f.readline()
            if line:
                comments.append(json.loads(line))
        return [submission, comments]


for df in df_iterator:
    print(df["name"])
    name = df["name"].iloc[0]
    submission, comments = load_submission_and_comments(name)
    assert submission["name"] == name
    skip = False
    if submission["selftext"] == "[deleted]":
        num_subs_deleted += 1
    elif submission["selftext"] == "[removed]":
        num_subs_removed += 1
    else:
        tlcs = list(top_level_comments(submission, comments))
        for tlc in tlcs:
            if tlc["body"] == "[deleted]":
                num_tlcs_deleted += 1
            elif tlc["body"] == "[removed]":
                num_tlcs_removed += 1
            else:
                child_comments = list(comment_children(tlc, comments))
                for twolc in child_comments:
                    discussions["submission_name"].append(submission["name"])
                    discussions["tlc"].append(tlc)
                    discussions["2lc"].append(twolc)
                    print(submission["id"], tlc["id"], twolc["id"], file=sys.stderr)


print("num subs deleted:", num_subs_deleted)
print("num subs removed:", num_subs_removed)

df = pd.DataFrame(discussions)
df.to_csv("data/ds1-pre.csv")
