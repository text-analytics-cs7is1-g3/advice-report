import pandas as pd
import ast
import json
import re
import sys

predf_iterator = pd.read_csv("data/ds1-step1.csv", chunksize=128)

pattern = r'\b(YTA|ESH)\b'
regex = re.compile(pattern)
def test_match(text):
    return regex.search(text)

found_valid = 0

submissions_with_valid_f = open("data/ds1_subs_with_valid.csv", "w")
submissions_with_valid_f.write("name\n")

def valid(row):
    submission_name = row["submission_name"]
    tlc = json.loads(row["tlc"])
    twolc = json.loads(row["2lc"])
    with open(f"data/submissions_with_comments/{submission_name}", "r") as f:
        line = f.readline()
        submission = json.loads(line)
    try:
        a1 = submission["author_fullname"]
        a2 = tlc["author_fullname"]
        a3 = twolc["author_fullname"]
    except:
        try:
            a1 = submission["author"]
            a2 = tlc["author"]
            a3 = twolc["author"]
        except:
            return False

    for text in submission["selftext"], tlc["body"], twolc["body"]:
        if text in ["[deleted]", "[removed]"]:
            return False
    
    if a1 is not None and a1 == a3:
        if test_match(tlc["body"]):
            print(submission["name"], tlc["id"], twolc["id"], a1, a3, "yay!", file=sys.stderr)
            return {
                "submission": submission,
                "tlc": tlc,
                "2lc": twolc,
            }
    else:
        return False

for predf in predf_iterator:
    for i, row in predf.iterrows():
        keep = valid(row)
        if keep:
            print(json.dumps(keep))
            submissions_with_valid_f.write(row["submission_name"] + "\n")
            found_valid += 1
            print(found_valid, file=sys.stderr)
