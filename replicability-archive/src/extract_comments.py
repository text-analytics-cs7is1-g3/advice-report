import pandas as pd
import sys
import json
import os
import shutil
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--copy", action="store_true", default=False)
args = ap.parse_args()

print(args)


class MissingNameException(Exception):
    pass


subnamesfile = "data/selected-50000.csv"
names = pd.read_csv(subnamesfile)["name"]
pre = "data/submissions_with_comments/"
os.makedirs(pre, exist_ok=True)

check_names = {}
for name in names:
    check_names[name] = True
    if args.copy:
        shutil.copy(f"data/grouped_submissions/{name}", pre + name)

finished = 0
decode_errors = 0
missing_field_errors = 0
linkf = open("comments-processed.txt", 'a')
for line in sys.stdin:
    try:
        obj = json.loads(line)
        if "link_id" not in obj or "id" not in obj:
            raise MissingNameException(line)
        link_id = obj["link_id"]
        linkf.write(obj["id"] + "\n")
        if link_id in check_names:
            with open(pre + "/" + link_id, 'a') as f:
                f.write(line)
                finished += 1
                print("comments added:", finished)
    except json.JSONDecodeError as e:
        print(e, file=sys.stderr)
        decode_errors += 1
        print("decode errors: ", decode_errors, file=sys.stderr)
    except MissingNameException as e:
        print(e, file=sys.stderr)
        missing_field_errors += 1
        print("missing_field_errors: ", missing_field_errors, file=sys.stderr)
