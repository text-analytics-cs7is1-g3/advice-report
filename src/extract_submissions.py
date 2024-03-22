import pandas as pd
import sys
import json
import os


class MissingNameException(Exception):
    pass


subnamesfile = "data/selected-submission.csv"
names = pd.read_csv(subnamesfile)["name"].str
pre = "data/submissions"
os.makedirs(pre, exist_ok=True)

check_names = {}
for name in names:
    check_names[name] = True

finished = 0

for line in sys.stdin:
    try:
        obj = json.loads(line)
        if "name" not in obj:
            raise MissingNameException(line)
        name = obj["name"]
        if name in check_names:
            with open(pre + "/" + obj["name"], 'w') as f:
                f.write(line + "\n")
                finished += 1
                print("finished:", finished)
    except json.JSONDecodeError as e:
        print(e, file=sys.stderr)
    except MissingNameException as e:
        print(e, file=sys.stderr)
