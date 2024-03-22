import pandas as pd
import sys
import json
import os


class MissingNameException(Exception):
    pass


subnamesfile = "data/submission-names-manual.csv"
names = pd.read_csv(subnamesfile)["name"].str
pre = "data/submissions-manual"
os.makedirs(pre, exist_ok=True)

finished = 0

for line in sys.stdin:
    try:
        obj = json.loads(line)
        if "name" not in obj:
            raise MissingNameException(line)
        name = obj["name"]
        if names.contains(name).any():
            with open(pre + "/" + obj["name"], 'w') as f:
                f.write(line)
                finished += 1
                print("finished:", finished)
    except json.JSONDecodeError as e:
        print(e, file=sys.stderr)
    except MissingNameException as e:
        print(e, file=sys.stderr)
