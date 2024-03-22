import pandas as pd
import sys
import json
import os


class MissingNameException(Exception):
    pass


subnamesfile = "data/submission-names-manual.csv"
names = pd.read_csv(subnamesfile)["name"].str
pre = "data/comments-manual"
os.makedirs(pre, exist_ok=True)

finished = 0

for line in sys.stdin:
    try:
        obj = json.loads(line)
        if "link_id" not in obj:
            raise MissingNameException("Missing link_id: " + line)
        link_id = obj["link_id"]
        if names.contains(link_id).any():
            with open(pre + "/" + obj["link_id"] + ".comments", 'a') as f:
                f.write(line)
                finished += 1
                print("finished:", finished)
    except json.JSONDecodeError as e:
        print(e, file=sys.stderr)
    except MissingNameException as e:
        print(e, file=sys.stderr)
