import pandas as pd
import sys
import json
import os


class MissingNameException(Exception):
    pass


pre = "data/grouped_submissions"
os.makedirs(pre, exist_ok=True)

finished = 0

for line in sys.stdin:
    try:
        obj = json.loads(line)
        if "name" not in obj:
            raise MissingNameException(line)
        name = obj["name"]
        with open(pre + "/" + obj["name"], 'w') as f:
            f.write(line)
            finished += 1
            print("finished:", finished)
    except json.JSONDecodeError:
        print("decode-error", file=sys.stderr)
    except MissingNameException:
        print("missing-name", file=sys.stderr)
