import json
import os
import pandas as pd

os.makedirs("data/ds1_grouped", exist_ok=True)

data = {"name": [], "count": []}
meta = pd.DataFrame(data)
meta.set_index('name', inplace=True)

with open("data/ds1.txt", "r") as f:
    line = f.readline()
    while line:
        o = json.loads(line)
        sname = o["submission"]["name"]
        if sname not in meta.index:
            print(sname)
            meta.loc[sname] = 1
        else:
            meta.loc[sname, 'count'] += 1
        with open(f"data/ds1_grouped/{sname}", "a") as f2:
            f2.write(line)
        line = f.readline()

meta.to_csv("data/ds1_counts_per_sub.csv")

# for sname in submission_groups:
#     group = submission_groups[sname]
#     print(json.dumps(random.choice(group)))
