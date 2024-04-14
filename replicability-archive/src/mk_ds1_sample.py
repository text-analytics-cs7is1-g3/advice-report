import json
import os
import pandas as pd
import random

df = pd.read_csv("data/ds1_counts_per_sub.csv")
data = {
    "name": [],
    "json": [],
}
df2 = pd.DataFrame(data)
df2.set_index('name', inplace=True)


def read_lines(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return lines


for i, row in df.iterrows():
    fname = f"data/ds1_grouped/{row['name']}"
    count = row['count']
    lines = read_lines(fname)
    assert count == len(lines)
    line = random.choice(lines)
    assert row['name'] == json.loads(line)["submission"]["name"]
    df2.loc[row['name']] = line

df2.to_csv("data/ds1.csv")
