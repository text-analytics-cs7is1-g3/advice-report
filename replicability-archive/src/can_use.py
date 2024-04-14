import pandas as pd
import json

df = pd.read_csv("data/submission-names-unique.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


def can_use(name):
    try:
        with open(f"data/grouped_submissions/{name}", "r") as f:
            o = json.load(f)
            if "selftext" not in o:
                return False
            if o["selftext"] in ["[deleted]", "[removed]"]:
                return False
            return True
    except FileNotFoundError:
        return False


for name in df["name"]:
    if can_use(name):
        print(name)
