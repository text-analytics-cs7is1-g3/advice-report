import pandas as pd

df = pd.read_csv("data/submission-names-unique.csv")
df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)

selected_submissions = df.iloc[0:10200].copy()
selected_submissions.loc[0:200, "purpose"] = "manual annotation"
selected_submissions.loc[200:10200, "purpose"] = "analysis"
selected_submissions.to_csv("data/selected-submissions.csv", index=False)
