import pandas as pd

df = pd.read_csv("data/submission-names-unique.csv")
df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)

manual = df[0:200]
analysis = df[200:10200]

manual.to_csv("data/submission-names-manual.csv")
analysis.to_csv("data/submission-names-analysis.csv")
