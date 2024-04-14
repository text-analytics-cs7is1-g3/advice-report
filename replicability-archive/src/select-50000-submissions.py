import pandas as pd

df = pd.read_csv("data/can-use.csv")
df = df[0:50000]

df.to_csv("data/selected-50000.csv")
