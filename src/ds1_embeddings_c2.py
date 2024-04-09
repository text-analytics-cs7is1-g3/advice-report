import pandas as pd
import json
import bert_cls
import os
import torch
import time

os.makedirs("data/ds1-embeddings/", exist_ok=True)
chunksize=128
dfi = pd.read_csv("data/ds1.csv", chunksize=chunksize)

def id2filename(i):
    return f"data/ds1-embeddings/{i}.pt"


for df in dfi:
    t0 = time.perf_counter()
    df["object"] = df["json"].apply(json.loads)
    df["id"] = df["object"].apply(lambda x: x["2lc"]["id"])
    df["2lcbody"] = df["object"].apply(lambda x: x["2lc"]['body'])
    inputs = df["2lcbody"].tolist()
    
    embeddings = bert_cls.four_layer_embeddings(inputs)

    for i in range(len(df)):
        row = df.iloc[i]
        filename = id2filename(row["id"])
        embedding = embeddings[i, :]
        torch.save(embedding, filename)
        print(".", end="", flush=True)
    t1 = time.perf_counter()
    print(f"\nsaved {len(df)} embeddings in {t1-t0:.0f} seconds")
