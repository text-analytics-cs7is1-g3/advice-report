import pandas as pd
import json
import bert_cls
import os
import torch
import time
import sys

split = int(sys.argv[1])
if split != 1 and split != 2:
    raise Exception

os.makedirs(f"data/ds2-embeddings-part{split}/", exist_ok=True)
chunksize = 128
dfi = pd.read_csv(f"data/processed_part{split}.csv", chunksize=chunksize)

def id2filename(i):
    return f"data/ds2-embeddings-part{split}/{i}.pt"


def need_to_create(id):
    try:
        tensor = torch.load(id2filename(id))
        if tensor.numpy().shape[0] == 3072:
            return False
        return True 
    except:
        return True

def nott(x):
    if x == 1:
        return 0
    return 1
total_skipped = 0
i=0
for df in dfi:
    i=i+1
    t0 = time.perf_counter()
    df["id"]=df["comment_id"]
    need_to_create_mask = df["id"].apply(need_to_create)
    skipping = sum(list(map(nott,need_to_create_mask)))
    total_skipped += skipping
    print(f"skipping {skipping} already saved tensors. total skipped: {total_skipped}")
    dfs = df[need_to_create_mask]

    dfs["body"]=dfs["comment_body"]
    inputs = dfs["body"].tolist()

    if len(inputs) > 0:
        print("running embeddings")
        embeddings = bert_cls.four_layer_embeddings(inputs)

        for i in range(len(dfs)):
            print(".", end="", flush=True)
            row = dfs.iloc[i]
            filename = id2filename(row["id"])
            embedding = embeddings[i, :]
            torch.save(embedding, filename)
            print(".", end="", flush=True)
    t1 = time.perf_counter()
    print(f"\nsaved {len(dfs)} embeddings in {t1 - t0:.0f} seconds")
