import pandas as pd
import json
import bert_cls
import os
import torch
import time
import sys
import cls
import features
import numpy as np

split = int(sys.argv[1])
if split != 1 and split != 2:
    raise Exception

dfa = None
try:
    dfa = pd.read_csv(f"data/ds2-animosity-part{split}.csv")
except:
    print("Exception: creating animosity file")
    dfa = pd.read_csv(f"data/processed_part{split}.csv", usecols=["comment_id"])
    dfa["animosity"] = dfa["comment_id"].apply(lambda x: 'none')
    dfa.to_csv(f"data/ds2-animosity-part{split}.csv")

os.makedirs(f"data/ds2-embeddings-part{split}/", exist_ok=True)
chunksize = 2048
dfi = pd.read_csv(f"data/processed_part{split}.csv", chunksize=chunksize)

def id2filename(i):
    return f"data/ds2-embeddings-part{split}/{i}.pt"

def load_tensor(i):
    try:
        tensor = torch.load(id2filename(i))
        return tensor
    except:
        return None

def need_to_classify(val):
    return val == 'none'

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
    need_to_create_mask = dfa.loc[df.index, "animosity"].apply(need_to_classify)
    skipping = sum(list(map(nott,need_to_create_mask)))
    total_skipped += skipping
    print(f"skipping {skipping} already saved tensors. total skipped: {total_skipped}")
    dfs = df[need_to_create_mask]
    dfs["embeddings"] = dfs["id"].apply(load_tensor)
    can_try = dfs["embeddings"].apply(lambda x: x is not None)
    dfs = dfs[can_try]
    dfs["embeddings"] = dfs["embeddings"].apply(lambda x: x.numpy())

    dfs["body"]=dfs["comment_body"]
    inputs = dfs["body"].tolist()
    X_bert = np.array(dfs["embeddings"].tolist())
    pred_animosity = cls.animosity(X_bert, inputs)

    dfa.loc[dfs.index, "animosity"] = pred_animosity

    t1 = time.perf_counter()
    print(f"\nclassified {len(dfs)} docs in {t1 - t0:.0f} seconds")
    dfa.to_csv(f"data/ds2-animosity-part{split}.csv")
