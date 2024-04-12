import pandas as pd
import json
import torch
import cls
import time
import numpy as np
import argparse
import stanza
import multiprocessing

dfi = pd.read_csv(f"data/ds1.csv",chunksize=32)
odf = pd.DataFrame(columns = ["you_as_subject", "c2_thankfulness"])

def id2filename(i):
    return f"data/ds1-embeddings/{i}.pt"

def is_you_subject(doc):
    for word in doc.sentences[0].words:
        if word.text.lower() == 'you' and (word.deprel == 'nsubj' or word.deprel == 'nsubj:pass') :
            return 1
    return 0

nlp = stanza.Pipeline(lang='en', processors="tokenize,mwt,pos,lemma,depparse", nthreads=multiprocessing.cpu_count())

for df in dfi:
    t0 = time.perf_counter()
    df["object"] = df["json"].apply(json.loads)
    df["subname"] = df["object"].apply(lambda x: x["submission"]["name"])
    df["2lc-id"] = df["object"].apply(lambda x: x["2lc"]["id"])
    df["2lcbody"] = df["object"].apply(lambda x: x["2lc"]['body'])
    df["2lc-embedding-fname"] = df["2lc-id"].apply(id2filename)
    df["2lc-embedding"] = df["2lc-embedding-fname"].apply(torch.load).apply(lambda x : x.numpy())
    df["2lc-stanza"] = nlp.bulk_process(df["2lcbody"])
    inputs = df["2lcbody"].tolist()
    
    y_pred = cls.thankfulness(np.array(df["2lc-embedding"].tolist()), inputs)
    new_rows_df = pd.DataFrame({
        "you_as_subject": df["2lc-stanza"].apply(is_you_subject),
        "c2_thankfulness": y_pred,
        "2lc-id": df["2lc-id"],
        "subname": df["subname"],
    }, index=df.index)
    odf = pd.concat([odf, new_rows_df])
    t1 = time.perf_counter()
    print(f"processed {len(df)} docs in {t1-t0:.0f} seconds")

odf.to_csv("data/ds1-you-thank.csv")
