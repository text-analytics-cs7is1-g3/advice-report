import pandas as pd
import bert_cls
import torch


def id2filename(i):
    return f"data/man-embeddings/{i}.pt"

def save(embedding, row):
    torch.save(embedding, id2filename(row['id']))

dfi = pd.read_csv("data/manual.csv", chunksize=428)
for df in dfi:
    texts = df["Document"].tolist()
    embeddings = bert_cls.four_layer_embeddings(texts)

    for i in range(len(df)):
        embedding = embeddings[i]
        row = df.iloc[i]
        save(embedding, row)
    
