import pandas as pd
import torch
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfTransformer


def id2filename(i):
    return f"data/man-embeddings/{i}.pt"

df = pd.read_csv("data/manual-incomplete.csv")
df["embedding-filename"]    = df["id"].apply(id2filename)
df["embedding"]             = df["embedding-filename"].apply(torch.load).apply(lambda x: x.numpy())

X_counts                    = CountVectorizer().fit_transform(df["Document"])
X_bow                       = CountVectorizer(binary=True).fit_transform(df["Document"])
X_tf                        = TfidfTransformer(use_idf=False).fit(X_counts).transform(X_counts)
X_tfidf                     = TfidfTransformer(use_idf=True).fit(X_counts).transform(X_counts)
X_bert                      = np.array(df["embedding"].to_list())
y_thankfulness              = np.array(df[["mode thankfulness"]].values)[:,0]
y_animosity                 = np.array(df[["mode animosity"]].values)[:,0]

if __name__ == "__main__":
    print("X_train_counts", X_train_counts.shape, X_train_counts)
    print("X_train_bow", X_train_bow.shape, X_train_bow)
    print("X_train_tf", X_train_tf.shape)

