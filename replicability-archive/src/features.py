import pandas as pd
import torch
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string
import nltk
import re
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def process_text(text):
    tokens = word_tokenize(text.lower())
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return stemmed_tokens


def id2filename(i):
    return f"data/man-embeddings/{i}.pt"

df = pd.read_csv("data/manual.csv")
print(df["Document"])
df["document-stemmed"]      = df["Document"].apply(process_text)
df["embedding-filename"]    = df["id"].apply(id2filename)
df["embedding"]             = df["embedding-filename"].apply(torch.load).apply(lambda x: x.numpy())

X_counts                    = CountVectorizer(stop_words=["nta","yta","esh","nah"]).fit_transform(df["Document"])
bow = CountVectorizer(binary=True,stop_words=["nta","yta","esh","nah"])
X_bow                       = bow.fit_transform(df["Document"])
bow_feature_names           = bow.get_feature_names_out()
X_tf                        = TfidfTransformer(use_idf=False).fit(X_counts).transform(X_counts)
X_tfidf                     = TfidfTransformer(use_idf=True).fit(X_counts).transform(X_counts)
X_bert                      = np.array(df["embedding"].to_list())
print(X_bert.shape)
y_thankfulness              = np.array(df[["mode thankfulness"]].values)[:,0]
y_animosity                 = np.array(df[["mode animosity"]].values)[:,0]

if __name__ == "__main__":
    print("X_counts", X_counts.shape)
    print("X_bow", X_bow.shape)
    print("X_tf", X_tf.shape)

