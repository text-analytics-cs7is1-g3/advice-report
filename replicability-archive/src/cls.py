import pandas as pd
from scipy.sparse import hstack
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold
import manual_bow
import features
import bert_cls
# import polynomial_animosity as polynomial
import polynomial

def report(model,  X_test, y_test):
    y_pred_test = model.predict(X_test)
    return {
        'accuracy_test': accuracy_score(y_test, y_pred_test),
        'precision_test': precision_score(y_test, y_pred_test),
        'recall_test': recall_score(y_test, y_pred_test),
        'f1_test': f1_score(y_test, y_pred_test),
    }

Xa = polynomial.X_animosity_poly
ya = features.y_animosity

model_animosity = LogisticRegression(penalty='l1', solver="liblinear", C=10**4, fit_intercept=True, random_state=42)
model_animosity.fit(Xa, ya)


Xt = polynomial.X_thankfulness_poly
yt = features.y_thankfulness

model_thankfulness = LogisticRegression(penalty='l1', solver="liblinear", C=10**4, fit_intercept=True, random_state=42)
model_thankfulness.fit(Xt, yt)

def animosity(comment_embeddings, comment_texts):
    bow_features = features.bow.transform(comment_texts)
    X_poly, X_in = polynomial.transform_animosity(comment_embeddings, bow_features)
    y_pred= model_animosity.predict(X_poly)
    return y_pred

def animosity_str(string):
    X_bert = bert_cls.four_layer_embeddings([string])
    bow_features = features.bow.transform([string])
    X_poly, X_in = polynomial.transform_animosity(X_bert, bow_features)
    y_pred= model_animosity.predict(X_poly)
    return y_pred

def thankfulness(comment_embeddings, comment_texts):
    bow_features = features.bow.transform(comment_texts)
    print(comment_embeddings.shape)
    X_poly, X_in = polynomial.transform_thankfulness(comment_embeddings, bow_features)
    y_pred= model_thankfulness.predict(X_poly)
    return y_pred

if __name__ == "__main__":
    # features.df
    # apred = animosity(features.X_bert, features.df["Document"])
    # tpred = thankfulness(features.X_bert, features.df["Document"])
    # df = pd.DataFrame({
    #     "animosity predictions": apred,
    #     "thankfulness predictions": tpred,})
    # print(df["animosity predictions"].mean())
    # print(df["thankfulness predictions"].mean())
    # df.to_csv("fig/sanity-check-predictions.csv")

    for s in ["Definitely NTA.", "NTA", "YTA", "NAH", "Nta", "ESH", "That's kind, thank you!"]:
        print(s, animosity_str(s))
