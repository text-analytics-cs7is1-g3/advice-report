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
# import polynomial_animosity as polynomial
import polynomial

def report(model,  X_test, y_test):
    y_pred_test = model.predict(X_test)
    return {
        'accuracy':    round(accuracy_score(y_test, y_pred_test), 3),
        'precision':   round(precision_score(y_test, y_pred_test), 3),
        'recall':      round(recall_score(y_test, y_pred_test), 3),
        'f1':          round(f1_score(y_test, y_pred_test), 3),
    }


print("# # # # # # # # animosity")
documents = features.df["Document"][:-4]
Xa = polynomial.X_animosity_poly[:-4]
ya = features.y_animosity[:-4]
print(Xa.shape, ya.shape)

print(Xa.loc[0], documents[0])
X_train, X_test, y_train, y_test = train_test_split(Xa, ya, test_size=0.2, random_state=43, shuffle=True, stratify=ya)
docs_train, docs_test, y_train_doc, y_test_doc = train_test_split(documents, ya, test_size=0.2, random_state=43, shuffle=True, stratify=ya)

print(len(docs_test), y_test_doc.shape)
docs_test = pd.concat([docs_test, features.df["Document"][-4:]])
y_test = np.concatenate([y_test, features.y_animosity[-4:]])
X_test = np.concatenate([X_test, polynomial.X_animosity_poly[-4:]])

mf = DummyClassifier(strategy="most_frequent")
mf.fit(X_train, y_train)
model = LogisticRegression(penalty='l1', solver="liblinear", C=10**4, random_state=42, fit_intercept=True)
model.fit(X_train, y_train)
repa = report(model, X_test, y_test)
print(repa)
duma = report(mf, X_test, y_test)
print(duma)

X_test_df = pd.DataFrame(X_test, columns=Xa.columns)
indices_test = X_test_df.index.values
documents_test = documents.loc[indices_test]

thankfulness_preds = pd.DataFrame({
    "doc": docs_test,
    "logreg": model.predict(X_test),
    "mf": mf.predict(X_test),
    "manual": y_test,
    })
print(thankfulness_preds)

thankfulness_preds.to_csv("animosity-eval-review.csv")


print("# # # # # # # thankfulness")
Xt = polynomial.X_thankfulness_poly[:-4]
yt = features.y_thankfulness[:-4]

X_train, X_test, y_train, y_test = train_test_split(Xt, yt, test_size=0.2, random_state=43, shuffle=True, stratify=yt)
docs_train, docs_test, y_train_doc, y_test_doc = train_test_split(documents, yt, test_size=0.2, random_state=43, shuffle=True, stratify=yt)


# include NTA ESH YTA and NAH in test set
docs_test = pd.concat([docs_test, features.df["Document"][-4:]])
y_test = np.concatenate([y_test, features.y_thankfulness[-4:]])
X_test = np.concatenate([X_test, polynomial.X_thankfulness_poly[-4:]])

mf = DummyClassifier(strategy="most_frequent")
mf.fit(X_train, y_train)
model = LogisticRegression(penalty='l1', solver="liblinear", C=10**4, random_state=42, fit_intercept=True)
model.fit(X_train, y_train)
rept = report(model, X_test, y_test)
print(rept)
dumt = report(mf, X_test, y_test)
print('dummy', dumt)

X_test_df = pd.DataFrame(X_test, columns=Xt.columns)
indices_test = X_test_df.index.values
documents_test = documents.loc[indices_test]

animosity_preds = pd.DataFrame({
    "doc": docs_test,
    "logreg": model.predict(X_test),
    "mf": mf.predict(X_test),
    "manual": y_test,
    })

animosity_preds.to_csv("thankfulness-eval-review.csv")


df = pd.DataFrame({
    'animosity most frequent': duma,
    'animosity logreg': repa,
    'thankfulness most frequent': dumt,
    'thankfulness logreg': rept,
    })
df.to_csv("fig/evals.csv")
