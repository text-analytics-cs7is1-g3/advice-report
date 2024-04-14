import pandas as pd
from scipy.sparse import hstack
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

folds = 50
X = polynomial.X_animosity_poly
y = features.y_animosity

X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=43, shuffle=True, stratify=y)

def report(model, X_val, y_val, X_test, y_test, i=None, C=None):
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    return {
        'accuracy_val': accuracy_score(y_val, y_pred_val),
        'accuracy_test': accuracy_score(y_test, y_pred_test),
        # 'precision': precision_score(y_val, y_pred),
        # 'recall': recall_score(y_val, y_pred),
        # 'f1': f1_score(y_val, y_pred),
        'i': i,
        'C': C,
    }

results = []
dummy_results_mf = []

for trainm, valm in StratifiedKFold(n_splits=folds).split(X_train, y_train):
    X_t, X_val = X_train[trainm], X_train[valm]
    y_t, y_val = y_train[trainm], y_train[valm]


    for i, C in enumerate(10.0 ** np.array(list(range(-5, 8)))):
        model = LogisticRegression(penalty='l1', solver="liblinear", C=C, fit_intercept=True)
        model.fit(X_t, y_t)
        y_pred = model.predict(X_val)
        results.append(report(model, X_val, y_val, X_test, y_test, i=i, C=C))

    mf = DummyClassifier(strategy="most_frequent")
    mf.fit(X_t, y_t)
    dummy_results_mf.append(report(mf, X_val, y_val, X_test, y_test))

df = pd.DataFrame(results)
df.to_csv("cv_animosity.csv")

df_mf = pd.DataFrame(dummy_results_mf)
df_mf.to_csv("dummy_mf_animosity.csv")

import manual_train_cls_animosity_vis
