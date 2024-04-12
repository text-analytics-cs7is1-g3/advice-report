import pandas as pd
from scipy.sparse import hstack
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures
import manual_bow
import features
import nltk
nltk.download('stopwords')

X = hstack([features.X_bert, features.X_tfidf, features.X_bow])
y = features.y_animosity
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=None, random_state=42, shuffle=True, stratify=y)

model = LogisticRegression(penalty='l1', solver="liblinear", C=2, fit_intercept=True)
model.fit(X_train, y_train)

def feature_type(i):
    if i < features.X_bert.shape[1]:
        return "BERT"
    elif i < features.X_bert.shape[1] + features.X_tfidf.shape[1]:
        return "TFIDF"
    else:
        return "BOW"

indices = []
for i, c in enumerate(model.coef_[0]):
    if c != 0.0:
        indices.append(i)
        print(feature_type(i), i)

mask = model.coef_[0] != 0.0
X_train_best = X_train[:, mask]
X_test_best = X_test[:, mask]

# Define feature names
feature_names = [f"BERT_{i+1}" for i in indices]

# Create polynomial features with degree 2
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly_train = poly.fit_transform(X_train_best)
X_poly_test = poly.transform(X_test_best)

if __name__ == "__main__":
    print("X_poly_train", X_poly_train.shape)
    print("X_poly_test", X_poly_test.shape)
