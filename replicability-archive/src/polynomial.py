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

X = hstack([features.X_bert, features.X_bow])

def feature_type(i):
    if i < features.X_bert.shape[1]:
        return "BERT_{" + str(i) + "}"
    else:
        return "BOW_{" + features.bow_feature_names[i - features.X_bert.shape[1]] + "}"

def poly_transformer(yname):
    y = None
    if yname == 'thankfulness':
        y = features.y_thankfulness
    elif yname == 'animosity':
        y = features.y_animosity
    else:
        raise Exception()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=None, random_state=42, shuffle=True, stratify=y)
    model = LogisticRegression(penalty='l1', solver="liblinear", C=2, fit_intercept=True, random_state=42)
    model.fit(X_train, y_train)
    indices = []
    for i, c in enumerate(model.coef_[0]):
        if c != 0.0:
            indices.append(i)
            print(feature_type(i), c)
    print("num features to include: ", len(indices))
    mask = model.coef_[0] != 0.0
    X_train_best = X_train[:, mask]
    X_test_best = X_test[:, mask]
    
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    columns = [feature_type(i) for i in indices]
    data = pd.DataFrame(X_train_best.toarray())
    data.columns = columns
    X_poly_train = poly.fit_transform(data)
    def transform(Xbert, Xbow):
        X_ = hstack([Xbert, Xbow])
        features_in = pd.DataFrame(X_.toarray()[:,mask])
        features_in.columns = columns
        features_out = pd.DataFrame(poly.transform(features_in))
        features_out.columns = poly.get_feature_names_out()
        return features_out, features_in
    return transform, mask, poly

transform_thankfulness, mask_thankfulness, poly_thankfulness = poly_transformer('thankfulness')
X_thankfulness_poly, X_thankfulness  = transform_thankfulness(features.X_bert, features.X_bow)

transform_animosity, mask_animosity, poly_animosity = poly_transformer('animosity')
X_animosity_poly, X_animosity = transform_animosity(features.X_bert, features.X_bow)

if __name__ == "__main__":
    print(X_thankfulness.shape)
    print(X_thankfulness_poly.shape)
    print(poly_thankfulness.get_feature_names_out())
    print(X_animosity.shape)
    print(X_animosity_poly.shape)
    print(poly_animosity.get_feature_names_out())
