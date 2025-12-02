# ML LAB Assignment
#=================1. KNN Classification=========================================
# Dataset: Breast Cancer Wisconsin Dataset (breast_cancer.csv)
# Question:
# Using the Breast Cancer Wisconsin dataset, build a K-Nearest Neighbors (KNN) classifier to predict whether a tumor is benign or malignant.
# Tasks:
#     I. Load and preprocess the dataset
#     II. Split the dataset into training and test sets.
#     III. Train KNN Classification models with different values of k (e.g., 3, 5, 7, 11).
#     IV. Evaluate each model using:
#     a. Accuracy
#     b. Precision
#     c. Recall
#     d. F1-score

#==================2. KNN Regression============================================
# Dataset: Boston Housing Dataset (boston_housing.csv)
# Question:
# Using the Boston Housing dataset, build a K-Nearest Neighbors (KNN) regression model to predict the median house value.
# Tasks:
#     I. Load and preprocess the dataset
#     II. Split the data into training and test sets.
#     III. Train KNN Regressors using different values of k (e.g., 2, 5, 10, 20).
#     IV. Evaluate performance using:
#     a. Mean Absolute Error (MAE)
#     b. Mean Squared Error (MSE)


import numpy as np
import pandas as pd
from collections import Counter

############################################################
# DISTANCE FUNCTION
############################################################

def euclidean(a, b):
    return np.sqrt(np.sum((a - b)**2))

############################################################
# KNN CLASSIFIER
############################################################

class KNNClassifier:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict_one(self, x):
        distances = [euclidean(x, p) for p in self.X]
        idx = np.argsort(distances)[:self.k]
        top_labels = self.y[idx]
        return Counter(top_labels).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

############################################################
# KNN REGRESSOR
############################################################

class KNNRegressor:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict_one(self, x):
        distances = [euclidean(x, p) for p in self.X]
        idx = np.argsort(distances)[:self.k]
        return np.mean(self.y[idx])

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

############################################################
# METRICS
############################################################

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp + 1e-12)

def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn + 1e-12)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-12)

# Regression metrics

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

############################################################
# TRAIN-TEST SPLIT
############################################################

def train_test_split(X, y, test_size=0.2):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(len(X)*(1-test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

############################################################
# NORMALIZATION
############################################################

def normalize(X):
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    return (X - mn) / (mx - mn + 1e-12)

############################################################
# PART 1: KNN CLASSIFICATION - BREAST CANCER
############################################################

cancer = pd.read_csv('breast_cancer.csv')

# Assuming last column = label
X = cancer.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').fillna(0).values
y = cancer.iloc[:, -1].values

X = normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2)

ks = [3, 5, 7, 11]

for k in ks:
    model = KNNClassifier(k=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"\nK = {k}")
    print("Accuracy:", accuracy(y_test, preds))
    print("Precision:", precision(y_test, preds))
    print("Recall:", recall(y_test, preds))
    print("F1-score:", f1_score(y_test, preds))

############################################################
# PART 2: KNN REGRESSION - BOSTON HOUSING
############################################################

bh = pd.read_csv('housing.csv')

X = bh.iloc[:, :-1].values
y = bh.iloc[:, -1].values

X = normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2)

ks_reg = [2, 5, 10, 20]

for k in ks_reg:
    model = KNNRegressor(k=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"\nK = {k}")
    print("MAE:", mae(y_test, preds))
    print("MSE:", mse(y_test, preds))