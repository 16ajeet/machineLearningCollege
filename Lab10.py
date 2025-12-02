#     1. Customer Churn Prediction (Classification)
# Lab Question 1:

# Given a customer dataset (Churn_Modelling.CSV) containing features such as account information, demographics, and usage statistics, build and train an Artificial Neural Network (ANN) to classify whether a customer will churn (leave) or not.
#     • Implement data pre-processing (scaling/encoding), model building, training, and evaluation.
#     • Visualize the confusion matrix for model performance.
#     • Experiment with varying network depth and activation functions, and evaluate accuracy.
# 2. Graduate Admission Prediction (Regression)
# Lab Question 2:

# Given a dataset (Admission_Predict_Ver1.1.CSV) containing student scores and GRE application data, build an ANN regression model to predict probability of admission.
#     • Apply feature normalization and split the data for training/testing.
#     • Train the regression model and visualize loss curves.
#     • Report the Mean Squared Error (MSE) on the test set.
#     • Experiment by tuning learning rate and number of neurons.


# FULL PYTHON CODE WITH MULTIPLE NETWORK DEPTH EXPERIMENTS (NO SKLEARN)
# Includes:
# 1. Customer Churn Prediction (Classification) + experiments with different ANN depths
# 2. Graduate Admission Prediction (Regression) + tuning example
# Only numpy, pandas, matplotlib used.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##########################################################
# UTILITY FUNCTIONS
##########################################################

def train_test_split(X, y, test_size=0.2):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(len(X) * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

def minmax_scale(X):
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    return (X - mn) / (mx - mn + 1e-12), mn, mx

def one_hot_encode(col):
    uniq = np.unique(col)
    mapping = {v:i for i,v in enumerate(uniq)}
    encoded = np.zeros((len(col), len(uniq)))
    for i,v in enumerate(col):
        encoded[i, mapping[v]] = 1
    return encoded

##########################################################
# BASIC ANN CLASS (from scratch)
##########################################################

class ANN:
    def __init__(self, layer_sizes, activations, lr=0.01):
        self.lr = lr
        self.layers = len(layer_sizes)
        self.activations = activations
        self.W = []
        self.b = []
        for i in range(self.layers - 1):
            self.W.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1)
            self.b.append(np.zeros((1, layer_sizes[i+1])))

    def act(self, x, mode):
        if mode == 'relu': return np.maximum(0, x)
        if mode == 'sigmoid': return 1/(1+np.exp(-x))
        return x

    def act_deriv(self, x, mode):
        if mode == 'relu': return (x > 0).astype(float)
        if mode == 'sigmoid':
            s = 1/(1+np.exp(-x))
            return s*(1-s)
        return np.ones_like(x)

    def forward(self, X):
        a = X
        zs = []
        acts = [X]
        for W, b, act in zip(self.W, self.b, self.activations):
            z = a @ W + b
            a = self.act(z, act)
            zs.append(z)
            acts.append(a)
        return zs, acts

    def backward(self, zs, acts, y, regression=False):
        grads_W = []
        grads_b = []

        delta = (acts[-1] - y)

        for i in reversed(range(self.layers - 1)):
            mode = self.activations[i]
            dz = delta * self.act_deriv(zs[i], mode)
            dw = acts[i].T @ dz
            db = np.sum(dz, axis=0, keepdims=True)

            grads_W.insert(0, dw)
            grads_b.insert(0, db)

            if i != 0:
                delta = dz @ self.W[i].T

        for i in range(self.layers - 1):
            self.W[i] -= self.lr * grads_W[i]
            self.b[i] -= self.lr * grads_b[i]

    def fit(self, X, y, epochs=200, regression=False):
        loss_list = []
        for ep in range(epochs):
            zs, acts = self.forward(X)
            if regression:
                loss = np.mean((acts[-1] - y)**2)
            else:
                loss = np.mean(-(y*np.log(acts[-1]+1e-12) + (1-y)*np.log(1-acts[-1]+1e-12)))
            loss_list.append(loss)
            self.backward(zs, acts, y, regression)
        return loss_list

    def predict(self, X):
        _, acts = self.forward(X)
        return acts[-1]

##########################################################
# PART 1: CUSTOMER CHURN (CLASSIFICATION)
##########################################################

churn = pd.read_csv('Churn_Modelling.csv')

X = churn[['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']]
y = churn['Exited'].values.reshape(-1,1)

geo = one_hot_encode(X['Geography'].values)
gen = one_hot_encode(X['Gender'].values)

num = X[['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']].values
X_all = np.concatenate([num, geo, gen], axis=1)
X_scaled, mn, mx = minmax_scale(X_all)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

##########################################################
# EXPERIMENT 1: BASELINE NETWORK
##########################################################

model1 = ANN(
    layer_sizes=[X_train.shape[1], 16, 8, 1],
    activations=['relu','relu','sigmoid'],
    lr=0.01
)
loss1 = model1.fit(X_train, y_train, epochs=200)
pred1 = (model1.predict(X_test) > 0.5).astype(int)
acc1 = np.mean(pred1 == y_test)
print("Accuracy (16-8):", acc1)

##########################################################
# EXPERIMENT 2: DEEPER NETWORK
##########################################################

model2 = ANN(
    layer_sizes=[X_train.shape[1], 32, 16, 8, 1],
    activations=['relu','relu','relu','sigmoid'],
    lr=0.01
)
loss2 = model2.fit(X_train, y_train, epochs=200)
pred2 = (model2.predict(X_test) > 0.5).astype(int)
acc2 = np.mean(pred2 == y_test)
print("Accuracy (32-16-8):", acc2)

##########################################################
# EXPERIMENT 3: SMALL NETWORK
##########################################################

model3 = ANN(
    layer_sizes=[X_train.shape[1], 8, 4, 1],
    activations=['relu','relu','sigmoid'],
    lr=0.01
)
loss3 = model3.fit(X_train, y_train, epochs=200)
pred3 = (model3.predict(X_test) > 0.5).astype(int)
acc3 = np.mean(pred3 == y_test)
print("Accuracy (8-4):", acc3)

##########################################################
# CONFUSION MATRIX FOR BEST MODEL
##########################################################

best_pred = pred2
cm = np.zeros((2,2), dtype=int)
for i in range(len(best_pred)):
    cm[y_test[i][0]][best_pred[i][0]] += 1
print("Confusion Matrix (Best Model):", cm)

plt.figure()
plt.plot(loss1, label='16-8')
plt.plot(loss2, label='32-16-8')
plt.plot(loss3, label='8-4')
plt.legend()
plt.title("Classification Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

##########################################################
# PART 2: ADMISSION PREDICTION (REGRESSION)
##########################################################

adm = pd.read_csv('Admission_Predict_Ver1.1.csv')

X = adm[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']].values
y = adm[['Chance of Admit ']].values

X_scaled, mn2, mx2 = minmax_scale(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

##########################################################
# REGRESSION MODEL EXPERIMENTS
##########################################################

reg1 = ANN(
    layer_sizes=[X_train.shape[1], 16, 8, 1],
    activations=['relu','relu','linear'],
    lr=0.01
)
loss_r1 = reg1.fit(X_train, y_train, epochs=200, regression=True)
pred_r1 = reg1.predict(X_test)
mse1 = np.mean((pred_r1 - y_test)**2)
print("Regression MSE (16-8):", mse1)

reg2 = ANN(
    layer_sizes=[X_train.shape[1], 32, 16, 8, 1],
    activations=['relu','relu','relu','linear'],
    lr=0.005
)
loss_r2 = reg2.fit(X_train, y_train, epochs=200, regression=True)
pred_r2 = reg2.predict(X_test)
mse2 = np.mean((pred_r2 - y_test)**2)
print("Regression MSE (32-16-8):", mse2)

plt.figure()
plt.plot(loss_r1, label='16-8')
plt.plot(loss_r2, label='32-16-8')
plt.title("Regression Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()
