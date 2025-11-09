# Question 1 — K-Modes Clustering on a Categorical Dataset

# Dataset: Adult Income Dataset (adult.csv)

# Task:
# Apply K-Modes clustering to group individuals based on categorical socioeconomic attributes.

# 	1.	Import the Adult Income dataset and preprocess it by:
# 	•	Selecting categorical features such as workclass, education, marital-status, and occupation.
# 	2.	Use the K-Modes algorithm to cluster the dataset.
# 	3.	Determine the optimal number of clusters using the cost function plot. 


# Question 2 — Bisecting K-Means on a Numerical Dataset

# Dataset: Mall Customers Dataset (mall_customers.csv)

# Task:
# Apply Bisecting K-Means clustering to segment customers based on their annual income and spending behavior.

# 	1.	Import the dataset and standardize numerical features such as Annual Income (k$) and Spending Score (1–100).
# 	2.	Perform clustering using Bisecting K-Means.
# 	3.	Compare the results with traditional K-Means using:
# 	•	Silhouette score
# 	4.	Visualize both clustering results using 2D scatter plots.

# ---------------------- LAB 8 ----------------------
# Question 1: K-Modes on Adult Dataset
# Question 2: Bisecting K-Means on Mall Customers Dataset
# --------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Run script from same folder
os.chdir(os.path.dirname(__file__))

# ==================================================
#               QUESTION 1 — K-MODES
# ==================================================

print("\n===== Question 1: K-MODES Clustering =====")

# Load Adult dataset
data = pd.read_csv("adult.csv")

# Fix column naming issues (dots, spaces, etc.)
data.columns = data.columns.str.strip().str.lower()

# Select categorical features
cat_features = ['workclass', 'education', 'marital.status', 'occupation']
data = data[cat_features].dropna()

# Encode categories as numbers
for col in data.columns:
    data[col] = data[col].astype('category').cat.codes

X = data.values

# --- K-MODES IMPLEMENTATION ---
def initialize_modes(X, k):
    idx = np.random.choice(len(X), k, replace=False)
    return X[idx]

def mode_of_columns(cluster_points):
    mode = []
    for i in range(cluster_points.shape[1]):
        values, counts = np.unique(cluster_points[:, i], return_counts=True)
        mode.append(values[np.argmax(counts)])
    return np.array(mode)

def kmodes(X, k, max_iter=10):
    modes = initialize_modes(X, k)
    for _ in range(max_iter):
        distances = np.array([[np.sum(x != mode) for mode in modes] for x in X])
        labels = np.argmin(distances, axis=1)
        new_modes = np.array([mode_of_columns(X[labels == i]) for i in range(k)])
        if np.all(modes == new_modes):
            break
        modes = new_modes
    cost = np.sum([np.sum(X[labels == i] != modes[i]) for i in range(k)])
    return labels, modes, cost

# --- Find Optimal K ---
costs = []
K = range(2, 9)
for k in K:
    _, _, cost = kmodes(X, k)
    costs.append(cost)

plt.figure(figsize=(6,4))
plt.plot(K, costs, marker='o')
plt.title("K-Modes Cost vs Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Cost")
plt.grid(True)
plt.tight_layout()
plt.show()

print("K-Modes completed successfully!\n")

# ==================================================
#        QUESTION 2 — BISECTING K-MEANS
# ==================================================

print("===== Question 2: Bisecting K-Means =====")

# Load Mall Customers dataset
data = pd.read_csv("Mall_Customers.csv.xls")

# Clean up column names
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('$', '')

print("Columns in Mall dataset:", data.columns.tolist())

# Select numerical features
X = data[['annual_income_k', 'spending_score_1-100']].values

# Standardize
X = (X - X.mean(axis=0)) / X.std(axis=0)

# --- Basic K-Means Functions ---
def initialize_centroids(X, k):
    idx = np.random.choice(len(X), k, replace=False)
    return X[idx]

def assign_clusters(X, centroids):
    distances = np.sqrt(((X[:, None, :] - centroids[None, :, :])**2).sum(axis=2))
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    centroids = []
    for i in range(k):
        centroids.append(X[labels == i].mean(axis=0))
    return np.array(centroids)

def kmeans(X, k=2, max_iter=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iter):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# --- Bisecting K-Means ---
def bisecting_kmeans(X, final_k=4):
    clusters = [np.arange(len(X))]
    labels = np.zeros(len(X), dtype=int)
    
    while len(clusters) < final_k:
        sizes = [len(c) for c in clusters]
        i = np.argmax(sizes)
        points_to_split = clusters.pop(i)
        sub_labels, _ = kmeans(X[points_to_split], k=2)
        clusters.insert(i, points_to_split[sub_labels == 0])
        clusters.insert(i + 1, points_to_split[sub_labels == 1])
    
    labels = np.zeros(len(X), dtype=int)
    for idx, cluster in enumerate(clusters):
        labels[cluster] = idx
    return labels

# --- Silhouette Score (simple version) ---
def silhouette_score(X, labels):
    n = len(X)
    sil = []
    for i in range(n):
        same = X[labels == labels[i]]
        other_clusters = [X[labels == l] for l in np.unique(labels) if l != labels[i]]
        a = np.mean(np.sqrt(np.sum((same - X[i])**2, axis=1)))
        b = np.min([np.mean(np.sqrt(np.sum((c - X[i])**2, axis=1))) for c in other_clusters])
        sil.append((b - a) / max(a, b))
    return np.mean(sil)

# --- Run both algorithms ---
labels_bisect = bisecting_kmeans(X, final_k=4)
labels_kmeans, _ = kmeans(X, k=4)

print("Silhouette (Bisecting K-Means):", silhouette_score(X, labels_bisect))
print("Silhouette (K-Means):", silhouette_score(X, labels_kmeans))

# --- Visualization ---
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1], c=labels_bisect, cmap='viridis')
plt.title("Bisecting K-Means")

plt.subplot(1,2,2)
plt.scatter(X[:,0], X[:,1], c=labels_kmeans, cmap='viridis')
plt.title("Traditional K-Means")

plt.tight_layout()
plt.show()

print("Bisecting K-Means completed successfully!\n")
