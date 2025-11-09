# Question 1: Perform k-means clustering on the given dataset for k values ranging from 2 to 12.
# 1. For each value of k, compute and record the average silhouette score.
# 2. Additionally, plot the silhouette distribution for k = 2, 4, and 8 to visualize how cluster quality varies.

# Question 2: Elbow
# Apply k-means clustering for k values from 1 to 12.
# For each k, calculate the within-cluster sum of squares (WCSS) and plot the elbow curve.

# ------------------- LAB 6 -------------------
# Question 1: K-Means with Silhouette Scores
# Question 2: Elbow Method
# --------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure the script runs from same folder
os.chdir(os.path.dirname(__file__))

# Load dataset
data = pd.read_csv("Seed_Data.csv")

# Drop label column if exists
if 'Type' in data.columns or 'type' in data.columns:
    X = data.drop(['Type'], axis=1).values
else:
    X = data.values

# Standardize features manually
X = (X - X.mean(axis=0)) / X.std(axis=0)

# ---------------- K-MEANS IMPLEMENTATION ----------------

def initialize_centroids(X, k):
    idx = np.random.choice(len(X), k, replace=False)
    return X[idx]

def assign_clusters(X, centroids):
    distances = np.sqrt(((X[:, None, :] - centroids[None, :, :])**2).sum(axis=2))
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        if np.any(labels == i):
            centroids[i] = X[labels == i].mean(axis=0)
    return centroids

def kmeans(X, k=3, max_iter=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iter):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    # Compute WCSS
    wcss = np.sum((X - centroids[labels])**2)
    return labels, centroids, wcss

# ---------------- SILHOUETTE SCORE ----------------
def silhouette_score_manual(X, labels):
    n = len(X)
    unique_labels = np.unique(labels)
    sil_scores = np.zeros(n)

    for i in range(n):
        same_cluster = X[labels == labels[i]]
        a = np.mean(np.sqrt(np.sum((same_cluster - X[i])**2, axis=1))) if len(same_cluster) > 1 else 0
        b = np.inf
        for lbl in unique_labels:
            if lbl == labels[i]:
                continue
            other_cluster = X[labels == lbl]
            dist = np.mean(np.sqrt(np.sum((other_cluster - X[i])**2, axis=1)))
            if dist < b:
                b = dist
        sil_scores[i] = (b - a) / max(a, b)
    return np.mean(sil_scores), sil_scores

# ==========================================================
#              QUESTION 1 — SILHOUETTE ANALYSIS
# ==========================================================

print("\n===== Question 1: Silhouette Analysis =====")

K_values = range(2, 13)
silhouette_avgs = []

for k in K_values:
    labels, centroids, _ = kmeans(X, k)
    avg_sil, _ = silhouette_score_manual(X, labels)
    silhouette_avgs.append(avg_sil)
    print(f"K={k}, Average Silhouette Score={avg_sil:.4f}")

# Plot average silhouette score vs K
plt.figure(figsize=(7,4))
plt.plot(K_values, silhouette_avgs, marker='o')
plt.title("Average Silhouette Score vs Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Average Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot silhouette distribution for k = 2, 4, 8
for k in [2, 4, 8]:
    labels, centroids, _ = kmeans(X, k)
    _, sil_samples = silhouette_score_manual(X, labels)
    y_lower = 0
    plt.figure(figsize=(6,4))
    for i in range(k):
        cluster_silhouette_vals = sil_samples[labels == i]
        cluster_silhouette_vals.sort()
        y_upper = y_lower + len(cluster_silhouette_vals)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals)
        y_lower = y_upper
    plt.axvline(x=np.mean(sil_samples), color='red', linestyle='--')
    plt.title(f"Silhouette Plot for k={k}")
    plt.xlabel("Silhouette Coefficient Values")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.show()

# ==========================================================
#              QUESTION 2 — ELBOW METHOD
# ==========================================================

print("\n===== Question 2: Elbow Method =====")

wcss_values = []
K_values = range(1, 13)

for k in K_values:
    labels, centroids, wcss = kmeans(X, k)
    wcss_values.append(wcss)
    print(f"K={k}, WCSS={wcss:.2f}")

# Plot elbow curve
plt.figure(figsize=(7,4))
plt.plot(K_values, wcss_values, marker='o')
plt.title("Elbow Method: WCSS vs Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nAll clustering and visualization completed successfully!")