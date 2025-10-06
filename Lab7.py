# Question 1: Silhouette Score Analysis using K-Means++
# Tasks:
# 1. Load the Mall Customers dataset and extract only the numeric
# features Annual Income (k$) and Spending Score (1-100).
# 2. Apply K-Means++ clustering for values of k = 2 to 10.
# 3. For each k, compute and record the average silhouette score.
# 4. Plot the silhouette score vs. number of clusters (k).
# 5. Identify the k value that gives the highest silhouette score
# 6. Additionally, plot the silhouette distribution for k = 2, 4, and 6
# Question 2: Elbow Method for Optimal Cluster Selection
# Tasks:
# 1. Apply standard K-Means clustering for k = 1 to 12.
# 2. For each k, compute the Within-Cluster Sum of Squares (WCSS).
# 3. Plot the Elbow Curve (WCSS vs. k).
# 4. Identify the “elbow point” visually where the rate of decrease in
# WCSS slows down.
# 5. Compare whether the optimal cluster number obtained from the
# Elbow method aligns with that from the silhouette analysis in Question 1.
# Question 3: Comparison of Initialization Techniques
# Tasks:
# 1. For a fixed number of clusters k = 5, run both:
# • K-Means with random initialization, and
# • K-Means++ initialization.
# 2. For each method, record:
# • Final WCSS
# • Average silhouette score
# • Number of iterations until convergence
# 3. Plot and compare the cluster visualizations for both initialization
# methods.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Helper Functions ---

def kmeans_plus_plus_init(X, k):
    np.random.seed(42)
    centroids = []
    # Randomly choose the first centroid
    centroids.append(X[np.random.choice(X.shape[0])])
    for _ in range(1, k):
        # Compute distance from each point to nearest centroid
        dist_sq = np.array([min(np.sum((x - c)**2) for c in centroids) for x in X])
        probs = dist_sq / dist_sq.sum()
        next_idx = np.random.choice(X.shape[0], p=probs)
        centroids.append(X[next_idx])
    return np.array(centroids)

def kmeans(X, k, init='random', max_iter=100):
    np.random.seed(42)
    if init == 'random':
        centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    elif init == 'plus_plus':
        centroids = kmeans_plus_plus_init(X, k)
    else:
        raise ValueError("init must be 'random' or 'plus_plus'")
    for iteration in range(max_iter):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids, iteration+1

def silhouette_score_manual(X, labels):
    n = X.shape[0]
    k = len(np.unique(labels))
    sil_samples = np.zeros(n)
    for i in range(n):
        same_cluster = labels == labels[i]
        a = np.mean(np.linalg.norm(X[i] - X[same_cluster], axis=1)) if np.sum(same_cluster) > 1 else 0
        b = np.min([
            np.mean(np.linalg.norm(X[i] - X[labels == j], axis=1))
            for j in np.unique(labels) if j != labels[i]
        ]) if k > 1 else 0
        sil_samples[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0
    return sil_samples.mean(), sil_samples

def wcss_manual(X, labels, centroids):
    return sum(np.sum((X[labels == i] - centroids[i])**2) for i in range(len(centroids)))

# --- Load Data ---
df = pd.read_csv('Mall_Customers.csv.xls')
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# --- Question 1: K-Means++ Silhouette Analysis ---
silhouette_avgs = []
k_range = range(2, 11)
for k in k_range:
    labels, centroids, _ = kmeans(X, k, init='plus_plus')
    sil_avg, sil_samples = silhouette_score_manual(X, labels)
    silhouette_avgs.append(sil_avg)
    if k in [2, 4, 6]:
        plt.figure(figsize=(7, 5))
        plt.title(f'Silhouette Distribution for k={k} (K-Means++)', fontsize=14)
        plt.hist(sil_samples, bins=30, color='skyblue', edgecolor='black')
        plt.xlabel('Silhouette Coefficient', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'silhouette_k{k}_plusplus.png', dpi=300, bbox_inches='tight')
        plt.close()

plt.figure(figsize=(7, 5))
plt.plot(list(k_range), silhouette_avgs, marker='o', linewidth=2)
plt.title('Average Silhouette Score vs. Number of Clusters (K-Means++)', fontsize=14)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Average Silhouette Score', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('silhouette_vs_k.png', dpi=300, bbox_inches='tight')
plt.close()

best_k = k_range[np.argmax(silhouette_avgs)]
print(f"Best k by silhouette score: {best_k}")

# --- Question 2: Elbow Method (Standard K-Means) ---
wcss = []
k_range_elbow = range(1, 13)
for k in k_range_elbow:
    labels, centroids, _ = kmeans(X, k, init='random')
    wcss.append(wcss_manual(X, labels, centroids))

plt.figure(figsize=(7, 5))
plt.plot(list(k_range_elbow), wcss, marker='o', linewidth=2)
plt.title('Elbow Curve for K-Means', fontsize=14)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('WCSS (Inertia)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('elbow_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Question 3: Initialization Comparison (k=5) ---
k_fixed = 5

# K-Means random
labels_r, centroids_r, iter_r = kmeans(X, k_fixed, init='random')
wcss_r = wcss_manual(X, labels_r, centroids_r)
sil_r, _ = silhouette_score_manual(X, labels_r)

# K-Means++
labels_pp, centroids_pp, iter_pp = kmeans(X, k_fixed, init='plus_plus')
wcss_pp = wcss_manual(X, labels_pp, centroids_pp)
sil_pp, _ = silhouette_score_manual(X, labels_pp)

print(f"K-Means (random): WCSS={wcss_r:.2f}, Silhouette={sil_r:.4f}, Iterations={iter_r}")
print(f"K-Means++      : WCSS={wcss_pp:.2f}, Silhouette={sil_pp:.4f}, Iterations={iter_pp}")

# Cluster visualizations
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels_r, cmap='tab10', s=30)
plt.scatter(centroids_r[:, 0], centroids_r[:, 1], c='red', marker='X', s=100)
plt.title('K-Means (Random Init)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels_pp, cmap='tab10', s=30)
plt.scatter(centroids_pp[:, 0], centroids_pp[:, 1], c='red', marker='X', s=100)
plt.title('K-Means++ Init')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

plt.tight_layout()
plt.savefig('cluster_comparison.png', dpi=300, bbox_inches='tight')
plt.close()