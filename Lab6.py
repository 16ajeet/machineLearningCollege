# Question 1: Perform k-means clustering on the given dataset for k values ranging from 2 to 12.
# 1. For each value of k, compute and record the average silhouette score.
# 2. Additionally, plot the silhouette distribution for k = 2, 4, and 8 to visualize how cluster quality varies.

# Question 2: Elbow
# Apply k-means clustering for k values from 1 to 12.
# For each k, calculate the within-cluster sum of squares (WCSS) and plot the elbow curve.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Helper functions ---

def kmeans(X, k, max_iter=100):
    # Randomly initialize centroids
    np.random.seed(42)
    print(f"this is row : {X.shape[0]}")
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iter):
        # Assign clusters
        #Calculates Euclidean distance from each point to each centroid.
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        #Assigns each point to the nearest centroid.
        labels = np.argmin(distances, axis=1)
        # Update centroids as mean of points in eacch cluster and if condition stops if centroid stops updating
        new_centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

def silhouette_score_manual(X, labels):
    #total number of datapoints -> row
    n = X.shape[0]
    #total number of clusters -> label
    k = len(np.unique(labels))
    #initializes an empty array of size n and value 0 in which we will store data
    sil_samples = np.zeros(n)
    for i in range(n):
        same_cluster = labels == labels[i]
        other_clusters = labels != labels[i]
        #calculating intracluster distance
        a = np.mean(np.linalg.norm(X[i] - X[same_cluster], axis=1)) if np.sum(same_cluster) > 1 else 0
        #calculating distance from other cluster and taking mean -> distance from closest neighbour
        b = np.min([
            np.mean(np.linalg.norm(X[i] - X[labels == j], axis=1))
            for j in np.unique(labels) if j != labels[i]
        ]) if k > 1 else 0
        sil_samples[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0
    return sil_samples.mean(), sil_samples

#for each cluster, sum squared distances of points to their centroid
# jitna cluster utna kam wcss -> after elbow point ye useless hoga 
def wcss_manual(X, labels, centroids):
    return sum(np.sum((X[labels == i] - centroids[i])**2) for i in range(len(centroids)))

# --- Load data ---
df = pd.read_csv('Seed_Data.csv')
X = df.drop(['target'], axis=1).values

# --- Question 1: Silhouette scores ---
silhouette_avgs = []
k_range = range(2, 13)
for k in k_range:
    labels, centroids = kmeans(X, k)
    sil_avg, sil_samples = silhouette_score_manual(X, labels)
    silhouette_avgs.append(sil_avg)
    if k in [2, 4, 8]:
        plt.figure(figsize=(7, 5))
        plt.title(f'Silhouette Distribution for k={k}', fontsize=14)
        plt.hist(sil_samples, bins=30, color='skyblue', edgecolor='black')
        plt.xlabel('Silhouette Coefficient', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'silhouette_k{k}.png', dpi=300, bbox_inches='tight')
        plt.close()

print("Average silhouette scores for k=2 to 12:")
for k, score in zip(k_range, silhouette_avgs):
    print(f"k={k}: {score:.4f}")

# --- Question 2: Elbow curve ---
wcss = []
k_range_elbow = range(1, 13)
for k in k_range_elbow:
    labels, centroids = kmeans(X, k)
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



