# Question 1: K-Means Clustering
# Part A
# Dataset: (1,2), (1,4), (1,0), (10,2), (10,4), (10,0)
# Tasks:
# • Apply K-Means clustering with k=2.
# • Use initial centroids at (1,2) and (10,2).
# • Assign each point to the nearest centroid and update centroids.
# • Print final cluster assignments and centroids.
# • Plot the points and centroids.
# Part B
# • Import the Iris dataset from sklearn.datasets.
# • Select features sepal length and petal length.
# • Apply K-Means with k=3.
# • Report cluster centroids and sizes.
# • Visualize the clusters in 2D.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Part A

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))  #calculates euclidean distance between two points

def kmeans(data, k, initial_centroids, max_iters=100): #initial centroid is [1,2] and [10,2]
    centroids = np.array(initial_centroids, dtype=float)
    for _ in range(max_iters):
        # Assign clusters
        clusters = [[] for _ in range(k)]
        for point in data:
            distance = np.array([[euclidean_distance(point, centroid) for centroid in centroids] for point in data])
            labels = np.argmin(distance, axis=1) #returns the index, tells the point to belong to cluster x because centroid x is nearest to it
            
            new_centroids = np.array([data[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(k)])
            
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids
    
    return labels, centroids

data_partA = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
# k is number of clusters that we want

labels_A, centroids_A = kmeans(data_partA, k=2, initial_centroids=[[1, 2], [10, 2]])
print("Part A - Final Cluster Assignments:", labels_A)
print("Part A - Final Centroids:", centroids_A)

plt.scatter(data_partA[:, 0], data_partA[:, 1], c=labels_A, cmap='viridis')
plt.scatter(centroids_A[:, 0], centroids_A[:, 1], s=200, c='red', marker='X')
plt.title('K-Means Clustering Part A')  
plt.show()

# part b 

iris = load_iris()
data_partB = iris.data[:, [0, 2]]  # sepal length and petal length
labels_B, centroids_B = kmeans(data_partB, k=3, initial_centroids = data_partB[:3])  # using first 3 points as initial centroids
print("Part B - Final Centroids:", centroids_B)
print("Part B - Cluster Sizes:", np.bincount(labels_B))

plt.scatter(data_partB[:, 0], data_partB[:, 1], c=labels_B, cmap='viridis')
plt.scatter(centroids_B[:, 0], centroids_B[:, 1], s=200, c='red', marker='X')
plt.title('K-Means Clustering Part B')
plt.show()  