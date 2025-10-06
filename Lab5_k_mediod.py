# Question 2: K-Medoids Clustering
# Part A
# Dataset: [1, 2, 2, 3, 10, 11, 12]
# Tasks:
# • Run K-Medoids with k=2, using Manhattan distance.
# • Start with initial medoids {2, 11}.
# • Assign points to nearest medoid and compute total cost.
# • Report final medoids and clusters.
# Part B
# • Import the Iris dataset.
# • Select features sepal length and petal length.
# • Apply K-Medoids with k=3 using Manhattan distance.
# • Report final medoids, total cost, and cluster assignments.
# • Visualize the clusters.

#actual point in the cluster whose total distance to all other points in cluster is minimal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
data_partB = iris.data[:, [0, 2]]

def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))

def kmedoids(data, k, initial_medoids, max_iters=100):
    medoid_indices = initial_medoids[:]
    medoids = data[medoid_indices]
    
    for _ in range(max_iters):
        
        distances = np.array([[manhattan_distance(point, medoid) for medoid in medoids] for point in data])
        labels = np.argmin(distances, axis=1)
        
        new_medoids = []
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) == 0:
                new_medoids.append(medoids[i])
                continue
            
            costs = [np.sum([manhattan_distance(point, candidate) for point in cluster_points]) for candidate in cluster_points]
            new_medoids.append(cluster_points[np.argmin(costs)])
        new_medoids = np.array(new_medoids)
        
        if np.all(medoids == new_medoids):
            break   
        medoids = new_medoids
        
    total_cost = np.sum([manhattan_distance(data[i], medoids[labels[i]]) for i in range(len(data))])
    return labels, medoids, total_cost

# Part A
data_partA = np.array([[1], [2], [2], [3], [10], [11], [12]])
labels_A, medoids_A, cost_A = kmedoids(data_partA, k=2, initial_medoids=[1, 5])
print("Part A - Final Medoids:", medoids_A.ravel())
print("Part A - Final Cluster Assignments:", labels_A)
print("Part A - Total Cost:", cost_A)

labels_B, medoids_B, cost_B = kmedoids(data_partB, k=3, initial_medoids=[0, 50, 100])
print("Part B - Final Medoids:", medoids_B)
print("Part B - Final Cluster Assignments:", np.bincount(labels_B))
print("Part B - Total Cost:", cost_B)

plt.scatter(data_partB[:, 0], data_partB[:, 1], c=labels_B, cmap='viridis')
plt.scatter(medoids_B[:, 0], medoids_B[:, 1], s=200, c='red', marker='X')
plt.title('K-Medoids Clustering Part B')        
plt.show()

