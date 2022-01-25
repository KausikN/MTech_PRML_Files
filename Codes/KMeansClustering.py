'''
K Means Clustering Algorithm
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Main Functions
# KMeans
def KMeansClustering(X, C, max_iterations=100):
    '''
    KMeans Clustering Algorithm
    '''
    # Initialize
    m = X.shape[0]
    # Initialize Cluster Assignments
    cluster_assignments = np.zeros((m, 1))
    # Initialize Centroids
    centroids = C
    # Initialize Error
    error = np.zeros((max_iterations, 1))
    # Init Old Centroids
    old_centroids = C

    # Run KMeans
    for iteration in range(max_iterations):
        old_centroids = np.copy(centroids)
        # Update Cluster Assignments
        for i in range(m):
            # Initialize Distance
            distance = np.zeros((centroids.shape[0], 1))
            # Calculate Distance
            for j in range(centroids.shape[0]):
                distance[j] = np.linalg.norm(X[i] - centroids[j])
            # Assign Cluster
            cluster_assignments[i] = np.argmin(distance)
        # Update Centroids
        for k in range(centroids.shape[0]):
            # Initialize Cluster
            cluster = X[np.where(cluster_assignments == k)]
            # Update Centroid
            centroids[k] = np.mean(cluster, axis=0)
        # Update Error
        error[iteration] = np.sum(np.power(cluster_assignments - np.arange(cluster_assignments.shape[0]), 2))
        # Check Convergence
        if np.array_equal(old_centroids, centroids):
            break
        # Update Iteration
        iteration += 1
        # Print
        print("IT " + str(iteration))
        print("Centroids: " + str(centroids))
        print("cluster_assignments: " + str(cluster_assignments))
        print()

    # Return
    return centroids, cluster_assignments, error

# Driver Code
# Params
X = [
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
]

C = [
    [0, 0],
    [10, 10],
]
# Params

# RunCode
X = np.array(X)
C = np.array(C)

centroids, cluster_assignments, error = KMeansClustering(X, C)
print(centroids)
print(cluster_assignments)