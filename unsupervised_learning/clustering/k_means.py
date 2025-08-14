# Import required libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

def main():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    
    # K-means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)
    
    # Interpretation of K-means Clusters
    kmeans_labels = kmeans.labels_
    kmeans_cluster_centers = kmeans.cluster_centers_
    
    print("K-means Clusters:")
    for cluster_id in range(3):
        cluster_data = X[kmeans_labels == cluster_id]
        print(f"Cluster {cluster_id + 1}:")
        print(f"Number of instances: {len(cluster_data)}")
        print(f"Cluster center: {kmeans_cluster_centers[cluster_id]}")
        print("Sample instances:")
        print(cluster_data[:5])  # Show first 5 instances

if __name__ == "__main__":
    main()