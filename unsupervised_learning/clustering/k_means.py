# Import required libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def main():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    
    print("Dataset Info:")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {iris.feature_names}")
    print()
    
    # K-means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)
    
    # Interpretation of K-means Clusters
    kmeans_labels = kmeans.labels_
    kmeans_cluster_centers = kmeans.cluster_centers_
    
    print("K-means Clustering Results:")
    print(f"Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")
    print()
    
    print("K-means Clusters:")
    for cluster_id in range(3):
        cluster_data = X[kmeans_labels == cluster_id]
        print(f"\nCluster {cluster_id + 1}:")
        print(f"Number of instances: {len(cluster_data)}")
        print(f"Cluster center: {kmeans_cluster_centers[cluster_id]}")
        
        # Show which actual iris species are in this cluster
        actual_species_in_cluster = iris.target[kmeans_labels == cluster_id]
        unique_species, counts = np.unique(actual_species_in_cluster, return_counts=True)
        print("Actual species distribution in this cluster:")
        for species_id, count in zip(unique_species, counts):
            print(f"  {iris.target_names[species_id]}: {count}")
    
    # Calculate cluster purity (how well clusters match actual species)
    print(f"\nCluster Labels: {kmeans_labels[:20]}...")  # Show first 20
    print(f"Actual Labels:  {iris.target[:20]}...")      # Show first 20
    
    # Show accuracy if we map clusters to most common species
    cluster_to_species = {}
    for cluster_id in range(3):
        cluster_mask = kmeans_labels == cluster_id
        actual_species_in_cluster = iris.target[cluster_mask]
        most_common_species = np.bincount(actual_species_in_cluster).argmax()
        cluster_to_species[cluster_id] = most_common_species
    
    predicted_species = np.array([cluster_to_species[cluster] for cluster in kmeans_labels])
    accuracy = np.mean(predicted_species == iris.target)
    print(f"\nClustering accuracy (mapped to closest species): {accuracy:.4f}")

if __name__ == "__main__":
    main()