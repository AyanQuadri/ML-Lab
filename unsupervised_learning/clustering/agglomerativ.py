# Import the required libraries
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs, load_iris
import numpy as np
import matplotlib.pyplot as plt

def demo_with_synthetic_data():
    """Demo with synthetic blob data"""
    print("=== Agglomerative Clustering with Synthetic Data ===")
    
    # Generate some sample data
    X, y_true = make_blobs(n_samples=100, centers=3, random_state=0, cluster_std=1.5)
    
    print(f"Generated {X.shape[0]} samples with {X.shape[1]} features")
    print(f"True clusters: {len(np.unique(y_true))}")
    print()
    
    # Create an instance of AgglomerativeClustering
    agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
    
    # Fit the model to the data
    agg_clustering.fit(X)
    
    # Get the cluster labels for each data point
    labels = agg_clustering.labels_
    
    print("Agglomerative Clustering Results:")
    print(f"Number of clusters: {agg_clustering.n_clusters_}")
    print()
    
    # Print the cluster information
    for cluster_id in range(3):
        cluster_mask = labels == cluster_id
        cluster_data = X[cluster_mask]
        print(f"Cluster {cluster_id + 1}:")
        print(f"  Number of points: {len(cluster_data)}")
        print(f"  Center (mean): {np.mean(cluster_data, axis=0)}")
    
    # Calculate accuracy compared to true clusters
    from sklearn.metrics import adjusted_rand_score
    ari_score = adjusted_rand_score(y_true, labels)
    print(f"\nAdjusted Rand Index (similarity to true clusters): {ari_score:.4f}")
    print()

def demo_with_iris_data():
    """Demo with Iris dataset"""
    print("=== Agglomerative Clustering with Iris Dataset ===")
    
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    
    print(f"Iris dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print()
    
    # Try different linkage methods
    linkage_methods = ['ward', 'complete', 'average', 'single']
    
    for linkage in linkage_methods:
        print(f"Linkage method: {linkage}")
        
        # Create and fit agglomerative clustering
        if linkage == 'ward':
            agg_clustering = AgglomerativeClustering(n_clusters=3, linkage=linkage)
        else:
            # Ward requires Euclidean distance, others can use different metrics
            agg_clustering = AgglomerativeClustering(n_clusters=3, linkage=linkage)
        
        agg_clustering.fit(X)
        labels = agg_clustering.labels_
        
        # Calculate how well clusters match actual species
        cluster_species_mapping = {}
        for cluster_id in range(3):
            cluster_mask = labels == cluster_id
            actual_species = iris.target[cluster_mask]
            most_common_species = np.bincount(actual_species).argmax()
            cluster_species_mapping[cluster_id] = most_common_species
        
        predicted_species = np.array([cluster_species_mapping[cluster] for cluster in labels])
        accuracy = np.mean(predicted_species == iris.target)
        
        print(f"  Accuracy (mapped to species): {accuracy:.4f}")
        
        # Show cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"  Cluster sizes: {dict(zip(unique_labels, counts))}")
        print()

def main():
    demo_with_synthetic_data()
    demo_with_iris_data()

if __name__ == "__main__":
    main()