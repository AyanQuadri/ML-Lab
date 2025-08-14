import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md("# K-means Clustering on Iris Dataset")
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.cluster import KMeans
    return KMeans, load_iris, np, plt


@app.cell
def _(load_iris):
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    feature_names = iris.feature_names

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {feature_names}")
    return X, feature_names, iris


@app.cell
def _(KMeans, X):
    # K-means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)

    kmeans_labels = kmeans.labels_
    kmeans_centers = kmeans.cluster_centers_

    print(f"Inertia: {kmeans.inertia_:.2f}")
    return kmeans_centers, kmeans_labels


@app.cell
def _(X, kmeans_centers, kmeans_labels):
    # Display clusters
    print("K-means Clusters:")
    for cluster_num in range(3):
        cluster_data = X[kmeans_labels == cluster_num]
        print(f"\nCluster {cluster_num + 1}:")
        print(f"Number of instances: {len(cluster_data)}")
        print(f"Cluster center: {kmeans_centers[cluster_num]}")
        print("Sample instances:")
        print(cluster_data[:5])
    return


@app.cell
def _(X, feature_names, kmeans_centers, kmeans_labels, plt):
    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = ['red', 'blue', 'green']
    for cluster_idx in range(3):
        points = X[kmeans_labels == cluster_idx]
        ax.scatter(points[:, 2], points[:, 3], 
                  c=colors[cluster_idx], label=f'Cluster {cluster_idx+1}', alpha=0.7)

    # Plot centroids
    ax.scatter(kmeans_centers[:, 2], kmeans_centers[:, 3], 
              c='black', marker='x', s=200, linewidths=3, label='Centroids')

    ax.set_xlabel(feature_names[2])
    ax.set_ylabel(feature_names[3])
    ax.set_title('K-means Clustering: Petal Length vs Petal Width')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(iris, kmeans_labels, np):
    # Calculate accuracy
    cluster_species = {}
    for c_id in range(3):
        species_in_cluster = iris.target[kmeans_labels == c_id]
        most_common = np.bincount(species_in_cluster).argmax()
        cluster_species[c_id] = most_common

    predicted = np.array([cluster_species[c] for c in kmeans_labels])
    accuracy = np.mean(predicted == iris.target)

    print(f"Clustering accuracy: {accuracy:.4f}")
    print("Cluster mappings:")
    for c_id, species_id in cluster_species.items():
        print(f"Cluster {c_id + 1} → {iris.target_names[species_id]}")
    return


if __name__ == "__main__":
    app.run()
