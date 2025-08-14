from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

# Generate some sample data
X, y = make_blobs(n_samples=100, centers=3, random_state=0)

# Create an instance of AgglomerativeClustering
agg_clustering = AgglomerativeClustering(n_clusters=3)

# Fit the model to the data
agg_clustering.fit(X)

# Get the cluster labels for each data point
labels = agg_clustering.labels_

# Print the cluster labels
print(labels)