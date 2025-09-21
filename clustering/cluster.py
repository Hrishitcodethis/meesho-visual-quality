import hdbscan
import numpy as np


def cluster_embeddings(embeddings: np.ndarray, min_cluster_size=5):
clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
labels = clusterer.fit_predict(embeddings)
return labels