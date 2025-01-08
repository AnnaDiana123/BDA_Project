import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix


def run_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    return kmeans, labels


def elbow_method(data, max_clusters=10, output_path=None):
    """
    Determine the optimal number of clusters using the elbow method and save the plot.

    Args:
        data (pd.DataFrame): Input data.
        max_clusters (int): Maximum number of clusters to evaluate.
        output_path (str): Path to save the plot (optional).

    Returns:
        None
    """
    distortions = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal Clusters')

    # Save the plot if output_path is provided
    if output_path:
        plt.savefig(output_path)
    plt.show()


# Align cluster labels with true labels using Hungarian algorithm
def align_labels(true_labels, cluster_labels):
    conf_matrix = confusion_matrix(true_labels, cluster_labels)
    row_indices, col_indices = linear_sum_assignment(-conf_matrix)
    mapping = {col: row for row, col in zip(row_indices, col_indices)}
    aligned_labels = np.array([mapping[label] for label in cluster_labels])
    return aligned_labels
