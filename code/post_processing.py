import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def post_process_with_rf(train_data, train_labels, test_data):
    rf = RandomForestClassifier(
        bootstrap=True,
        ccp_alpha=0.0,
        class_weight=None,
        criterion='gini',
        max_depth=None,
        # max_features='auto',
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        n_estimators=200,
        n_jobs=None,
        oob_score=False,
        random_state=0,
        verbose=0,
        warm_start=False
    )

    rf.fit(train_data, train_labels)
    predicted_labels = rf.predict(test_data)

    return pd.Series(predicted_labels, index=test_data.index)


def split_stable_and_misclassified(data, labels, centroids, threshold=0.5):
    """
    Split data into stable and misclassified points based on the SC criterion.

    Args:
        data (pd.DataFrame): Dataset with features.
        labels (pd.Series): Cluster labels for the data.
        centroids (np.ndarray): Array of cluster centroids.
        threshold (float): SC threshold to identify misclassified points.

    Returns:
        tuple: (stable_data, stable_labels, misclassified_data, misclassified_labels)
    """
    # Ensure data is a NumPy array for computation
    data_np = data.values if isinstance(data, pd.DataFrame) else data

    # Compute distances from each point to all centroids
    distances = np.linalg.norm(data_np[:, np.newaxis] - centroids, axis=2)  # Shape: (n_points, n_centroids)

    # Compute the SC ratios
    min_distances = np.min(distances, axis=1, keepdims=True)
    sc_ratios = min_distances / distances  # Shape: (n_points, n_centroids)

    # Identify misclassified points based on SC
    misclassified_indices = np.any((sc_ratios > threshold) & (sc_ratios != 1.0), axis=1)

    # Split stable and misclassified points
    stable_indices = ~misclassified_indices

    # Convert back to pandas DataFrame/Series
    stable_data = data[stable_indices]
    stable_labels = pd.Series(labels[stable_indices], index=stable_data.index, name="Cluster")

    misclassified_data = data[misclassified_indices]
    misclassified_labels = pd.Series(-1, index=misclassified_data.index, name="Cluster")

    return stable_data, stable_labels, misclassified_data, misclassified_labels
