from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
import pandas as pd
import numpy as np

def post_process_with_rf(train_data, train_labels, test_data, test_labels=None):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_data, train_labels)
    predicted_labels = rf.predict(test_data)

    # Ensure labels are of consistent type
    if test_labels is not None:
        test_labels = test_labels.astype(str)
        predicted_labels = [str(label) for label in predicted_labels]

    metrics = {}
    if test_labels is not None:
        metrics['precision'] = precision_score(
            test_labels, predicted_labels, average='weighted', zero_division=0
        )
        metrics['recall'] = recall_score(
            test_labels, predicted_labels, average='weighted', zero_division=0
        )
        metrics['accuracy'] = accuracy_score(test_labels, predicted_labels)

    return pd.Series(predicted_labels, index=test_data.index), metrics





def split_stable_and_misclassified(data, labels, centroids, threshold=0.5):
    """
    Split data into stable and misclassified points based on distance to centroids.

    Args:
        data (pd.DataFrame): Dataset with features.
        labels (pd.Series): Cluster labels for the data.
        centroids (np.ndarray): Array of cluster centroids.
        threshold (float): Distance threshold to identify misclassified points.

    Returns:
        tuple: (stable_data, stable_labels, misclassified_data)
    """
    # Compute distances to assigned centroids
    distances = np.linalg.norm(data.values - centroids[labels], axis=1)

    # Split stable and misclassified points
    stable_indices = distances <= threshold
    stable_data = data[stable_indices]
    stable_labels = labels[stable_indices]
    misclassified_data = data[~stable_indices]

    return stable_data, stable_labels, misclassified_data

