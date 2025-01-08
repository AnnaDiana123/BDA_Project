import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(file_path):
    """
    Load and preprocess data from a given file path.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Scaled feature data.
        pd.Series: Encoded numeric labels.
        dict: Mapping of original labels to numeric values.
    """
    data = pd.read_csv(file_path, header=None)

    # Scale the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.iloc[:, :-1])
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns[:-1])

    # Encode the labels numerically
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(data.iloc[:, -1])
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    return scaled_df, pd.Series(encoded_labels), label_mapping


def save_results(data, labels, output_path):
    results = data.copy()
    results['Cluster'] = labels
    results.to_csv(output_path, index=False)


def plot_clusters(data, labels, title, output_path=None):
    """
    Generate scatter plot for clustering results.

    Args:
        data (pd.DataFrame): Dataset with features.
        labels (pd.Series): Cluster labels for the data.
        title (str): Plot title.
        output_path (str): Path to save the plot (optional).
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=data.iloc[:, 0],
        y=data.iloc[:, 1],
        hue=labels,
        palette="viridis",
        legend="full"
    )
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(title="Cluster", loc='upper right')
    if output_path:
        plt.savefig(output_path)
    plt.show()


# Function to calculate metrics
def calculate_metrics(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    accuracy = accuracy_score(true_labels, predicted_labels)
    return {"precision": precision, "recall": recall, "accuracy": accuracy}


# Plot metrics for all steps
def plot_metrics(metrics_dict):
    steps = metrics_dict.keys()
    precision = [metrics["precision"] for metrics in metrics_dict.values()]
    recall = [metrics["recall"] for metrics in metrics_dict.values()]
    accuracy = [metrics["accuracy"] for metrics in metrics_dict.values()]

    x = range(len(steps))

    plt.figure(figsize=(10, 6))
    plt.plot(x, precision, label="Precision", marker='o')
    plt.plot(x, recall, label="Recall", marker='o')
    plt.plot(x, accuracy, label="Accuracy", marker='o')
    plt.xticks(x, steps)
    plt.ylim(0, 1)
    plt.xlabel("Steps")
    plt.ylabel("Score")
    plt.title("Metrics Across Steps")
    plt.legend()
    plt.grid()
    plt.show()
