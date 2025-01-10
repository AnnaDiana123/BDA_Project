from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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





def display_and_save_metrics_table(metrics_dict, output_path="metrics_table.png"):
    """
    Display a nicely formatted metrics table with colors and save it as an image.

    Args:
        metrics_dict (dict): Dictionary with keys as steps and values as dictionaries of metrics.
        output_path (str): Path to save the image of the metrics table.

    Returns:
        None
    """
    # Prepare data for the table
    steps = list(metrics_dict.keys())
    table_data = []

    for step, metrics in metrics_dict.items():
        row = {"Step": step}
        row.update(metrics)
        table_data.append(row)

    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame(table_data)

    # Reorder columns for better readability
    metrics_df = metrics_df[["Step", "precision", "recall", "accuracy"]]

    # Format the metrics for display using map
    for col in ["precision", "recall", "accuracy"]:
        metrics_df[col] = metrics_df[col].map(lambda x: f"{x:.2f}")

    # Use Seaborn to create a heatmap-style table
    plt.figure(figsize=(8, len(metrics_df) * 0.6))
    sns.heatmap(
        metrics_df.set_index("Step").astype(float),
        annot=True,
        fmt=".2f",
        cmap="Blues",  # Better contrast with black text
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        annot_kws={"size": 10, "color": "black"},  # Ensure black text
    )

    # Add title and save the image
    plt.title("Metrics Table", fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Metrics table saved to {output_path}")
