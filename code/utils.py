import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.iloc[:, :-1])
    return pd.DataFrame(scaled_data, columns=data.columns[:-1]), data.iloc[:, -1]

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
