from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

