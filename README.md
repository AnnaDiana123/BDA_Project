### Project Overview

This project involves clustering and post-processing of datasets to analyze the quality of clustering results and refine them using techniques like the Split Criterion (SC), silhouette scores, and supervised learning (Random Forest). The project is implemented using Python, with several modules and Jupyter notebooks for different datasets.

---

### **Project Goals**
1. **Clustering**:
   - Use the K-Means algorithm to cluster datasets.
   - Determine the optimal number of clusters using the elbow method.
   - Align clustering results with true labels when available.

2. **Post-Processing**:
   - Identify misclassified points using SC and silhouette scores.
   - Reclassify misclassified points using a supervised learning model (Random Forest).
   - Evaluate clustering performance before and after post-processing.

3. **Documentation**:
   - Provide visualizations for clustering results and metrics.
   - Compare metrics across thresholds and approaches.
   - Save results and metrics for further analysis.

---

### **Current Implementation**

#### 1. **Modules**
- **`clustering.py`**:
  - Implements core clustering functionality, including:
    - **K-Means Clustering**: `run_kmeans`.
    - **Elbow Method**: Determines optimal clusters based on distortion scores.
    - **Label Alignment**: Aligns K-Means labels with true labels using the Hungarian algorithm.

- **`post_processing.py`**:
  - Handles the refinement of clustering results:
    - **Random Forest Reclassification**: Reclassifies misclassified points using supervised learning.
    - **Split Criterion (SC)**: Identifies stable and misclassified points based on SC ratios.
    - **Silhouette Score Analysis**: Detects misclassified points using silhouette scores.

- **`utils.py`**:
  - Utilities for data loading, metrics computation, and visualization:
    - **Data Preprocessing**: Scales data and encodes labels (when available).
    - **Metrics Calculation**: Computes precision, recall, and accuracy.
    - **Cluster Visualization**: Generates scatter plots for clustering results.
    - **Metrics Table**: Displays and saves metrics as a heatmap-style table.

---

#### 2. **Jupyter Notebooks**
- **`experiment_iris.ipynb`**:
  - Implements clustering and post-processing for the Iris dataset.
  - Includes misclassification detection using SC. 
  - Visualizes clustering results and compares metrics before and after post-processing.

- **`experiment_dataset2.ipynb`**:
  - Processes a second dataset without labels.
  - Focuses on identifying optimal clusters and detecting misclassified points using SC and without silhouette scores. 

- **`experiment_iris_siluette.ipynb`**:
  - Focuses on comparing misclassification detection approaches using silhouette scores.

- **`experiment_iris_thresholds.ipynb`**:
  - Evaluates the impact of different threshold values for SC and silhouette scores on clustering quality.

---

### **Pipeline Overview**

1. **Data Loading**:
   - Labeled and unlabeled datasets are loaded using `load_data` or `load_data_dataset2` (for cleaning and scaling data).

2. **Clustering**:
   - Use `run_kmeans` to perform clustering.
   - Visualize the elbow method to determine the optimal number of clusters.
   - For labeled datasets, align cluster labels using `align_labels_and_centroids`.

3. **Post-Processing**:
   - Detect misclassified points using SC (`split_stable_and_misclassified`) or silhouette scores (`detect_misclassified_with_silhouette`).
   - Reclassify misclassified points using `post_process_with_rf`.

4. **Evaluation**:
   - Compute metrics (precision, recall, accuracy) using `calculate_metrics`.
   - Compare metrics across different thresholds for SC and silhouette scores.

5. **Visualization**:
   - Scatter plots for clustering results.
   - Heatmap-style tables for metrics.

---

### **Key Functions and Their Purpose**

#### **Clustering**
- `run_kmeans(data, n_clusters)`:
  Performs K-Means clustering and returns cluster labels.

- `elbow_method(data, max_clusters, output_path)`:
  Plots the elbow method to determine the optimal number of clusters.

- `align_labels_and_centroids(true_labels, cluster_labels, centroids)`:
  Aligns cluster labels with true labels using the Hungarian algorithm.

#### **Post-Processing**
- `split_stable_and_misclassified(data, labels, centroids, threshold)`:
  Splits data into stable and misclassified points based on SC.

- `detect_misclassified_with_silhouette(data, labels, threshold)`:
  Identifies misclassified points using silhouette scores.

- `post_process_with_rf(train_data, train_labels, test_data)`:
  Reclassifies misclassified points using a Random Forest model.

#### **Utilities**
- `load_data(file_path, has_labels)`:
  Loads and preprocesses datasets with or without labels.

- `plot_clusters(data, labels, title, output_path)`:
  Creates scatter plots for clustering results.

- `calculate_metrics(true_labels, predicted_labels)`:
  Calculates precision, recall, and accuracy.

- `display_and_save_metrics_table(metrics_dict, output_path)`:
  Displays and saves a metrics table as a heatmap.
