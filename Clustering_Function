import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Helper Functions ---

def calculate_metrics(X, labels):
    """Calculates clustering evaluation metrics."""
    if len(np.unique(labels)) < 2 or len(X) < 2:
        return {'Silhouette_Score': 0, 'Davies_Bouldin_Score': float('inf'), 'Calinski_Harabasz_Score': 0}

    metrics = {}
    try:
        metrics['Silhouette_Score'] = silhouette_score(X, labels)
    except Exception:
        metrics['Silhouette_Score'] = None

    try:
        metrics['Davies_Bouldin_Score'] = davies_bouldin_score(X, labels)
    except Exception:
        metrics['Davies_Bouldin_Score'] = None

    try:
        metrics['Calinski_Harabasz_Score'] = calinski_harabasz_score(X, labels)
    except Exception:
        metrics['Calinski_Harabasz_Score'] = None

    return metrics

def load_spreadsheet():
    """Prompts the user for a spreadsheet file path and loads it into a pandas DataFrame."""
    while True:
        file_path = input("Please enter the path to your spreadsheet file (e.g., data.xlsx, data.csv): ")
        if not os.path.exists(file_path):
            print(f"Error: File not found at '{file_path}'. Please try again.")
            continue

        try:
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                print("Unsupported file format. Please provide a .csv, .xls, or .xlsx file.")
                continue

            print(f"Successfully loaded data from '{file_path}'. Shape: {df.shape}")
            return df

        except Exception as e:
            print(f"Error loading file: {e}. Please ensure the file is not corrupted and try again.")

def preprocess_data(df):
    """Handles data cleaning, missing values, and preprocessing."""
    print("\n--- Data Cleaning and Preprocessing ---")

    # Data Cleaning and Handling Missing Values
    df_processed = df.copy()
    numerical_cols = df_processed.select_dtypes(include=np.number).columns
    categorical_cols = df_processed.select_dtypes(include='object').columns

    # Imputation for numerical data
    imputer_num = SimpleImputer(strategy='median')
    if not numerical_cols.empty:
        df_processed[numerical_cols] = imputer_num.fit_transform(df_processed[numerical_cols])
    else:
        print("Note: No numerical columns found. Skipping numerical imputation.")


    # Handle missing values and mixed types in categorical data
    if not categorical_cols.empty:
        # Fill missing values in categorical columns first
        df_processed[categorical_cols] = df_processed[categorical_cols].fillna('missing')
        # Convert all categorical columns to string type to handle mixed types
        for col in categorical_cols:
             df_processed[col] = df_processed[col].astype(str)
        print("Categorical missing values handled and types converted to string.")
    else:
         print("Note: No categorical columns found.")


    df_processed.drop_duplicates(inplace=True)
    print("Duplicates removed.")

    # Data Preprocessing (Encoding and Normalization)
    if not categorical_cols.empty:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cols = encoder.fit_transform(df_processed[categorical_cols])
        encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols), index=df_processed.index)
        df_processed = pd.concat([df_processed.drop(columns=categorical_cols), encoded_df], axis=1)
        print("Categorical data encoded.")
    else:
        print("Note: No categorical columns to encode.")

    # Interactive Normalization selection
    while True:
        normalize_input = input("Should the data be normalized using StandardScaler (recommended)? (yes/no): ").lower()
        if normalize_input in ['yes', 'y']:
            scaler = StandardScaler()
            numerical_cols_processed = df_processed.select_dtypes(include=np.number).columns
            if not numerical_cols_processed.empty:
                df_processed[numerical_cols_processed] = scaler.fit_transform(df_processed[numerical_cols_processed])
                print("Data normalized.")
            else:
                print("Data has no numerical features after encoding. Skipping normalization.")
            break
        elif normalize_input in ['no', 'n']:
            print("Skipping normalization.")
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")


    X = df_processed.dropna().copy()
    if X.empty:
        raise ValueError("DataFrame is empty after cleaning and preprocessing.")

    return X

def get_clustering_parameters(available_methods):
    """Interactively asks the user for clustering methods and their parameters."""
    print("\n--- Clustering Setup ---")
    print(f"Available clustering methods: {', '.join(available_methods)}")

    while True:
        methods_input = input("Enter methods to compare (comma-separated, or 'all'): ").lower()
        if methods_input == 'all':
            clustering_methods = available_methods
            break

        clustering_methods = [method.strip() for method in methods_input.split(',')]
        valid_methods = True
        for method in clustering_methods:
            if method not in available_methods:
                print(f"'{method}' is not valid. Choose from {available_methods} or 'all'.")
                valid_methods = False
                break
        if valid_methods:
            break

    user_params = {}

    if 'kmeans' in clustering_methods:
        while True:
            try:
                k = int(input(f"Enter the number of clusters (K) for K-Means: "))
                user_params['n_clusters_kmeans'] = k
                break
            except ValueError:
                print("Invalid input. Please enter an integer.")

    if 'agglomerative' in clustering_methods:
        while True:
            try:
                n_agg = int(input(f"Enter the number of clusters for Agglomerative Clustering: "))
                user_params['n_clusters_agg'] = n_agg
                break
            except ValueError:
                print("Invalid input. Please enter an integer.")

    if 'dbscan' in clustering_methods:
        while True:
            try:
                eps = float(input(f"Enter eps for DBSCAN (e.g., 0.5): "))
                min_samples = int(input(f"Enter min_samples for DBSCAN (e.g., 5): "))
                user_params['dbscan_eps'] = eps
                user_params['dbscan_min_samples'] = min_samples
                break
            except ValueError:
                print("Invalid input. Please enter valid numbers.")

    return clustering_methods, user_params

def perform_clustering(X, methods, params):
    """Executes clustering algorithms and stores results."""

    print("\n--- Running Clustering Algorithms ---")
    results = {}
    metrics_summary = {}

    # K-Means
    if 'kmeans' in methods:
        print(f"Running K-Means (K={params['n_clusters_kmeans']})...")
        try:
            kmeans = KMeans(n_clusters=params['n_clusters_kmeans'], random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(X)
            metrics = calculate_metrics(X, kmeans_labels)
            results['KMeans'] = {
                'labels': kmeans_labels,
                'metrics': metrics,
                'model': kmeans # Store the model for potential use (e.g., cluster centers)
            }
            metrics_summary['KMeans'] = metrics
        except Exception as e:
            print(f"KMeans failed: {e}")

    # DBSCAN
    if 'dbscan' in methods:
        print(f"Running DBSCAN (eps={params['dbscan_eps']}, min_samples={params['dbscan_min_samples']})...")
        try:
            dbscan = DBSCAN(eps=params['dbscan_eps'], min_samples=params['dbscan_min_samples'])
            dbscan_labels = dbscan.fit_predict(X)

            # Evaluate only if valid clusters (more than one unique label excluding noise -1)
            if len(np.unique(dbscan_labels[dbscan_labels != -1])) > 1:
                metrics = calculate_metrics(X[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1])
                results['DBSCAN'] = {'labels': dbscan_labels, 'metrics': metrics, 'model': dbscan}
                metrics_summary['DBSCAN'] = metrics
            else:
                print("DBSCAN resulted in less than 2 valid clusters.")
        except Exception as e:
            print(f"DBSCAN failed: {e}")

    # Agglomerative Clustering
    if 'agglomerative' in methods:
        print(f"Running Agglomerative Clustering (n_clusters={params['n_clusters_agg']})...")
        try:
            agglomerative = AgglomerativeClustering(n_clusters=params['n_clusters_agg'])
            agg_labels = agglomerative.fit_predict(X)
            metrics = calculate_metrics(X, agg_labels)
            results['Agglomerative'] = {'labels': agg_labels, 'metrics': metrics, 'model': agglomerative}
            metrics_summary['Agglomerative'] = metrics
        except Exception as e:
            print(f"Agglomerative Clustering failed: {e}")

    return results, metrics_summary

def visualize_results(X, results):
    """
    Visualizes clustering results using PCA for dimensionality reduction.
    """
    print("\n--- Generating Cluster Visualizations (using PCA) ---")

    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        print(f"Data reduced to 2 components using PCA for visualization.")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.2f}")
    else:
        X_pca = X.to_numpy()
        print("Data already in 2 dimensions or less. Skipping PCA.")

    X_vis = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

    for method_name, result in results.items():
        labels = result['labels']

        # Skip visualization if clustering resulted in only one cluster or all noise (-1)
        if len(np.unique(labels)) <= 1 or (method_name == 'DBSCAN' and len(np.unique(labels[labels != -1])) == 0):
            print(f"Skipping visualization for {method_name}: Insufficient clusters.")
            continue

        plt.figure(figsize=(8, 6))

        # Create a DataFrame for plotting, handling potential index mismatches after preprocessing
        plot_data = X_vis.copy()
        plot_data['Cluster'] = labels[:len(plot_data)] # Ensure labels match the length of X_vis

        # Convert cluster labels to strings for better visualization handling (especially DBSCAN noise)
        plot_data['Cluster'] = plot_data['Cluster'].astype(str)

        sns.scatterplot(data=plot_data, x='PC1', y='PC2', hue='Cluster', palette='viridis', legend='full')

        # Add centroids for KMeans
        if method_name == 'KMeans' and 'model' in result and hasattr(result['model'], 'cluster_centers_'):
            centers = result['model'].cluster_centers_
            if X.shape[1] > 2:
                # Transform centroids using the same PCA model
                centers_pca = pca.transform(centers)
                plt.scatter(centers_pca[:, 0], centers_pca[:, 1], marker='X', s=200, color='red', edgecolor='black', label='Centroids')
            else:
                plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, color='red', edgecolor='black', label='Centroids')

        plt.title(f'Clustering Results: {method_name} (Visualized via PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Cluster')
        plt.grid(True)
        plt.show()

def present_results(results, metrics_summary):
    """Presents the comparison, clustering metrics, and suggests the best method."""

    print("\n--- Detailed Clustering Results and Metrics ---")

    # 1. Present Metrics Summary Table
    if metrics_summary:
        metrics_df = pd.DataFrame(metrics_summary).T
        print(metrics_df)

        # 2. Suggest the best model
        if 'Silhouette_Score' in metrics_df.columns and not metrics_df['Silhouette_Score'].isnull().all():
            best_model_name = metrics_df['Silhouette_Score'].idxmax()
            best_score = metrics_df.loc[best_model_name, 'Silhouette_Score']

            suggestion = f"\n**The suggested clustering method is:** **{best_model_name}**."
            suggestion += f"\nReasoning: This method yielded the highest Silhouette Score of {best_score:.4f}, indicating strong cluster separation and cohesion."
        else:
            suggestion = "\n**Suggestion:** Cannot determine the best method using Silhouette Score. All algorithms failed to produce valid clusters for evaluation."

        print(suggestion)
    else:
        print("No valid clustering results were obtained.")
        suggestion = "No valid clustering results."

    # 3. Present Clustering Results (Assignments)
    print("\n--- Cluster Assignments Sample ---")
    if results:
        # We can show a snippet of the data with cluster assignments for the first successful result
        first_result_name = next(iter(results.keys()), None)
        if first_result_name:
            labels = results[first_result_name]['labels']
            print(f"Sample data points showing assignments from {first_result_name}:")

            # Create a sample DataFrame combining original indices and labels
            # Ensure we only sample from valid indices if any rows were dropped during preprocessing
            sample_df = pd.DataFrame({'Cluster_ID': labels})
            print(sample_df['Cluster_ID'].value_counts().sort_index())
            print("\n(Displayed counts per cluster. Detailed assignments available in the output.)")

    return suggestion

# --- Main Interactive Function ---

def interactive_clustering_session():
    """Manages the full interactive clustering session."""

    print("Welcome to the AI Clustering Assistant.")

    # 1. Load the spreadsheet from the user
    df_raw = load_spreadsheet()

    if df_raw is None:
        return

    # 2. Preprocess the data
    try:
        X_processed = preprocess_data(df_raw)
    except ValueError as e:
        print(f"Error during preprocessing: {e}")
        return

    # 3. Interactive clustering loop
    available_methods = ['kmeans', 'dbscan', 'agglomerative']

    while True:
        print("\n--- Starting New Clustering Analysis ---")

        # Get methods and parameters from the user
        methods, params = get_clustering_parameters(available_methods)

        # Perform clustering
        results, metrics_summary = perform_clustering(X_processed, methods, params)

        # Present results (metrics, suggestion, and graphical visualization)
        suggestion = present_results(results, metrics_summary)

        # Visualize the results
        if results:
            visualize_results(X_processed, results)

        # 4. Ask the user if they want to re-run or change parameters
        print("\n--- Analysis Complete ---")
        rerun_choice = input("Do you want to (r)erun with different methods/parameters, (s)tart a new analysis with a different file, or (q)uit? (r/s/q): ").lower()

        if rerun_choice == 'q':
            print("Thank you for using the AI Clustering Assistant. Goodbye!")
            break
        elif rerun_choice == 's':
            print("\nStarting a new session...")
            # If 's' selected, we return from this function, allowing a fresh start if called in a loop.
            return
        elif rerun_choice == 'r':
            print("Rerunning analysis with different settings...")
            continue # Continue the current while loop to prompt for new settings
        else:
            print("Invalid input. Exiting session.")
            break

# Example of how the interactive session would start:
if __name__ == '__main__':
    interactive_clustering_session()
