"""
K-Means Clustering
Unsupervised learning algorithm that partitions data into K clusters.
"""

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import matplotlib.pyplot as plt

def train_kmeans():
    # Generate synthetic data
    X, y_true = make_blobs(
        n_samples=300, 
        centers=4, 
        n_features=2,
        cluster_std=0.60, 
        random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and train model
    model = KMeans(
        n_clusters=4,
        init='k-means++',    # smart initialization
        n_init=10,           # number of times to run with different seeds
        max_iter=300,
        random_state=42
    )
    
    model.fit(X_scaled)
    labels = model.labels_
    
    # Evaluate
    silhouette_avg = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    inertia = model.inertia_
    
    print(f"Number of clusters: {model.n_clusters}")
    print(f"Silhouette Score: {silhouette_avg:.4f} (higher is better)")
    print(f"Davies-Bouldin Score: {davies_bouldin:.4f} (lower is better)")
    print(f"Inertia (within-cluster sum of squares): {inertia:.4f}")
    
    # Elbow method to find optimal K
    print("\n--- Elbow Method for Optimal K ---")
    inertias = []
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Plot Elbow curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(K_range, inertias, marker='o')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(True)
    
    ax2.plot(K_range, silhouette_scores, marker='o', color='orange')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score vs K')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('kmeans_optimization.png')
    print("Plot saved as 'kmeans_optimization.png'")
    
    # Visualize clusters
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
    centers = scaler.inverse_transform(model.cluster_centers_)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8, marker='X')
    plt.title('K-Means Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)
    plt.savefig('kmeans_clusters.png')
    print("Cluster visualization saved as 'kmeans_clusters.png'")
    
    return model, scaler

if __name__ == "__main__":
    model, scaler = train_kmeans()
