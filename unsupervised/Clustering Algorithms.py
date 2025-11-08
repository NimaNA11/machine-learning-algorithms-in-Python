"""
Clustering Algorithms
Unsupervised learning methods to group similar data points.
"""

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_kmeans():
    """K-Means Clustering"""
    print("=" * 50)
    print("K-MEANS CLUSTERING")
    print("=" * 50)
    
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
    print(f"Davies-Bouldin Score: {davies_bouldin_score(X_scaled, labels):.4f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, labels):.4f}")
    
    return kmeans, X, labels

def demonstrate_dbscan():
    """DBSCAN - Density-Based Spatial Clustering"""
    print("\n" + "=" * 50)
    print("DBSCAN (Density-Based Clustering)")
    print("=" * 50)
    
    X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    
    if n_clusters > 1:
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[dbscan.core_sample_indices_] = True
        
        # Only calculate scores for non-noise points
        mask = labels != -1
        if np.sum(mask) > 0 and len(set(labels[mask])) > 1:
            print(f"Silhouette Score: {silhouette_score(X_scaled[mask], labels[mask]):.4f}")
    
    return dbscan, X, labels

def demonstrate_hierarchical():
    """Agglomerative Hierarchical Clustering"""
    print("\n" + "=" * 50)
    print("HIERARCHICAL CLUSTERING")
    print("=" * 50)
    
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    hierarchical = AgglomerativeClustering(
        n_clusters=4,
        linkage='ward'  # 'ward', 'complete', 'average', 'single'
    )
    labels = hierarchical.fit_predict(X_scaled)
    
    print(f"Number of clusters: {hierarchical.n_clusters_}")
    print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
    
    return hierarchical, X, labels

def demonstrate_gaussian_mixture():
    """Gaussian Mixture Model"""
    print("\n" + "=" * 50)
    print("GAUSSIAN MIXTURE MODEL (GMM)")
    print("=" * 50)
    
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    gmm = GaussianMixture(
        n_components=4,
        covariance_type='full',  # 'full', 'tied', 'diag', 'spherical'
        random_state=42
    )
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    
    print(f"Number of components: {gmm.n_components}")
    print(f"Converged: {gmm.converged_}")
    print(f"BIC Score: {gmm.bic(X_scaled):.4f}")
    print(f"AIC Score: {gmm.aic(X_scaled):.4f}")
    print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
    
    return gmm, X, labels

def demonstrate_meanshift():
    """Mean Shift Clustering"""
    print("\n" + "=" * 50)
    print("MEAN SHIFT CLUSTERING")
    print("=" * 50)
    
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    meanshift = MeanShift(bandwidth=0.8)
    labels = meanshift.fit_predict(X_scaled)
    
    n_clusters = len(np.unique(labels))
    print(f"Number of clusters found: {n_clusters}")
    print(f"Number of cluster centers: {len(meanshift.cluster_centers_)}")
    print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
    
    return meanshift, X, labels

def demonstrate_spectral():
    """Spectral Clustering"""
    print("\n" + "=" * 50)
    print("SPECTRAL CLUSTERING")
    print("=" * 50)
    
    X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    spectral = SpectralClustering(
        n_clusters=2,
        affinity='nearest_neighbors',  # 'rbf', 'nearest_neighbors'
        random_state=42
    )
    labels = spectral.fit_predict(X_scaled)
    
    print(f"Number of clusters: {spectral.n_clusters}")
    print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
    
    return spectral, X, labels

def visualize_all_clustering():
    """Visualize all clustering algorithms"""
    print("\n" + "=" * 50)
    print("VISUALIZING ALL CLUSTERING ALGORITHMS")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # K-Means
    kmeans, X_km, labels_km = demonstrate_kmeans()
    axes[0, 0].scatter(X_km[:, 0], X_km[:, 1], c=labels_km, cmap='viridis', s=30)
    axes[0, 0].set_title('K-Means')
    
    # DBSCAN
    dbscan, X_db, labels_db = demonstrate_dbscan()
    axes[0, 1].scatter(X_db[:, 0], X_db[:, 1], c=labels_db, cmap='viridis', s=30)
    axes[0, 1].set_title('DBSCAN')
    
    # Hierarchical
    hierarchical, X_hc, labels_hc = demonstrate_hierarchical()
    axes[0, 2].scatter(X_hc[:, 0], X_hc[:, 1], c=labels_hc, cmap='viridis', s=30)
    axes[0, 2].set_title('Hierarchical')
    
    # GMM
    gmm, X_gmm, labels_gmm = demonstrate_gaussian_mixture()
    axes[1, 0].scatter(X_gmm[:, 0], X_gmm[:, 1], c=labels_gmm, cmap='viridis', s=30)
    axes[1, 0].set_title('Gaussian Mixture')
    
    # Mean Shift
    meanshift, X_ms, labels_ms = demonstrate_meanshift()
    axes[1, 1].scatter(X_ms[:, 0], X_ms[:, 1], c=labels_ms, cmap='viridis', s=30)
    axes[1, 1].set_title('Mean Shift')
    
    # Spectral
    spectral, X_sp, labels_sp = demonstrate_spectral()
    axes[1, 2].scatter(X_sp[:, 0], X_sp[:, 1], c=labels_sp, cmap='viridis', s=30)
    axes[1, 2].set_title('Spectral')
    
    plt.tight_layout()
    plt.savefig('clustering_comparison.png')
    print("\nClustering comparison saved as 'clustering_comparison.png'")

if __name__ == "__main__":
    visualize_all_clustering()
