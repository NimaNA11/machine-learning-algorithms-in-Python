"""
Principal Component Analysis (PCA) and Dimensionality Reduction
Techniques to reduce the number of features while preserving information.
"""

from sklearn.decomposition import PCA, TruncatedSVD, NMF, FastICA
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_pca():
    """Principal Component Analysis"""
    print("=" * 50)
    print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("=" * 50)
    
    # Load dataset
    data = load_digits()
    X, y = data.data, data.target
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=0.95, random_state=42)  # keep 95% variance
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Original number of features: {X.shape[1]}")
    print(f"Reduced number of features: {X_pca.shape[1]}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"Number of components: {pca.n_components_}")
    
    # Plot explained variance
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    plt.xlabel('Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Variance per Component')
    plt.tight_layout()
    plt.savefig('pca_variance.png')
    print("Variance plots saved as 'pca_variance.png'")
    
    # Compare classification with and without PCA
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
    # Without PCA
    clf_original = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_original.fit(X_train, y_train)
    acc_original = accuracy_score(y_test, clf_original.predict(X_test))
    
    # With PCA
    clf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_pca.fit(X_train_pca, y_train)
    acc_pca = accuracy_score(y_test, clf_pca.predict(X_test_pca))
    
    print(f"\nAccuracy without PCA: {acc_original:.4f}")
    print(f"Accuracy with PCA: {acc_pca:.4f}")
    print(f"Feature reduction: {(1 - X_pca.shape[1]/X.shape[1])*100:.1f}%")
    
    return pca, scaler

def demonstrate_tsne():
    """t-Distributed Stochastic Neighbor Embedding"""
    print("\n" + "=" * 50)
    print("t-SNE (t-Distributed Stochastic Neighbor Embedding)")
    print("=" * 50)
    
    data = load_digits()
    X, y = data.data, data.target
    
    # First reduce with PCA (recommended for t-SNE)
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Apply t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter=1000,
        random_state=42
    )
    X_tsne = tsne.fit_transform(X_pca)
    
    print(f"Original dimensions: {X.shape[1]}")
    print(f"t-SNE dimensions: {X_tsne.shape[1]}")
    print(f"KL divergence: {tsne.kl_divergence_:.4f}")
    
    # Visualize
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=20, alpha=0.7)
    plt.colorbar(scatter, label='Digit Class')
    plt.title('t-SNE Visualization of Digits Dataset')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig('tsne_visualization.png')
    print("t-SNE visualization saved as 'tsne_visualization.png'")
    
    return tsne

def demonstrate_svd():
    """Truncated Singular Value Decomposition"""
    print("\n" + "=" * 50)
    print("TRUNCATED SVD")
    print("=" * 50)
    
    data = load_digits()
    X, y = data.data, data.target
    
    # Apply Truncated SVD
    svd = TruncatedSVD(n_components=30, random_state=42)
    X_svd = svd.fit_transform(X)
    
    print(f"Original dimensions: {X.shape[1]}")
    print(f"Reduced dimensions: {X_svd.shape[1]}")
    print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
    
    return svd

def demonstrate_ica():
    """Independent Component Analysis"""
    print("\n" + "=" * 50)
    print("INDEPENDENT COMPONENT ANALYSIS (ICA)")
    print("=" * 50)
    
    data = load_digits()
    X, y = data.data, data.target
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply ICA
    ica = FastICA(n_components=30, random_state=42, max_iter=1000)
    X_ica = ica.fit_transform(X_scaled)
    
    print(f"Original dimensions: {X.shape[1]}")
    print(f"ICA components: {X_ica.shape[1]}")
    print(f"Number of iterations: {ica.n_iter_}")
    
    return ica

def demonstrate_nmf():
    """Non-negative Matrix Factorization"""
    print("\n" + "=" * 50)
    print("NON-NEGATIVE MATRIX FACTORIZATION (NMF)")
    print("=" * 50)
    
    data = load_digits()
    X, y = data.data, data.target
    
    # NMF requires non-negative data
    nmf = NMF(n_components=30, random_state=42, max_iter=1000)
    X_nmf = nmf.fit_transform(X)
    
    print(f"Original dimensions: {X.shape[1]}")
    print(f"NMF components: {X_nmf.shape[1]}")
    print(f"Reconstruction error: {nmf.reconstruction_err_:.4f}")
    
    return nmf

if __name__ == "__main__":
    pca, scaler = demonstrate_pca()
    tsne = demonstrate_tsne()
    svd = demonstrate_svd()
    ica = demonstrate_ica()
    nmf = demonstrate_nmf()
