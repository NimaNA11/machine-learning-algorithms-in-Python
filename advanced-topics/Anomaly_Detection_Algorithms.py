"""
Anomaly Detection Algorithms
Methods to identify outliers and unusual patterns in data.
"""

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

def generate_anomaly_data():
    """Generate dataset with anomalies"""
    # Normal data
    X_normal, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.5, random_state=42)
    
    # Add anomalies
    rng = np.random.RandomState(42)
    X_anomalies = rng.uniform(low=-4, high=4, size=(20, 2))
    
    X = np.vstack([X_normal, X_anomalies])
    y_true = np.array([1] * len(X_normal) + [-1] * len(X_anomalies))
    
    return X, y_true

def demonstrate_isolation_forest():
    """Isolation Forest - Tree-based anomaly detection"""
    print("=" * 50)
    print("ISOLATION FOREST")
    print("=" * 50)
    
    X, y_true = generate_anomaly_data()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and train model
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.1,      # expected proportion of outliers
        max_samples='auto',
        random_state=42
    )
    
    # Fit and predict
    y_pred = iso_forest.fit_predict(X_scaled)
    
    # Calculate metrics
    n_errors = (y_pred != y_true).sum()
    accuracy = 1 - n_errors / len(y_true)
    n_outliers_detected = (y_pred == -1).sum()
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Number of outliers detected: {n_outliers_detected}")
    print(f"Number of actual outliers: {(y_true == -1).sum()}")
    
    # Get anomaly scores
    scores = iso_forest.decision_function(X_scaled)
    print(f"Anomaly score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    return iso_forest, scaler, X, y_pred

def demonstrate_local_outlier_factor():
    """Local Outlier Factor - Density-based anomaly detection"""
    print("\n" + "=" * 50)
    print("LOCAL OUTLIER FACTOR (LOF)")
    print("=" * 50)
    
    X, y_true = generate_anomaly_data()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and train model
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.1,
        novelty=False  # True for prediction on new data
    )
    
    # Fit and predict
    y_pred = lof.fit_predict(X_scaled)
    
    # Calculate metrics
    n_errors = (y_pred != y_true).sum()
    accuracy = 1 - n_errors / len(y_true)
    n_outliers_detected = (y_pred == -1).sum()
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Number of outliers detected: {n_outliers_detected}")
    print(f"Number of actual outliers: {(y_true == -1).sum()}")
    
    # Get negative outlier factors
    scores = lof.negative_outlier_factor_
    print(f"LOF score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    return lof, scaler, X, y_pred

def demonstrate_one_class_svm():
    """One-Class SVM - Support vector-based anomaly detection"""
    print("\n" + "=" * 50)
    print("ONE-CLASS SVM")
    print("=" * 50)
    
    X, y_true = generate_anomaly_data()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and train model
    oc_svm = OneClassSVM(
        kernel='rbf',
        gamma='auto',
        nu=0.1  # upper bound on fraction of outliers
    )
    
    # Fit and predict
    y_pred = oc_svm.fit_predict(X_scaled)
    
    # Calculate metrics
    n_errors = (y_pred != y_true).sum()
    accuracy = 1 - n_errors / len(y_true)
    n_outliers_detected = (y_pred == -1).sum()
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Number of outliers detected: {n_outliers_detected}")
    print(f"Number of actual outliers: {(y_true == -1).sum()}")
    print(f"Number of support vectors: {len(oc_svm.support_vectors_)}")
    
    # Get decision scores
    scores = oc_svm.decision_function(X_scaled)
    print(f"Decision score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    return oc_svm, scaler, X, y_pred

def demonstrate_elliptic_envelope():
    """Elliptic Envelope - Gaussian distribution-based anomaly detection"""
    print("\n" + "=" * 50)
    print("ELLIPTIC ENVELOPE (Robust Covariance)")
    print("=" * 50)
    
    X, y_true = generate_anomaly_data()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and train model
    elliptic = EllipticEnvelope(
        contamination=0.1,
        random_state=42
    )
    
    # Fit and predict
    y_pred = elliptic.fit_predict(X_scaled)
    
    # Calculate metrics
    n_errors = (y_pred != y_true).sum()
    accuracy = 1 - n_errors / len(y_true)
    n_outliers_detected = (y_pred == -1).sum()
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Number of outliers detected: {n_outliers_detected}")
    print(f"Number of actual outliers: {(y_true == -1).sum()}")
    
    # Get Mahalanobis distances
    scores = elliptic.mahalanobis(X_scaled)
    print(f"Mahalanobis distance range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    return elliptic, scaler, X, y_pred

def visualize_all_methods():
    """Visualize all anomaly detection methods"""
    print("\n" + "=" * 50)
    print("VISUALIZING ALL ANOMALY DETECTION METHODS")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Isolation Forest
    iso_forest, scaler1, X1, y_pred1 = demonstrate_isolation_forest()
    axes[0, 0].scatter(X1[:, 0], X1[:, 1], c=y_pred1, cmap='coolwarm', s=50, alpha=0.6)
    axes[0, 0].set_title('Isolation Forest')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    
    # Local Outlier Factor
    lof, scaler2, X2, y_pred2 = demonstrate_local_outlier_factor()
    axes[0, 1].scatter(X2[:, 0], X2[:, 1], c=y_pred2, cmap='coolwarm', s=50, alpha=0.6)
    axes[0, 1].set_title('Local Outlier Factor')
    axes[0, 1].set_xlabel('Feature 1')
    axes[0, 1].set_ylabel('Feature 2')
    
    # One-Class SVM
    oc_svm, scaler3, X3, y_pred3 = demonstrate_one_class_svm()
    axes[1, 0].scatter(X3[:, 0], X3[:, 1], c=y_pred3, cmap='coolwarm', s=50, alpha=0.6)
    axes[1, 0].set_title('One-Class SVM')
    axes[1, 0].set_xlabel('Feature 1')
    axes[1, 0].set_ylabel('Feature 2')
    
    # Elliptic Envelope
    elliptic, scaler4, X4, y_pred4 = demonstrate_elliptic_envelope()
    axes[1, 1].scatter(X4[:, 0], X4[:, 1], c=y_pred4, cmap='coolwarm', s=50, alpha=0.6)
    axes[1, 1].set_title('Elliptic Envelope')
    axes[1, 1].set_xlabel('Feature 1')
    axes[1, 1].set_ylabel('Feature 2')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Normal'),
        Patch(facecolor='red', label='Anomaly')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('anomaly_detection_comparison.png')
    print("\nComparison plot saved as 'anomaly_detection_comparison.png'")

def compare_methods():
    """Compare all methods on the same dataset"""
    print("\n" + "=" * 50)
    print("COMPARING ALL METHODS")
    print("=" * 50)
    
    X, y_true = generate_anomaly_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    methods = {
        'Isolation Forest': IsolationForest(contamination=0.1, random_state=42),
        'LOF': LocalOutlierFactor(contamination=0.1),
        'One-Class SVM': OneClassSVM(nu=0.1),
        'Elliptic Envelope': EllipticEnvelope(contamination=0.1, random_state=42)
    }
    
    results = {}
    for name, model in methods.items():
        y_pred = model.fit_predict(X_scaled)
        accuracy = 1 - (y_pred != y_true).sum() / len(y_true)
        results[name] = accuracy
    
    print("\nAccuracy Comparison:")
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {acc:.4f}")

if __name__ == "__main__":
    visualize_all_methods()
    compare_methods()
