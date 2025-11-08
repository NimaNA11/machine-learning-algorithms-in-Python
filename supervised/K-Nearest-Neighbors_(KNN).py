"""
K-Nearest Neighbors (KNN)
Classifies based on the majority class among k nearest neighbors.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

def train_knn():
    # Load dataset
    data = load_digits()
    X, y = data.data, data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model = KNeighborsClassifier(
        n_neighbors=5,           # number of neighbors
        weights='uniform',       # 'uniform' or 'distance'
        algorithm='auto',        # 'ball_tree', 'kd_tree', 'brute', 'auto'
        metric='minkowski',      # distance metric
        p=2,                     # p=2 for Euclidean distance
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Find optimal K
    print("\n--- Finding Optimal K ---")
    k_range = range(1, 31)
    scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        scores.append(knn.score(X_test_scaled, y_test))
    
    optimal_k = k_range[np.argmax(scores)]
    print(f"Optimal K: {optimal_k} with accuracy: {max(scores):.4f}")
    
    # Plot K vs Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, scores, marker='o')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.title('KNN: K Value vs Accuracy')
    plt.grid(True)
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    plt.legend()
    plt.savefig('knn_k_optimization.png')
    print("\nPlot saved as 'knn_k_optimization.png'")
    
    return model, scaler

if __name__ == "__main__":
    model, scaler = train_knn()
