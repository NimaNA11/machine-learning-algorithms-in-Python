"""
Support Vector Machine (SVM)
Finds the hyperplane that best separates classes in high-dimensional space.
"""

from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

def train_svm():
    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features (important for SVM!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model = SVC(
        C=1.0,                # regularization parameter
        kernel='rbf',         # 'linear', 'poly', 'rbf', 'sigmoid'
        gamma='scale',        # kernel coefficient
        random_state=42,
        probability=True      # enable probability estimates
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Accuracy: {accuracy:.4f}")
    print(f"Number of support vectors: {model.n_support_}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Hyperparameter tuning with GridSearch
    print("\n--- Hyperparameter Tuning ---")
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }
    
    grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return model, scaler

if __name__ == "__main__":
    model, scaler = train_svm()
