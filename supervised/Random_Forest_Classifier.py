"""
Random Forest Classifier
An ensemble method that builds multiple decision trees and merges them together.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def train_random_forest():
    # Load dataset
    data = load_wine()
    X, y = data.data, data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = RandomForestClassifier(
        n_estimators=100,      # number of trees
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',   # features to consider at each split
        bootstrap=True,        # use bootstrap samples
        random_state=42,
        n_jobs=-1              # use all CPU cores
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    # Feature importance
    print("\nTop 5 Most Important Features:")
    feature_imp = sorted(zip(data.feature_names, model.feature_importances_), 
                        key=lambda x: x[1], reverse=True)
    for name, importance in feature_imp[:5]:
        print(f"{name}: {importance:.4f}")
    
    return model

if __name__ == "__main__":
    model = train_random_forest()
