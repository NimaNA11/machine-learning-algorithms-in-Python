"""
Logistic Regression
Linear model for binary and multiclass classification using logistic function.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

def train_logistic_regression():
    # Load dataset
    data = load_breast_cancer()
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
    model = LogisticRegression(
        penalty='l2',           # 'l1', 'l2', 'elasticnet', or None
        C=1.0,                  # inverse of regularization strength
        solver='lbfgs',         # 'liblinear', 'saga', 'lbfgs', 'newton-cg'
        max_iter=1000,
        multi_class='auto',     # 'ovr' or 'multinomial'
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature coefficients
    print("\nTop 10 Most Important Features (by absolute coefficient):")
    coef_abs = np.abs(model.coef_[0])
    top_indices = np.argsort(coef_abs)[-10:][::-1]
    for idx in top_indices:
        print(f"{data.feature_names[idx]}: {model.coef_[0][idx]:.4f}")
    
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression')
    plt.legend()
    plt.grid(True)
    plt.savefig('logistic_regression_roc.png')
    print("\nROC curve saved as 'logistic_regression_roc.png'")
    
    return model, scaler

if __name__ == "__main__":
    model, scaler = train_logistic_regression()
