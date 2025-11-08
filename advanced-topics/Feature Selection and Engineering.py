"""
Feature Selection and Engineering
Methods to select the most important features and create new ones.
"""

from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_variance_threshold():
    """Remove low variance features"""
    print("=" * 50)
    print("VARIANCE THRESHOLD")
    print("=" * 50)
    
    # Generate data with low variance features
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                               n_redundant=5, random_state=42)
    
    # Add constant feature
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    
    print(f"Original number of features: {X.shape[1]}")
    
    # Remove low variance features
    selector = VarianceThreshold(threshold=0.1)
    X_selected = selector.fit_transform(X)
    
    print(f"Number of features after variance threshold: {X_selected.shape[1]}")
    print(f"Removed {X.shape[1] - X_selected.shape[1]} features")
    
    # Show variances
    variances = np.var(X, axis=0)
    print(f"\nFeature variances (first 5): {variances[:5]}")
    
    return selector

def demonstrate_univariate_selection():
    """Univariate Feature Selection"""
    print("\n" + "=" * 50)
    print("UNIVARIATE FEATURE SELECTION")
    print("=" * 50)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    print(f"Original number of features: {X.shape[1]}")
    
    # Select k best features using ANOVA F-value
    selector_f = SelectKBest(f_classif, k=10)
    X_selected_f = selector_f.fit_transform(X, y)
    
    print(f"\nANOVA F-test:")
    print(f"Features selected: {X_selected_f.shape[1]}")
    
    # Get selected feature indices and scores
    scores = selector_f.scores_
    selected_indices = selector_f.get_support(indices=True)
    
    print("\nTop 10 features by F-score:")
    feature_scores = sorted(zip(data.feature_names, scores), key=lambda x: x[1], reverse=True)
    for name, score in feature_scores[:10]:
        print(f"{name}: {score:.2f}")
    
    # Mutual Information
    print("\n" + "-" * 50)
    print("Mutual Information:")
    selector_mi = SelectKBest(mutual_info_classif, k=10)
    X_selected_mi = selector_mi.fit_transform(X, y)
    
    mi_scores = selector_mi.scores_
    print("\nTop 10 features by Mutual Information:")
    mi_feature_scores = sorted(zip(data.feature_names, mi_scores), key=lambda x: x[1], reverse=True)
    for name, score in mi_feature_scores[:10]:
        print(f"{name}: {score:.4f}")
    
    return selector_f, selector_mi

def demonstrate_rfe():
    """Recursive Feature Elimination"""
    print("\n" + "=" * 50)
    print("RECURSIVE FEATURE ELIMINATION (RFE)")
    print("=" * 50)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # RFE with Logistic Regression
    estimator = LogisticRegression(random_state=42, max_iter=1000)
    rfe = RFE(estimator, n_features_to_select=10, step=1)
    
    rfe.fit(X_train_scaled, y_train)
    
    # Transform data
    X_train_rfe = rfe.transform(X_train_scaled)
    X_test_rfe = rfe.transform(X_test_scaled)
    
    # Train model on selected features
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_rfe, y_train)
    
    # Compare accuracies
    acc_original = LogisticRegression(random_state=42, max_iter=1000).fit(
        X_train_scaled, y_train).score(X_test_scaled, y_test)
    acc_rfe = model.score(X_test_rfe, y_test)
    
    print(f"Accuracy with all features: {acc_original:.4f}")
    print(f"Accuracy with RFE features: {acc_rfe:.4f}")
    
    # Show selected features
    selected_features = [data.feature_names[i] for i in range(len(data.feature_names)) 
                        if rfe.support_[i]]
    print(f"\nSelected features: {selected_features}")
    
    # Show ranking
    print("\nFeature rankings (1 = selected):")
    rankings = sorted(zip(data.feature_names, rfe.ranking_), key=lambda x: x[1])
    for name, rank in rankings[:10]:
        print(f"{name}: {rank}")
    
    return rfe

def demonstrate_model_based_selection():
    """Model-based Feature Selection"""
    print("\n" + "=" * 50)
    print("MODEL-BASED FEATURE SELECTION")
    print("=" * 50)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest based selection
    print("\nRandom Forest based selection:")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    selector_rf = SelectFromModel(rf, prefit=True, threshold='median')
    X_train_rf = selector_rf.transform(X_train_scaled)
    X_test_rf = selector_rf.transform(X_test_scaled)
    
    print(f"Features selected: {X_train_rf.shape[1]} out of {X.shape[1]}")
    
    # Compare accuracies
    acc_original = rf.score(X_test_scaled, y_test)
    rf_new = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_new.fit(X_train_rf, y_train)
    acc_selected = rf_new.score(X_test_rf, y_test)
    
    print(f"Accuracy with all features: {acc_original:.4f}")
    print(f"Accuracy with selected features: {acc_selected:.4f}")
    
    # Show important features
    selected_features = [data.feature_names[i] for i in range(len(data.feature_names)) 
                        if selector_rf.get_support()[i]]
    print(f"\nSelected features: {selected_features}")
    
    # Lasso based selection
    print("\n" + "-" * 50)
    print("Lasso (L1) based selection:")
    lasso = Lasso(alpha=0.01, random_state=42, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    
    selector_lasso = SelectFromModel(lasso, prefit=True)
    X_train_lasso = selector_lasso.transform(X_train_scaled)
    
    print(f"Features selected: {X_train_lasso.shape[1]} out of {X.shape[1]}")
    
    # Show features with non-zero coefficients
    non_zero_features = [data.feature_names[i] for i in range(len(data.feature_names)) 
                        if selector_lasso.get_support()[i]]
    print(f"Features with non-zero coefficients: {non_zero_features}")
    
    return selector_rf, selector_lasso

def demonstrate_polynomial_features():
    """Polynomial Feature Engineering"""
    print("\n" + "=" * 50)
    print("POLYNOMIAL FEATURES")
    print("=" * 50)
    
    # Simple dataset
    X = np.array([[1, 2], [3, 4], [5, 6]])
    
    print("Original features:")
    print(X)
    print(f"Shape: {X.shape}")
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    print("\nPolynomial features (degree=2):")
    print(X_poly)
    print(f"Shape: {X_poly.shape}")
    print(f"\nFeature names: {poly.get_feature_names_out(['x1', 'x2'])}")
    
    # Example with real data
    data = load_breast_cancer()
    X, y = data.data[:, :2], data.target  # Use only 2 features for simplicity
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Without polynomial features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    acc_original = lr.score(X_test_scaled, y_test)
    
    # With polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    X_train_poly_scaled = scaler.fit_transform(X_train_poly)
    X_test_poly_scaled = scaler.transform(X_test_poly)
    
    lr_poly = LogisticRegression(random_state=42, max_iter=1000)
    lr_poly.fit(X_train_poly_scaled, y_train)
    acc_poly = lr_poly.score(X_test_poly_scaled, y_test)
    
    print(f"\nAccuracy with original features: {acc_original:.4f}")
    print(f"Accuracy with polynomial features: {acc_poly:.4f}")
    
    return poly

def visualize_feature_importance():
    """Visualize feature importance from different methods"""
    print("\n" + "=" * 50)
    print("VISUALIZING FEATURE IMPORTANCE")
    print("=" * 50)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Plot top 10 features
    importances = rf.feature_importances_
    indices = np.argsort(importances)[-10:]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [data.feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved as 'feature_importance.png'")

if __name__ == "__main__":
    variance_selector = demonstrate_variance_threshold()
    univariate_selectors = demonstrate_univariate_selection()
    rfe_selector = demonstrate_rfe()
    model_selectors = demonstrate_model_based_selection()
    poly = demonstrate_polynomial_features()
    visualize_feature_importance()
