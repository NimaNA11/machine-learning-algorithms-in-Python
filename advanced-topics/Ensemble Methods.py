"""
Ensemble Methods
Combining multiple models to improve prediction performance.
"""

from sklearn.ensemble import (
    AdaBoostClassifier, 
    BaggingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def demonstrate_adaboost():
    """AdaBoost - Adaptive Boosting"""
    print("=" * 50)
    print("ADABOOST (Adaptive Boosting)")
    print("=" * 50)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create AdaBoost classifier
    adaboost = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),  # weak learner
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
    
    adaboost.fit(X_train_scaled, y_train)
    y_pred = adaboost.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Number of estimators: {len(adaboost.estimators_)}")
    
    # Feature importance
    print("\nTop 5 Important Features:")
    feature_imp = sorted(zip(data.feature_names, adaboost.feature_importances_), 
                        key=lambda x: x[1], reverse=True)
    for name, importance in feature_imp[:5]:
        print(f"{name}: {importance:.4f}")
    
    return adaboost, scaler

def demonstrate_bagging():
    """Bagging - Bootstrap Aggregating"""
    print("\n" + "=" * 50)
    print("BAGGING (Bootstrap Aggregating)")
    print("=" * 50)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create Bagging classifier
    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50,
        max_samples=0.8,      # fraction of samples for each estimator
        max_features=0.8,     # fraction of features for each estimator
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    bagging.fit(X_train_scaled, y_train)
    y_pred = bagging.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Number of estimators: {len(bagging.estimators_)}")
    
    # Cross-validation
    cv_scores = cross_val_score(bagging, X_train_scaled, y_train, cv=5)
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f}")
    
    return bagging, scaler

def demonstrate_extra_trees():
    """Extra Trees - Extremely Randomized Trees"""
    print("\n" + "=" * 50)
    print("EXTRA TREES (Extremely Randomized Trees)")
    print("=" * 50)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create Extra Trees classifier
    extra_trees = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    extra_trees.fit(X_train_scaled, y_train)
    y_pred = extra_trees.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nTop 5 Important Features:")
    feature_imp = sorted(zip(data.feature_names, extra_trees.feature_importances_), 
                        key=lambda x: x[1], reverse=True)
    for name, importance in feature_imp[:5]:
        print(f"{name}: {importance:.4f}")
    
    return extra_trees, scaler

def demonstrate_voting():
    """Voting Classifier - Hard and Soft Voting"""
    print("\n" + "=" * 50)
    print("VOTING CLASSIFIER")
    print("=" * 50)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define base estimators
    estimators = [
        ('lr', LogisticRegression(random_state=42, max_iter=1000)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('nb', GaussianNB()),
        ('svc', SVC(probability=True, random_state=42))
    ]
    
    # Hard Voting
    voting_hard = VotingClassifier(estimators=estimators, voting='hard')
    voting_hard.fit(X_train_scaled, y_train)
    y_pred_hard = voting_hard.predict(X_test_scaled)
    accuracy_hard = accuracy_score(y_test, y_pred_hard)
    
    # Soft Voting
    voting_soft = VotingClassifier(estimators=estimators, voting='soft')
    voting_soft.fit(X_train_scaled, y_train)
    y_pred_soft = voting_soft.predict(X_test_scaled)
    accuracy_soft = accuracy_score(y_test, y_pred_soft)
    
    print(f"Hard Voting Accuracy: {accuracy_hard:.4f}")
    print(f"Soft Voting Accuracy: {accuracy_soft:.4f}")
    
    # Compare with individual models
    print("\nIndividual Model Performance:")
    for name, model in estimators:
        model.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_scaled))
        print(f"{name}: {acc:.4f}")
    
    return voting_soft, scaler

def demonstrate_stacking():
    """Stacking Classifier"""
    print("\n" + "=" * 50)
    print("STACKING CLASSIFIER")
    print("=" * 50)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define base estimators
    base_estimators = [
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('nb', GaussianNB()),
        ('svc', SVC(probability=True, random_state=42))
    ]
    
    # Define meta-learner
    meta_learner = LogisticRegression(random_state=42, max_iter=1000)
    
    # Create Stacking classifier
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    stacking.fit(X_train_scaled, y_train)
    y_pred = stacking.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Stacking Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    # Compare with base estimators
    print("\nBase Estimator Performance:")
    for name, model in base_estimators:
        model.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_scaled))
        print(f"{name}: {acc:.4f}")
    
    return stacking, scaler

def compare_all_ensembles():
    """Compare all ensemble methods"""
    print("\n" + "=" * 50)
    print("COMPARING ALL ENSEMBLE METHODS")
    print("=" * 50)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Single Decision Tree (baseline)
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_scaled, y_train)
    results['Decision Tree'] = accuracy_score(y_test, dt.predict(X_test_scaled))
    
    # Test each ensemble method
    models = {
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'Bagging': BaggingClassifier(n_estimators=50, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
    }
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        results[name] = accuracy_score(y_test, model.predict(X_test_scaled))
    
    print("\nResults:")
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {acc:.4f}")

if __name__ == "__main__":
    adaboost, scaler1 = demonstrate_adaboost()
    bagging, scaler2 = demonstrate_bagging()
    extra_trees, scaler3 = demonstrate_extra_trees()
    voting, scaler4 = demonstrate_voting()
    stacking, scaler5 = demonstrate_stacking()
    compare_all_ensembles()
