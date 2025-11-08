"""
Gradient Boosting Algorithms
Ensemble methods that build models sequentially to correct previous errors.
Includes: Gradient Boosting, XGBoost, LightGBM, and CatBoost
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Note: Install these with: pip install xgboost lightgbm catboost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not installed. Install with: pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not installed. Install with: pip install catboost")

def train_sklearn_gradient_boosting():
    """Scikit-learn's Gradient Boosting (always available)"""
    print("=" * 50)
    print("SKLEARN GRADIENT BOOSTING")
    print("=" * 50)
    
    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    return model

def train_xgboost():
    """XGBoost - Extreme Gradient Boosting"""
    if not XGBOOST_AVAILABLE:
        return None
        
    print("\n" + "=" * 50)
    print("XGBOOST")
    print("=" * 50)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    # Feature importance
    print("\nTop 5 Important Features:")
    feature_imp = sorted(zip(data.feature_names, model.feature_importances_), 
                        key=lambda x: x[1], reverse=True)
    for name, importance in feature_imp[:5]:
        print(f"{name}: {importance:.4f}")
    
    return model

def train_lightgbm():
    """LightGBM - Light Gradient Boosting Machine"""
    if not LIGHTGBM_AVAILABLE:
        return None
        
    print("\n" + "=" * 50)
    print("LIGHTGBM")
    print("=" * 50)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    return model

def train_catboost():
    """CatBoost - Categorical Boosting"""
    if not CATBOOST_AVAILABLE:
        return None
        
    print("\n" + "=" * 50)
    print("CATBOOST")
    print("=" * 50)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=3,
        random_state=42,
        verbose=False
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    return model

if __name__ == "__main__":
    sklearn_model = train_sklearn_gradient_boosting()
    xgb_model = train_xgboost()
    lgb_model = train_lightgbm()
    cat_model = train_catboost()
