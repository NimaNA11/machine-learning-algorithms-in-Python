"""
Linear Regression
Predicts continuous values by fitting a linear relationship between features and target.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

def train_linear_regression():
    # Load dataset
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("=" * 50)
    print("ORDINARY LEAST SQUARES REGRESSION")
    print("=" * 50)
    
    # Ordinary Least Squares
    ols_model = LinearRegression()
    ols_model.fit(X_train_scaled, y_train)
    y_pred_ols = ols_model.predict(X_test_scaled)
    
    print(f"R² Score: {r2_score(y_test, y_pred_ols):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ols)):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred_ols):.4f}")
    
    print("\n" + "=" * 50)
    print("RIDGE REGRESSION (L2 Regularization)")
    print("=" * 50)
    
    # Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge_model.predict(X_test_scaled)
    
    print(f"R² Score: {r2_score(y_test, y_pred_ridge):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred_ridge):.4f}")
    
    print("\n" + "=" * 50)
    print("LASSO REGRESSION (L1 Regularization)")
    print("=" * 50)
    
    # Lasso Regression
    lasso_model = Lasso(alpha=0.1, max_iter=10000)
    lasso_model.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso_model.predict(X_test_scaled)
    
    print(f"R² Score: {r2_score(y_test, y_pred_lasso):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lasso)):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred_lasso):.4f}")
    print(f"Number of non-zero coefficients: {np.sum(lasso_model.coef_ != 0)}")
    
    print("\n" + "=" * 50)
    print("ELASTIC NET (L1 + L2 Regularization)")
    print("=" * 50)
    
    # Elastic Net
    elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
    elastic_model.fit(X_train_scaled, y_train)
    y_pred_elastic = elastic_model.predict(X_test_scaled)
    
    print(f"R² Score: {r2_score(y_test, y_pred_elastic):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_elastic)):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred_elastic):.4f}")
    
    # Feature importance comparison
    print("\n" + "=" * 50)
    print("FEATURE COEFFICIENTS")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    models = [ols_model, ridge_model, lasso_model, elastic_model]
    titles = ['OLS', 'Ridge', 'Lasso', 'Elastic Net']
    
    for ax, model, title in zip(axes.flat, models, titles):
        coef = model.coef_
        ax.barh(data.feature_names, coef)
        ax.set_xlabel('Coefficient Value')
        ax.set_title(f'{title} Coefficients')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_regression_coefficients.png')
    print("Coefficient comparison saved as 'linear_regression_coefficients.png'")
    
    # Prediction vs Actual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_ols, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Linear Regression: Actual vs Predicted')
    plt.grid(True, alpha=0.3)
    plt.savefig('linear_regression_predictions.png')
    print("Prediction plot saved as 'linear_regression_predictions.png'")
    
    return ols_model, ridge_model, lasso_model, elastic_model, scaler

if __name__ == "__main__":
    models = train_linear_regression()
