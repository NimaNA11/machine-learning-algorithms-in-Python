"""
Neural Networks - Multi-layer Perceptron (MLP)
Feed-forward artificial neural networks with multiple hidden layers.
"""

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.datasets import load_digits, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

def train_mlp_classifier():
    """Multi-layer Perceptron for Classification"""
    print("=" * 50)
    print("MLP CLASSIFIER")
    print("=" * 50)
    
    # Load dataset
    data = load_digits()
    X, y = data.data, data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features (important for neural networks!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # two hidden layers
        activation='relu',              # 'relu', 'tanh', 'logistic'
        solver='adam',                  # 'sgd', 'adam', 'lbfgs'
        alpha=0.0001,                   # L2 regularization
        batch_size='auto',
        learning_rate='adaptive',       # 'constant', 'invscaling', 'adaptive'
        learning_rate_init=0.001,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Number of iterations: {model.n_iter_}")
    print(f"Number of layers: {model.n_layers_}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_curve_)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('MLP Training Loss Curve')
    plt.grid(True)
    plt.savefig('mlp_loss_curve.png')
    print("\nLoss curve saved as 'mlp_loss_curve.png'")
    
    return model, scaler

def train_mlp_regressor():
    """Multi-layer Perceptron for Regression"""
    print("\n" + "=" * 50)
    print("MLP REGRESSOR")
    print("=" * 50)
    
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
    
    # Create and train model
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Number of iterations: {model.n_iter_}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('MLP Regressor: Actual vs Predicted')
    plt.grid(True)
    plt.savefig('mlp_regressor_predictions.png')
    print("Prediction plot saved as 'mlp_regressor_predictions.png'")
    
    return model, scaler

def compare_architectures():
    """Compare different network architectures"""
    print("\n" + "=" * 50)
    print("COMPARING DIFFERENT ARCHITECTURES")
    print("=" * 50)
    
    data = load_digits()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    architectures = [
        (50,),
        (100,),
        (100, 50),
        (100, 50, 25),
        (200, 100, 50)
    ]
    
    results = []
    for arch in architectures:
        model = MLPClassifier(
            hidden_layer_sizes=arch,
            max_iter=200,
            random_state=42,
            verbose=False
        )
        model.fit(X_train_scaled, y_train)
        accuracy = model.score(X_test_scaled, y_test)
        results.append((arch, accuracy))
        print(f"Architecture {arch}: Accuracy = {accuracy:.4f}")
    
    return results

if __name__ == "__main__":
    clf_model, clf_scaler = train_mlp_classifier()
    reg_model, reg_scaler = train_mlp_regressor()
    architecture_results = compare_architectures()
