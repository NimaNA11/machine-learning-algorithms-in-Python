# 🤖 Classic Machine Learning Algorithms

A comprehensive collection of classic machine learning algorithms implemented in Python using scikit-learn. Perfect for learning, reference, or as a starting point for your ML projects.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Overview](#overview)
- [Algorithms Included](#algorithms-included)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Algorithm Selection Guide](#algorithm-selection-guide)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository contains **17 well-documented Python files** implementing over **40 machine learning algorithms**. Each algorithm includes:

- ✅ Complete working examples with real datasets
- ✅ Detailed comments and documentation
- ✅ Performance evaluation metrics
- ✅ Visualization code
- ✅ Hyperparameter tuning examples
- ✅ Comparison with other methods

Perfect for:
- **Students** learning machine learning
- **Data Scientists** needing quick reference implementations
- **Developers** looking for production-ready code templates
- **Researchers** comparing algorithm performance

## 🧠 Algorithms Included

### Supervised Learning - Classification
| Algorithm | File | Use Case |
|-----------|------|----------|
| Decision Tree | `decision_tree.py` | Interpretable models, feature importance |
| Random Forest | `random_forest.py` | General purpose, robust baseline |
| SVM | `svm_classifier.py` | High-dimensional data, text classification |
| KNN | `knn_classifier.py` | Pattern recognition, simple baseline |
| Naive Bayes | `naive_bayes.py` | Text classification, spam detection |
| Logistic Regression | `logistic_regression.py` | Binary classification, probability outputs |
| Neural Networks | `neural_networks.py` | Complex patterns, large datasets |
| Gradient Boosting | `gradient_boosting.py` | Maximum accuracy, competitions |
| Ensemble Methods | `ensemble_methods.py` | AdaBoost, Bagging, Voting, Stacking |

### Supervised Learning - Regression
| Algorithm | File | Use Case |
|-----------|------|----------|
| Linear Regression | `linear_regression.py` | Continuous prediction, baseline |
| Ridge Regression | `linear_regression.py` | L2 regularization, multicollinearity |
| Lasso Regression | `linear_regression.py` | L1 regularization, feature selection |
| Elastic Net | `linear_regression.py` | Combined L1 + L2 regularization |
| MLP Regressor | `neural_networks.py` | Non-linear regression |

### Unsupervised Learning - Clustering
| Algorithm | File | Use Case |
|-----------|------|----------|
| K-Means | `kmeans_clustering.py` | Customer segmentation, compression |
| DBSCAN | `clustering_algorithms.py` | Arbitrary shapes, noise detection |
| Hierarchical | `clustering_algorithms.py` | Dendrograms, hierarchical relationships |
| Gaussian Mixture | `clustering_algorithms.py` | Soft clustering, probability-based |
| Mean Shift | `clustering_algorithms.py` | Unknown number of clusters |
| Spectral | `clustering_algorithms.py` | Non-convex clusters |

### Dimensionality Reduction
| Algorithm | File | Use Case |
|-----------|------|----------|
| PCA | `pca_dimensionality.py` | Feature reduction, noise reduction |
| t-SNE | `pca_dimensionality.py` | 2D/3D visualization |
| Truncated SVD | `pca_dimensionality.py` | Sparse data, LSA |
| ICA | `pca_dimensionality.py` | Signal separation |
| NMF | `pca_dimensionality.py` | Non-negative data, topic modeling |

### Anomaly Detection
| Algorithm | File | Use Case |
|-----------|------|----------|
| Isolation Forest | `anomaly_detection.py` | Fast, scalable outlier detection |
| Local Outlier Factor | `anomaly_detection.py` | Density-based anomaly detection |
| One-Class SVM | `anomaly_detection.py` | Novelty detection |
| Elliptic Envelope | `anomaly_detection.py` | Gaussian distribution outliers |

### Time Series Analysis
| Algorithm | File | Use Case |
|-----------|------|----------|
| Moving Average | `time_series.py` | Simple smoothing and forecasting |
| Exponential Smoothing | `time_series.py` | Weighted smoothing |
| ARIMA | `time_series.py` | Univariate forecasting |
| SARIMA | `time_series.py` | Seasonal forecasting |
| Holt-Winters | `time_series.py` | Trend and seasonality |

### Feature Engineering
| Method | File | Use Case |
|--------|------|----------|
| Variance Threshold | `feature_selection.py` | Remove low-variance features |
| Univariate Selection | `feature_selection.py` | Statistical feature selection |
| RFE | `feature_selection.py` | Recursive feature elimination |
| Model-based Selection | `feature_selection.py` | Feature importance-based |
| Polynomial Features | `feature_selection.py` | Create interaction terms |

## 🚀 Installation

### Basic Requirements
```bash
# Clone the repository
git clone https://github.com/yourusername/ml-algorithms.git
cd ml-algorithms

# Install required packages
pip install scikit-learn numpy matplotlib pandas
```

### Optional Packages (for extended functionality)
```bash
# For advanced gradient boosting
pip install xgboost lightgbm catboost

# For time series analysis
pip install statsmodels

# For deep learning examples
pip install tensorflow torch
```

### Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## 💻 Usage

### Quick Start

Run any algorithm file directly:
```bash
python decision_tree.py
python random_forest.py
python kmeans_clustering.py
```

### Using as a Module

```python
# Import specific algorithm
from decision_tree import train_decision_tree
from random_forest import train_random_forest

# Train models
dt_model = train_decision_tree()
rf_model = train_random_forest()
```

### Example: Classification Pipeline

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = load_iris()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

### Example: Clustering Pipeline

```python
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Generate data
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cluster
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Evaluate
score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {score:.4f}")
```

## 📁 File Structure

```
ml-algorithms/
│
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
│
├── classification/
│   ├── decision_tree.py          # Decision Tree implementation
│   ├── random_forest.py          # Random Forest implementation
│   ├── svm_classifier.py         # SVM implementation
│   ├── knn_classifier.py         # KNN implementation
│   ├── naive_bayes.py            # Naive Bayes variants
│   ├── logistic_regression.py    # Logistic Regression
│   └── neural_networks.py        # MLP Classifier/Regressor
│
├── regression/
│   └── linear_regression.py      # Linear, Ridge, Lasso, Elastic Net
│
├── clustering/
│   ├── kmeans_clustering.py      # K-Means implementation
│   └── clustering_algorithms.py  # DBSCAN, Hierarchical, GMM, etc.
│
├── dimensionality_reduction/
│   └── pca_dimensionality.py     # PCA, t-SNE, SVD, ICA, NMF
│
├── ensemble/
│   ├── gradient_boosting.py      # XGBoost, LightGBM, CatBoost
│   └── ensemble_methods.py       # AdaBoost, Bagging, Voting, Stacking
│
├── anomaly_detection/
│   └── anomaly_detection.py      # Isolation Forest, LOF, One-Class SVM
│
├── time_series/
│   └── time_series.py            # ARIMA, SARIMA, Exponential Smoothing
│
├── feature_engineering/
│   └── feature_selection.py      # Various feature selection methods
│
├── utils/
│   ├── data_preprocessing.py     # Data cleaning utilities
│   ├── model_evaluation.py       # Evaluation metrics
│   └── visualization.py          # Plotting utilities
│
└── examples/
    ├── complete_pipeline.py      # End-to-end ML pipeline
    ├── model_comparison.py       # Compare multiple algorithms
    └── hyperparameter_tuning.py  # Tuning examples
```

## 🎓 Algorithm Selection Guide

### Decision Tree
```
┌─ CLASSIFICATION TASK?
│
├─ Small dataset (< 1000 samples)
│   → Naive Bayes, Logistic Regression, SVM
│
├─ Medium dataset (1K - 100K samples)
│   → Random Forest, Gradient Boosting, Neural Network
│
└─ Large dataset (> 100K samples)
    → Gradient Boosting (XGBoost/LightGBM), Deep Learning
```

### When to Use Each Algorithm

| Problem Type | Best Algorithms | Notes |
|--------------|----------------|-------|
| Binary Classification | Logistic Regression, SVM, Random Forest | Start with Logistic Regression |
| Multi-class Classification | Random Forest, Gradient Boosting, Neural Network | Use one-vs-rest if needed |
| Imbalanced Classes | Gradient Boosting, Random Forest with class weights | Consider SMOTE oversampling |
| High-dimensional data | SVM, Lasso Regression, Random Forest | Apply PCA for dimensionality reduction |
| Need Interpretability | Decision Tree, Logistic Regression, Linear Regression | Avoid black-box models |
| Text Classification | Naive Bayes, SVM, Neural Networks | Use TF-IDF or embeddings |
| Time Series | ARIMA, SARIMA, Prophet | Check for stationarity first |
| Anomaly Detection | Isolation Forest, One-Class SVM | Consider domain knowledge |
| Clustering | K-Means (known K), DBSCAN (unknown K) | Use elbow method for K |

### Preprocessing Requirements

| Algorithm | Scaling Required | Handles Missing | Handles Categorical |
|-----------|-----------------|-----------------|---------------------|
| Decision Tree | ❌ | ✅ | ✅ |
| Random Forest | ❌ | ✅ | ✅ |
| SVM | ✅ | ❌ | ❌ |
| KNN | ✅ | ❌ | ❌ |
| Naive Bayes | ❌ | ❌ | ✅ |
| Logistic Regression | ✅ | ❌ | ❌ |
| Neural Networks | ✅ | ❌ | ❌ |
| Gradient Boosting | ❌ | ✅ | ✅ |
| K-Means | ✅ | ❌ | ❌ |

## 📊 Performance Comparison

Typical performance on standard datasets:

| Algorithm | Iris (Acc) | Wine (Acc) | Digits (Acc) | Training Time |
|-----------|-----------|-----------|--------------|---------------|
| Decision Tree | 96% | 92% | 85% | Fast |
| Random Forest | 97% | 98% | 97% | Medium |
| SVM | 98% | 98% | 98% | Slow |
| KNN | 97% | 96% | 98% | Instant/Slow predict |
| Naive Bayes | 95% | 97% | 84% | Very Fast |
| Logistic Regression | 97% | 97% | 96% | Fast |
| Neural Network | 98% | 98% | 98% | Slow |
| Gradient Boosting | 97% | 99% | 98% | Medium |

*Note: Results vary based on hyperparameter tuning and data preprocessing*

## 🛠️ Advanced Features

### Cross-Validation Example
```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=5)
print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
```

### Pipeline Example
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

## 📚 Learning Resources

### Recommended Books
- **"Hands-On Machine Learning"** by Aurélien Géron
- **"Pattern Recognition and Machine Learning"** by Christopher Bishop
- **"The Elements of Statistical Learning"** by Hastie, Tibshirani, Friedman

### Online Courses
- [Coursera: Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning)
- [fast.ai: Practical Deep Learning](https://course.fast.ai/)
- [Kaggle Learn](https://www.kaggle.com/learn)

### Documentation
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Contribution Guidelines
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new algorithms
- Update documentation as needed
- Ensure all tests pass before submitting

## 🐛 Issues and Support

Found a bug or have a question? Please:
1. Check existing [Issues](https://github.com/NimaNA11/ml-algorithms/issues)
2. Create a new issue with detailed description
3. Use appropriate labels (bug, enhancement, question)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- **scikit-learn** team for the excellent ML library
- **Python community** for continuous support
- All **contributors** who help improve this repository

## 📬 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/NimaNA11)
- **LinkedIn**: [Your Profile](https://linkedin.com/in/NimaNA11)

## ⭐ Star History

If you find this repository helpful, please consider giving it a star! ⭐

---

**Made with ❤️ for the ML community**

*Last Updated: November 2025*
