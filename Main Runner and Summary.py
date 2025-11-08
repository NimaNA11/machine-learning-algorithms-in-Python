"""
Machine Learning Algorithms - Main Runner and Summary
Comprehensive collection of classic ML algorithms with examples.

AVAILABLE ALGORITHMS:
=====================

1. SUPERVISED LEARNING - CLASSIFICATION
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Naive Bayes (Gaussian, Multinomial, Bernoulli)
   - Logistic Regression
   - Neural Networks (MLP Classifier)
   - Gradient Boosting (XGBoost, LightGBM, CatBoost)
   - AdaBoost
   - Bagging
   - Extra Trees
   - Voting Classifier
   - Stacking Classifier

2. SUPERVISED LEARNING - REGRESSION
   - Linear Regression
   - Ridge Regression (L2)
   - Lasso Regression (L1)
   - Elastic Net
   - Neural Networks (MLP Regressor)
   - SVR (Support Vector Regression)

3. UNSUPERVISED LEARNING - CLUSTERING
   - K-Means
   - DBSCAN
   - Hierarchical Clustering
   - Gaussian Mixture Models
   - Mean Shift
   - Spectral Clustering

4. DIMENSIONALITY REDUCTION
   - Principal Component Analysis (PCA)
   - t-SNE
   - Truncated SVD
   - Independent Component Analysis (ICA)
   - Non-negative Matrix Factorization (NMF)

5. ANOMALY DETECTION
   - Isolation Forest
   - Local Outlier Factor
   - One-Class SVM
   - Elliptic Envelope

6. TIME SERIES
   - Moving Average
   - Exponential Smoothing
   - ARIMA
   - SARIMA
   - Holt-Winters

7. FEATURE SELECTION & ENGINEERING
   - Variance Threshold
   - Univariate Selection
   - Recursive Feature Elimination (RFE)
   - Model-based Selection
   - Polynomial Features

INSTALLATION:
=============
pip install scikit-learn numpy matplotlib pandas

Optional packages for extended functionality:
pip install xgboost lightgbm catboost statsmodels

FILE STRUCTURE:
===============
1. decision_tree.py - Decision Tree Classifier
2. random_forest.py - Random Forest Classifier
3. svm_classifier.py - Support Vector Machine
4. knn_classifier.py - K-Nearest Neighbors
5. kmeans_clustering.py - K-Means Clustering
6. naive_bayes.py - Naive Bayes variants
7. logistic_regression.py - Logistic Regression
8. linear_regression.py - Linear Regression variants
9. gradient_boosting.py - Gradient Boosting algorithms
10. neural_networks.py - Multi-layer Perceptron
11. pca_dimensionality.py - PCA and dimensionality reduction
12. clustering_algorithms.py - Various clustering methods
13. ensemble_methods.py - Ensemble learning techniques
14. anomaly_detection.py - Outlier detection methods
15. time_series.py - Time series analysis
16. feature_selection.py - Feature selection & engineering
17. ml_algorithms_main.py - This file

USAGE EXAMPLES:
===============
"""

import warnings
warnings.filterwarnings('ignore')

def print_algorithm_summary():
    """Print a comprehensive summary of all algorithms"""
    
    print("\n" + "=" * 80)
    print("MACHINE LEARNING ALGORITHMS CHEAT SHEET")
    print("=" * 80)
    
    algorithms = {
        "CLASSIFICATION ALGORITHMS": {
            "Decision Tree": {
                "Use Case": "Interpretable, handles non-linear relationships",
                "Pros": "Easy to understand, no scaling needed",
                "Cons": "Can overfit, unstable",
                "Best For": "Small to medium datasets, need interpretability"
            },
            "Random Forest": {
                "Use Case": "General purpose, handles high dimensions",
                "Pros": "Reduces overfitting, feature importance",
                "Cons": "Can be slow, less interpretable",
                "Best For": "Most classification tasks, robust baseline"
            },
            "SVM": {
                "Use Case": "High-dimensional data, clear margin of separation",
                "Pros": "Effective in high dimensions, memory efficient",
                "Cons": "Slow on large datasets, requires scaling",
                "Best For": "Text classification, image recognition"
            },
            "KNN": {
                "Use Case": "Simple baseline, non-parametric",
                "Pros": "Simple, no training phase",
                "Cons": "Slow predictions, memory intensive",
                "Best For": "Small datasets, pattern recognition"
            },
            "Naive Bayes": {
                "Use Case": "Text classification, fast training",
                "Pros": "Fast, works well with small data",
                "Cons": "Assumes feature independence",
                "Best For": "Text classification, spam detection"
            },
            "Logistic Regression": {
                "Use Case": "Binary classification, baseline model",
                "Pros": "Fast, interpretable, probability outputs",
                "Cons": "Assumes linear relationships",
                "Best For": "Binary classification, need probabilities"
            },
            "Neural Networks": {
                "Use Case": "Complex patterns, large datasets",
                "Pros": "Handles complex relationships",
                "Cons": "Requires lots of data, hyperparameter tuning",
                "Best For": "Large datasets, complex patterns"
            },
            "Gradient Boosting": {
                "Use Case": "Highest accuracy, competitions",
                "Pros": "Often best performance, handles missing data",
                "Cons": "Can overfit, slow training",
                "Best For": "Kaggle competitions, maximum accuracy"
            }
        },
        
        "REGRESSION ALGORITHMS": {
            "Linear Regression": {
                "Use Case": "Continuous prediction, baseline",
                "Pros": "Simple, interpretable, fast",
                "Cons": "Assumes linearity, sensitive to outliers",
                "Best For": "Simple relationships, baseline"
            },
            "Ridge/Lasso": {
                "Use Case": "Regularized regression",
                "Pros": "Prevents overfitting, feature selection (Lasso)",
                "Cons": "May underfit with too much regularization",
                "Best For": "High-dimensional data, multicollinearity"
            }
        },
        
        "CLUSTERING ALGORITHMS": {
            "K-Means": {
                "Use Case": "General clustering, known number of clusters",
                "Pros": "Fast, simple, scalable",
                "Cons": "Need to specify k, assumes spherical clusters",
                "Best For": "Customer segmentation, image compression"
            },
            "DBSCAN": {
                "Use Case": "Unknown number of clusters, arbitrary shapes",
                "Pros": "Finds arbitrary shapes, identifies noise",
                "Cons": "Struggles with varying densities",
                "Best For": "Spatial data, anomaly detection"
            },
            "Hierarchical": {
                "Use Case": "Hierarchical relationships, dendrograms",
                "Pros": "No need to specify k, creates hierarchy",
                "Cons": "Slow on large datasets",
                "Best For": "Small datasets, need hierarchy"
            }
        },
        
        "DIMENSIONALITY REDUCTION": {
            "PCA": {
                "Use Case": "Feature reduction, visualization",
                "Pros": "Fast, reduces noise",
                "Cons": "Linear, loses interpretability",
                "Best For": "Pre-processing, visualization"
            },
            "t-SNE": {
                "Use Case": "Visualization only",
                "Pros": "Great visualizations",
                "Cons": "Slow, can't transform new data easily",
                "Best For": "2D/3D visualizations"
            }
        }
    }
    
    for category, algos in algorithms.items():
        print(f"\n{category}")
        print("-" * 80)
        for algo_name, details in algos.items():
            print(f"\n{algo_name}:")
            for key, value in details.items():
                print(f"  {key}: {value}")

def print_quick_selection_guide():
    """Print a decision tree for algorithm selection"""
    
    print("\n" + "=" * 80)
    print("ALGORITHM SELECTION GUIDE")
    print("=" * 80)
    
    guide = """
    CLASSIFICATION TASK?
    │
    ├─ YES: Is your data labeled?
    │   │
    │   ├─ Small dataset (< 1000 samples)
    │   │   → Try: Naive Bayes, Logistic Regression, SVM
    │   │
    │   ├─ Medium dataset (1K - 100K samples)
    │   │   → Try: Random Forest, Gradient Boosting, Neural Network
    │   │
    │   └─ Large dataset (> 100K samples)
    │       → Try: Gradient Boosting (XGBoost/LightGBM), Neural Network
    │
    ├─ NO (CLUSTERING): How many clusters?
    │   │
    │   ├─ Known number of clusters
    │   │   → Try: K-Means, Gaussian Mixture
    │   │
    │   └─ Unknown number of clusters
    │       → Try: DBSCAN, Hierarchical, Mean Shift
    │
    ├─ ANOMALY DETECTION?
    │   → Try: Isolation Forest, One-Class SVM, LOF
    │
    ├─ TIME SERIES?
    │   │
    │   ├─ Short-term forecast
    │   │   → Try: ARIMA, Exponential Smoothing
    │   │
    │   └─ Long-term forecast
    │       → Try: SARIMA, Prophet, Neural Networks (LSTM)
    │
    └─ DIMENSIONALITY REDUCTION?
        │
        ├─ For preprocessing
        │   → Try: PCA, SVD
        │
        └─ For visualization
            → Try: t-SNE, PCA
    
    GENERAL TIPS:
    ============
    1. Always start with a simple baseline (Logistic Regression, Decision Tree)
    2. Try Random Forest next - it's robust and works well on most problems
    3. For maximum accuracy: Gradient Boosting (XGBoost, LightGBM)
    4. Always scale your features for: SVM, KNN, Neural Networks, Clustering
    5. Use cross-validation to evaluate models properly
    6. Feature engineering often matters more than algorithm choice
    """
    
    print(guide)

def print_preprocessing_guide():
    """Print preprocessing recommendations"""
    
    print("\n" + "=" * 80)
    print("PREPROCESSING GUIDE")
    print("=" * 80)
    
    guide = """
    SCALING REQUIRED:
    ================
    ✓ SVM, KNN, Neural Networks, Logistic Regression
    ✓ K-Means, Hierarchical Clustering
    ✓ PCA, LDA
    ✗ Tree-based methods (Decision Tree, Random Forest, Gradient Boosting)
    
    SCALING METHODS:
    ===============
    - StandardScaler: Mean=0, Std=1 (most common)
    - MinMaxScaler: Scale to [0, 1]
    - RobustScaler: Use median and IQR (good for outliers)
    
    HANDLING MISSING DATA:
    =====================
    - Mean/Median imputation: Simple, works for most cases
    - KNN imputation: More sophisticated
    - Tree-based methods: Can handle missing values natively
    
    ENCODING CATEGORICAL VARIABLES:
    ==============================
    - One-Hot Encoding: For nominal categories
    - Label Encoding: For ordinal categories
    - Target Encoding: For high cardinality
    
    HANDLING IMBALANCED DATA:
    ========================
    - Oversampling: SMOTE
    - Undersampling: Random undersampling
    - Class weights: Most algorithms support this
    - Ensemble methods: Often robust to imbalance
    """
    
    print(guide)

def print_hyperparameter_tuning_guide():
    """Print hyperparameter tuning recommendations"""
    
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING GUIDE")
    print("=" * 80)
    
    guide = """
    METHODS:
    ========
    1. Grid Search: Exhaustive search over parameter grid
       → Use when: Small parameter space, need best results
    
    2. Random Search: Random sampling of parameter space
       → Use when: Large parameter space, faster results needed
    
    3. Bayesian Optimization: Smart search using past results
       → Use when: Expensive evaluations, complex space
    
    KEY PARAMETERS BY ALGORITHM:
    ===========================
    
    Random Forest:
    - n_estimators: 100-500 (more is usually better)
    - max_depth: None, or limit to prevent overfitting
    - min_samples_split: 2-10
    - max_features: 'sqrt' for classification, 'log2' or fraction
    
    Gradient Boosting:
    - n_estimators: 100-1000 (use early stopping)
    - learning_rate: 0.01-0.3 (lower for more estimators)
    - max_depth: 3-10 (shallow trees work well)
    - subsample: 0.5-1.0 (introduce randomness)
    
    SVM:
    - C: 0.1, 1, 10, 100 (regularization)
    - kernel: 'rbf', 'linear', 'poly'
    - gamma: 'scale', 'auto', or custom values
    
    Neural Networks:
    - hidden_layer_sizes: (100,), (100, 50), etc.
    - activation: 'relu', 'tanh', 'logistic'
    - alpha: 0.0001-0.01 (regularization)
    - learning_rate_init: 0.001-0.01
    
    K-Means:
    - n_clusters: Use elbow method or silhouette score
    - init: 'k-means++' (recommended)
    
    BEST PRACTICES:
    ==============
    1. Use cross-validation during tuning
    2. Start with default parameters
    3. Tune one parameter at a time initially
    4. Use learning curves to diagnose over/underfitting
    5. Consider computational budget
    """
    
    print(guide)

def main():
    """Main function to display all information"""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "MACHINE LEARNING ALGORITHMS REFERENCE")
    print("=" * 80)
    print("\nWelcome to the comprehensive ML algorithms reference!")
    print("This collection includes implementations of all classic ML algorithms.")
    print("\nEach algorithm is in a separate file with:")
    print("  • Complete working examples")
    print("  • Detailed comments")
    print("  • Evaluation metrics")
    print("  • Visualization code")
    print("\n" + "=" * 80)
    
    # Print all guides
    print_algorithm_summary()
    print_quick_selection_guide()
    print_preprocessing_guide()
    print_hyperparameter_tuning_guide()
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("""
    1. Understand your problem (classification, regression, clustering, etc.)
    2. Explore and preprocess your data
    3. Start with a simple baseline model
    4. Try 2-3 different algorithms
    5. Tune hyperparameters of the best performer
    6. Ensemble multiple models for best results
    7. Evaluate on hold-out test set
    
    Remember: "There is no free lunch" - no single algorithm is best for all problems!
    """)
    
    print("=" * 80)
    print("Happy Machine Learning! 🚀")
    print("=" * 80)

if __name__ == "__main__":
    main()
