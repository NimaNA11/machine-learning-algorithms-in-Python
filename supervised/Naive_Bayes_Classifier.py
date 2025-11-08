"""
Naive Bayes Classifier
Probabilistic classifier based on Bayes' theorem with independence assumptions.
"""

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

def train_gaussian_nb():
    """Gaussian Naive Bayes - for continuous features"""
    print("=" * 50)
    print("GAUSSIAN NAIVE BAYES")
    print("=" * 50)
    
    # Load dataset
    data = load_iris()
    X, y = data.data, data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = GaussianNB(
        var_smoothing=1e-9  # portion of largest variance added to variances
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    print("\nClass Prior Probabilities:")
    for i, prob in enumerate(model.class_prior_):
        print(f"{data.target_names[i]}: {prob:.4f}")
    
    return model

def train_multinomial_nb():
    """Multinomial Naive Bayes - for discrete count data (text classification)"""
    print("\n" + "=" * 50)
    print("MULTINOMIAL NAIVE BAYES (Text Classification)")
    print("=" * 50)
    
    # Load text data
    categories = ['sci.space', 'comp.graphics']
    newsgroups = fetch_20newsgroups(
        subset='all',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        newsgroups.data, newsgroups.target, test_size=0.2, random_state=42
    )
    
    # Vectorize text
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)
    
    # Create and train model
    model = MultinomialNB(alpha=1.0)  # alpha is smoothing parameter
    model.fit(X_train_counts, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_counts)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=categories))
    
    # Show most informative features
    print("\nMost informative features per class:")
    feature_names = vectorizer.get_feature_names_out()
    for i, category in enumerate(categories):
        top_indices = np.argsort(model.feature_log_prob_[i])[-10:]
        top_features = [feature_names[idx] for idx in top_indices]
        print(f"\n{category}: {', '.join(top_features)}")
    
    return model, vectorizer

def train_bernoulli_nb():
    """Bernoulli Naive Bayes - for binary/boolean features"""
    print("\n" + "=" * 50)
    print("BERNOULLI NAIVE BAYES")
    print("=" * 50)
    
    # Generate binary data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )
    
    # Binarize features
    X_binary = (X > 0).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_binary, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = BernoulliNB(alpha=1.0, binarize=0.0)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model

if __name__ == "__main__":
    gaussian_model = train_gaussian_nb()
    multinomial_model, vectorizer = train_multinomial_nb()
    bernoulli_model = train_bernoulli_nb()
