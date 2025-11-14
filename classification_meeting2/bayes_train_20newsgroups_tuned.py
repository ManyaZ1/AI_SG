import random
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

def load_20newsgroups_data():
    """Load the 20newsgroups dataset from sklearn."""
    print("Loading 20newsgroups dataset...")
    
    # Load training and test data
    data_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    data_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    
    X_train, y_train = data_train.data, data_train.target
    X_test, y_test = data_test.data, data_test.target
    eligible_classes = data_train.target_names
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of classes: {len(eligible_classes)}")
    print(f"Classes: {eligible_classes}")
    
    return X_train, y_train, X_test, y_test, eligible_classes

def train_and_evaluate_nb_tuned(X_train, y_train, X_test, y_test, eligible_classes):
    """Train and evaluate Naive Bayes classifier with tuned parameters."""
    print("Training Naive Bayes classifier with optimized parameters...")
    
    # Enhanced TfidfVectorizer with more parameters
    vectorizer = TfidfVectorizer(
        max_features=20000,        # Increased vocabulary size
        min_df=2,                  # Ignore words appearing in < 2 documents
        max_df=0.85,               # Ignore words appearing in > 85% of documents
        ngram_range=(1, 2),        # Include unigrams and bigrams
        sublinear_tf=True,         # Use sublinear term frequency scaling
        stop_words='english',
        norm='l2'                  # L2 normalization
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Try both MultinomialNB and ComplementNB with different alpha values
    classifiers = {
        'MultinomialNB': MultinomialNB(alpha=0.1),      # Lower smoothing
        'ComplementNB': ComplementNB(alpha=0.1)         # Often better for text classification
    }
    
    best_score = 0
    best_classifier = None
    best_name = ""
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train_vec, y_train)
        score = clf.score(X_test_vec, y_test)
        print(f"{name} accuracy: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_classifier = clf
            best_name = name
    
    print(f"\nBest classifier: {best_name} with accuracy: {best_score:.4f}")
    
    # Use best classifier for final evaluation
    y_pred = best_classifier.predict(X_test_vec)
    
    print(f"\nClassification report ({best_name}):")
    print(classification_report(y_test, y_pred, target_names=eligible_classes, zero_division=0))
    
    print("Micro avg (precision, recall, f1-score):")
    report = classification_report(y_test, y_pred, target_names=eligible_classes, output_dict=True, zero_division=0)
    print("micro avg:", report.get("micro avg", {}))

    # Plot and save confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=eligible_classes
    )
    disp.plot(cmap="viridis", ax=ax, include_values=False)
    ax.set_title(f"Normalized Confusion Matrix ({best_name} - Tuned)", fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("bayes_confusion_matrix_tuned.png", dpi=150, bbox_inches='tight')
    print("Confusion matrix saved to bayes_confusion_matrix_tuned.png")
    plt.close()

def grid_search_tuning(X_train, y_train, X_test, y_test, eligible_classes):
    """Perform grid search to find optimal parameters."""
    print("Performing grid search for optimal parameters...")
    
    # Define parameter grid
    param_grid = {
        'vectorizer__max_features': [15000, 20000, 30000],
        'vectorizer__min_df': [1, 2, 3],
        'vectorizer__max_df': [0.8, 0.9, 0.95],
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'classifier__alpha': [0.01, 0.1, 0.5, 1.0]
    }
    
    # Create pipeline-like approach for grid search
    from sklearn.pipeline import Pipeline
    
    pipe = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english', sublinear_tf=True)),
        ('classifier', MultinomialNB())
    ])
    
    # Perform grid search with 3-fold cross-validation
    grid_search = GridSearchCV(
        pipe, 
        param_grid, 
        cv=3, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print("Fitting grid search (this may take a while)...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred = grid_search.predict(X_test)
    test_score = grid_search.score(X_test, y_test)
    print(f"Test set accuracy: {test_score:.4f}")
    
    return grid_search.best_estimator_

def main():
    """Main function to run the 20newsgroups Naive Bayes training."""
    # Load 20newsgroups data
    X_train, y_train, X_test, y_test, eligible_classes = load_20newsgroups_data()
    
    print(f"Training on {len(X_train)} samples from {len(eligible_classes)} classes.")
    print(f"Testing on {len(X_test)} samples.")
    
    # Method 1: Manual tuning with predefined good parameters
    #train_and_evaluate_nb_tuned(X_train, y_train, X_test, y_test, eligible_classes)
    
    # Method 2: Grid search (uncomment to run - takes longer)
    print("\n" + "="*50)
    grid_search_tuning(X_train, y_train, X_test, y_test, eligible_classes)

if __name__ == "__main__":
    main()
