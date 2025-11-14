import random
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
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

def train_and_evaluate_nb(X_train, y_train, X_test, y_test, eligible_classes):
    """Train and evaluate Naive Bayes classifier."""
    print("Training Naive Bayes classifier...")
    
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    
    print("Classification report (Naive Bayes):")
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
    ax.set_title("Normalized Confusion Matrix (Naive Bayes)", fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix_nb_20news.png", dpi=150, bbox_inches='tight')
    print("Confusion matrix saved to confusion_matrix_nb_20news.png")
    plt.close()

def main():
    """Main function to run the 20newsgroups Naive Bayes training."""
    # Load 20newsgroups data
    X_train, y_train, X_test, y_test, eligible_classes = load_20newsgroups_data()
    
    print(f"Training on {len(X_train)} samples from {len(eligible_classes)} classes.")
    print(f"Testing on {len(X_test)} samples.")
    
    train_and_evaluate_nb(X_train, y_train, X_test, y_test, eligible_classes)

if __name__ == "__main__":
    main()
