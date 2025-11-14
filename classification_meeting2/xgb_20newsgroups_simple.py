import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# Load data
print("Loading 20newsgroups dataset...")
data_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
data_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

X_train, y_train = data_train.data, data_train.target
X_test, y_test = data_test.data, data_test.target
eligible_classes = data_train.target_names

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Number of classes: {len(eligible_classes)}")

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train XGBoost classifier
clf = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.3,
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False
)
print("Training XGBoost...")
clf.fit(X_train_vec, y_train)
print("Training done.")

# Predict and evaluate
print("Predicting...")
y_pred = clf.predict(X_test_vec)

print("Classification report (XGBoost):")
report_str = classification_report(y_test, y_pred, target_names=eligible_classes, zero_division=0)
print(report_str)

report = classification_report(y_test, y_pred, target_names=eligible_classes, output_dict=True, zero_division=0)
accuracy = report['accuracy']
macro_f1 = report['macro avg']['f1-score']
weighted_f1 = report['weighted avg']['f1-score']

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

# Plot and save normalized confusion matrix
cm = confusion_matrix(y_test, y_pred, normalize="true")
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=eligible_classes
)
disp.plot(cmap="viridis", ax=ax, include_values=False)
ax.set_title(f"Normalized Confusion Matrix (XGBoost)", fontsize=14)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

cm_filename = f"confusion_matrix_xgb_20news_simple.png"
plt.savefig(cm_filename, dpi=150, bbox_inches='tight')
print(f"Confusion matrix saved to {cm_filename}")
plt.close()

def test_individual_articles(clf, vectorizer, X_test, y_test, eligible_classes, num_samples=5):
    """Test classifier on individual articles and show predictions vs true labels."""
    print("\n" + "="*80)
    print("INDIVIDUAL ARTICLE PREDICTIONS")
    print("="*80)
    
    # Randomly select some test articles
    np.random.seed(42)
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        article = X_test[idx]
        true_label_idx = y_test[idx]
        true_label = eligible_classes[true_label_idx]
        
        # Vectorize the single article
        article_vec = vectorizer.transform([article])
        
        # Get prediction and probability
        pred_label_idx = clf.predict(article_vec)[0]
        pred_label = eligible_classes[pred_label_idx]
        pred_proba = clf.predict_proba(article_vec)[0]
        confidence = pred_proba[pred_label_idx]
        
        # Get top 3 predictions
        top3_indices = np.argsort(pred_proba)[-3:][::-1]
        top3_labels = [(eligible_classes[idx], pred_proba[idx]) for idx in top3_indices]
        
        print(f"\n--- ARTICLE {i+1} ---")
        print(f"TRUE LABEL: {true_label}")
        print(f"PREDICTED: {pred_label} (Confidence: {confidence:.3f})")
        print(f"CORRECT: {'✓' if pred_label == true_label else '✗'}")
        
        print(f"\nTop 3 predictions:")
        for j, (label, prob) in enumerate(top3_labels):
            marker = "★" if label == true_label else " "
            print(f"{marker} {j+1}. {label}: {prob:.3f}")
        
        print(f"\nARTICLE TEXT (first 500 chars):")
        print("-" * 50)
        print(article[:500] + "..." if len(article) > 500 else article)
        print("-" * 50)

# Test individual articles
test_individual_articles(clf, vectorizer, X_test, y_test, eligible_classes, num_samples=5)
