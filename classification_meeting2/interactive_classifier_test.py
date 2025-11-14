import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import re

def load_and_train_classifier():
    """Load data and train the XGBoost classifier."""
    print("Loading 20newsgroups dataset...")
    data_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    data_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    
    X_train, y_train = data_train.data, data_train.target
    X_test, y_test = data_test.data, data_test.target
    eligible_classes = data_train.target_names
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Train classifier
    print("Training XGBoost classifier...")
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    clf.fit(X_train_vec, y_train)
    
    return clf, vectorizer, X_test, y_test, eligible_classes

def find_articles_by_category(X_test, y_test, eligible_classes, target_category, max_articles=5):
    """Find articles from a specific category."""
    try:
        category_idx = eligible_classes.index(target_category)
    except ValueError:
        print(f"Category '{target_category}' not found!")
        print("Available categories:", eligible_classes)
        return []
    
    # Find indices of articles in this category
    category_indices = np.where(y_test == category_idx)[0]
    
    # Randomly select some articles from this category
    np.random.seed(42)
    selected_indices = np.random.choice(category_indices, 
                                      min(max_articles, len(category_indices)), 
                                      replace=False)
    
    return [(idx, X_test[idx], y_test[idx]) for idx in selected_indices]

def find_articles_by_keyword(X_test, y_test, keyword, max_articles=5):
    """Find articles containing a specific keyword."""
    matching_articles = []
    
    for idx, article in enumerate(X_test):
        if re.search(keyword, article, re.IGNORECASE) and len(matching_articles) < max_articles:
            matching_articles.append((idx, article, y_test[idx]))
    
    return matching_articles

def analyze_article(clf, vectorizer, article_idx, article_text, true_label_idx, eligible_classes):
    """Analyze a single article with detailed predictions."""
    true_label = eligible_classes[true_label_idx]
    
    # Vectorize the article
    article_vec = vectorizer.transform([article_text])
    
    # Get prediction and probabilities
    pred_label_idx = clf.predict(article_vec)[0]
    pred_label = eligible_classes[pred_label_idx]
    pred_proba = clf.predict_proba(article_vec)[0]
    confidence = pred_proba[pred_label_idx]
    
    # Get top 5 predictions
    top5_indices = np.argsort(pred_proba)[-5:][::-1]
    top5_predictions = [(eligible_classes[idx], pred_proba[idx]) for idx in top5_indices]
    
    print(f"\n{'='*80}")
    print(f"ARTICLE ANALYSIS (Index: {article_idx})")
    print(f"{'='*80}")
    
    print(f"TRUE LABEL: {true_label}")
    print(f"PREDICTED: {pred_label}")
    print(f"CONFIDENCE: {confidence:.3f}")
    print(f"CORRECT: {'✓ YES' if pred_label == true_label else '✗ NO'}")
    
    print(f"\nTOP 5 PREDICTIONS:")
    for i, (label, prob) in enumerate(top5_predictions):
        marker = "★" if label == true_label else " "
        print(f"{marker} {i+1:2d}. {label:25s}: {prob:.4f}")
    
    print(f"\nARTICLE TEXT:")
    print("-" * 80)
    print(article_text[:1000] + "..." if len(article_text) > 1000 else article_text)
    print("-" * 80)

def interactive_test():
    """Interactive testing interface."""
    # Load and train
    clf, vectorizer, X_test, y_test, eligible_classes = load_and_train_classifier()
    
    print(f"\n{'='*80}")
    print("INTERACTIVE CLASSIFIER TESTING")
    print(f"{'='*80}")
    print("Available categories:")
    for i, cat in enumerate(eligible_classes):
        print(f"{i:2d}. {cat}")
    
    while True:
        print(f"\n{'='*50}")
        print("OPTIONS:")
        print("1. Test random articles")
        print("2. Test articles from specific category")
        print("3. Test articles containing keyword")
        print("4. Test specific article by index")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            num_articles = int(input("How many random articles to test? (1-10): ") or 5)
            indices = np.random.choice(len(X_test), min(num_articles, len(X_test)), replace=False)
            
            for idx in indices:
                analyze_article(clf, vectorizer, idx, X_test[idx], y_test[idx], eligible_classes)
                
        elif choice == "2":
            print("\nAvailable categories:")
            for i, cat in enumerate(eligible_classes):
                print(f"{i:2d}. {cat}")
            
            try:
                cat_num = int(input("\nEnter category number: "))
                if 0 <= cat_num < len(eligible_classes):
                    category = eligible_classes[cat_num]
                    articles = find_articles_by_category(X_test, y_test, eligible_classes, category, 3)
                    
                    for idx, article, true_label in articles:
                        analyze_article(clf, vectorizer, idx, article, true_label, eligible_classes)
                else:
                    print("Invalid category number!")
            except ValueError:
                print("Please enter a valid number!")
                
        elif choice == "3":
            keyword = input("Enter keyword to search for: ").strip()
            if keyword:
                articles = find_articles_by_keyword(X_test, y_test, keyword, 3)
                
                if articles:
                    for idx, article, true_label in articles:
                        analyze_article(clf, vectorizer, idx, article, true_label, eligible_classes)
                else:
                    print(f"No articles found containing '{keyword}'")
            
        elif choice == "4":
            try:
                idx = int(input(f"Enter article index (0-{len(X_test)-1}): "))
                if 0 <= idx < len(X_test):
                    analyze_article(clf, vectorizer, idx, X_test[idx], y_test[idx], eligible_classes)
                else:
                    print("Invalid index!")
            except ValueError:
                print("Please enter a valid number!")
                
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please enter 1-5.")

if __name__ == "__main__":
    interactive_test()
