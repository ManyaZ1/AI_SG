import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from print_document_details import load_from_pickle

def test_split_and_training():
    # Load the data
    doc_dict = load_from_pickle("documents.pkl")

    # Convert to DataFrame
    data = pd.DataFrame.from_dict(doc_dict, orient="index", columns=["text", "category"])
    data.reset_index(drop=True, inplace=True)  # Reset index to ensure compatibility

    # Split data
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Re-encode labels after splitting
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_data["category"])
    y_test = label_encoder.transform(test_data["category"])

    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))  # Reduced features for testing
    X_train = vectorizer.fit_transform(train_data["text"])
    X_test = vectorizer.transform(test_data["text"])

    # Train XGBoost classifier
    clf = xgb.XGBClassifier(objective="multi:softmax", num_class=len(label_encoder.classes_), max_depth=3, n_estimators=10)
    clf.fit(X_train, y_train)

    # Evaluate
    accuracy = clf.score(X_test, y_test)
    print(f"Test completed successfully. Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    test_split_and_training()