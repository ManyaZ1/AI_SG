# TF-IDF XGBoost pipeline

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder


def load_data(text_file_path, label_file_path):
    # Each line in text_file is a document, each line in label_file is the corresponding label
    with open(text_file_path, encoding="utf-8") as f:
        texts = [line.strip() for line in f]
    with open(label_file_path, encoding="utf-8") as f:
        labels = [line.strip() for line in f]
    data = pd.DataFrame({"text": texts, "target": labels})
    return data

def preprocess_data(data):
    # Basic preprocessing steps like handling missing values 
    data = data.dropna()
    return data

def split_data(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, test_data

def vectorize_text(train_data, test_data, text_column):
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2)) # min_df=1 ensures at least 1 doc, avoids ValueError
    X_train_tfidf = vectorizer.fit_transform(train_data[text_column])
    X_test_tfidf = vectorizer.transform(test_data[text_column])
    return X_train_tfidf, X_test_tfidf, vectorizer

def train_xgboost_classifier(X_train_tfidf, y_train, num_classes):
    clf = xgb.XGBClassifier(objective='multi:softmax', num_class=num_classes, eval_metric='merror')
    clf.fit(X_train_tfidf, y_train)
    return clf

def evaluate_xgboost_classifier(model, X_test_tfidf, y_test,test_data):
    predictions = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, predictions)
    #print prediction test and prediction and real label
    print("Predictions vs Actuals:")
    for i, (pred, actual) in enumerate(zip(predictions, y_test)):
        print(f"Predicted: {pred}, Actual: {actual}")
        text = test_data['text'].iloc[i]  # positional index in test set
        print(f"actual text: {text}")
    print(f"Accuracy: {acc:.4f}")


def main():
    text_file_path = "/home/zografoum/my_class/data.txt"   # Each line: one document
    label_file_path = "/home/zografoum/my_class/labels.txt"  # Each line: one label (same order as texts)

    data = load_data(text_file_path, label_file_path)
    data = preprocess_data(data)

    # Encode labels as integers
    label_encoder = LabelEncoder()
    data["target"] = label_encoder.fit_transform(data["target"])

    train_data, test_data = split_data(data)
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(train_data, test_data, "text")
    y_train = train_data["target"].values
    y_test = test_data["target"].values

    num_classes = len(label_encoder.classes_)
    xgboost_model = train_xgboost_classifier(X_train_tfidf, y_train, num_classes)
    evaluate_xgboost_classifier(xgboost_model, X_test_tfidf, y_test,test_data)

if __name__ == "__main__":
    main()