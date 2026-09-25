"""
Twitter Sentiment Analysis - Machine Learning Project
=======================================================
Dataset: Twitter Sentiment Analysis Training Corpus (Ibrahim Naji, 2012)
Source : https://raw.githubusercontent.com/cblancac/SentimentAnalysisBert/main/data/
Format : label \t tweet_text   (label: 0 = negative, 1 = positive)

Pipeline:
  1. Load data (tab-separated, label + text)
  2. Clean tweets (remove URLs, mentions, hashtags symbols, punctuation, lowercase)
  3. Vectorize text with TF-IDF
  4. Train + compare 3 classifiers: Logistic Regression, Naive Bayes, Linear SVM
  5. Evaluate (accuracy, precision/recall/F1, confusion matrix)
  6. Save the best model + vectorizer for reuse
  7. Predict on custom example tweets
"""

import re
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

TRAIN_PATH = "data/train_150k.txt"
TEST_PATH = "data/test_62k.txt"

# Use a subset for speed; bump these up for a more thorough run.
TRAIN_SAMPLE_SIZE = 40000
TEST_SAMPLE_SIZE = 8000


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data(path, sample_size=None):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            label, text = line.split("\t", 1)
            try:
                label = int(label)
            except ValueError:
                continue
            rows.append((label, text))
    df = pd.DataFrame(rows, columns=["label", "text"])
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Clean tweets
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_SYMBOL_RE = re.compile(r"#")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s']")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_tweet(text):
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_SYMBOL_RE.sub("", text)   # keep the word, drop the '#'
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def prepare(df):
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_tweet)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    train_df = load_data(TRAIN_PATH, TRAIN_SAMPLE_SIZE)
    test_df = load_data(TEST_PATH, TEST_SAMPLE_SIZE)
    print(f"  train: {len(train_df)} rows | test: {len(test_df)} rows")
    print("  label distribution (train):\n", train_df["label"].value_counts())

    print("\nCleaning tweets...")
    train_df = prepare(train_df)
    test_df = prepare(test_df)

    X_train_text, y_train = train_df["clean_text"], train_df["label"]
    X_test_text, y_test = test_df["clean_text"], test_df["label"]

    # ---------------------------------------------------------------
    # 3. TF-IDF vectorization
    # ---------------------------------------------------------------
    print("\nVectorizing (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # ---------------------------------------------------------------
    # 4. Train + compare models
    # ---------------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(),
    }

    results = {}
    best_model_name, best_acc, best_model = None, -1, None

    for name, model in models.items():
        print(f"\nTraining: {name}")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"  train time: {train_time:.1f}s | test accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

        if acc > best_acc:
            best_acc, best_model_name, best_model = acc, name, model

    print("\n" + "=" * 50)
    print("Model comparison (test accuracy):")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = "  <-- best" if name == best_model_name else ""
        print(f"  {name:<28s} {acc:.4f}{marker}")
    print("=" * 50)

    # Confusion matrix for the best model
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    print(f"\nConfusion matrix for {best_model_name}:")
    print("             pred_neg  pred_pos")
    print(f"actual_neg   {cm[0][0]:<9d} {cm[0][1]:<9d}")
    print(f"actual_pos   {cm[1][0]:<9d} {cm[1][1]:<9d}")

    # ---------------------------------------------------------------
    # 6. Save best model + vectorizer
    # ---------------------------------------------------------------
    joblib.dump(best_model, "sentiment_model.joblib")
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
    print(f"\nSaved best model ({best_model_name}) -> sentiment_model.joblib")
    print("Saved vectorizer -> tfidf_vectorizer.joblib")

    # ---------------------------------------------------------------
    # 7. Try it on custom tweets
    # ---------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Predictions on custom example tweets:")
    sample_tweets = [
        "I absolutely love this new phone, best purchase ever!",
        "This is the worst customer service I've ever experienced.",
        "Can't wait for the weekend, going to be amazing!",
        "I'm so frustrated, my flight got cancelled again.",
        "It's an okay movie, nothing special but not bad either.",
    ]
    cleaned = [clean_tweet(t) for t in sample_tweets]
    vec = vectorizer.transform(cleaned)
    preds = best_model.predict(vec)
    for tweet, pred in zip(sample_tweets, preds):
        label = "POSITIVE" if pred == 1 else "NEGATIVE"
        print(f"  [{label:8s}] {tweet}")


if __name__ == "__main__":
    main()
