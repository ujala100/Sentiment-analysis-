"""
Quick predictor - loads the already-trained model and classifies your own tweets.
Usage:
    python predict.py "I love this!" "This is terrible."
"""
import sys
import joblib
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_SYMBOL_RE = re.compile(r"#")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s']")
MULTI_SPACE_RE = re.compile(r"\s+")

def clean_tweet(text):
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_SYMBOL_RE.sub("", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text

def main():
    tweets = sys.argv[1:]
    if not tweets:
        tweets = [input("Enter a tweet: ")]

    model = joblib.load("sentiment_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")

    cleaned = [clean_tweet(t) for t in tweets]
    X = vectorizer.transform(cleaned)
    preds = model.predict(X)

    for tweet, pred in zip(tweets, preds):
        label = "POSITIVE" if pred == 1 else "NEGATIVE"
        print(f"[{label}] {tweet}")

if __name__ == "__main__":
    main()
