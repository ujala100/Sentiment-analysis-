# Twitter Sentiment Analysis (ML Project)

A complete, ready-to-run sentiment analysis project using classic ML (TF-IDF + Logistic Regression / Naive Bayes / Linear SVM).

## Dataset
`data/train_150k.txt` and `data/test_62k.txt` — Twitter Sentiment Analysis Training Corpus
(Ibrahim Naji, 2012), ~150k + 62k labeled tweets. Format: `label<TAB>tweet_text`
(0 = negative, 1 = positive). Already included — no download needed.

## Setup
```bash
pip install -r requirements.txt
```

## Run training (from scratch)
```bash
python sentiment_analysis.py
```
This will:
- Load and clean the tweets
- Vectorize with TF-IDF
- Train & compare Logistic Regression, Naive Bayes, and Linear SVM
- Print accuracy, precision/recall/F1, and a confusion matrix
- Save the best model to `sentiment_model.joblib` and the vectorizer to
  `tfidf_vectorizer.joblib` (already included pre-trained, so this step is optional)
- Run predictions on a few example tweets

## Predict on your own tweets (using the pre-trained model)
```bash
python predict.py "I absolutely love this!" "This ruined my whole day."
```
or run it interactively:
```bash
python predict.py
```

## Files
- `sentiment_analysis.py` — full training/evaluation pipeline
- `predict.py` — loads the saved model to classify new text instantly
- `sentiment_model.joblib` — pre-trained Logistic Regression model
- `tfidf_vectorizer.joblib` — pre-trained TF-IDF vectorizer
- `data/` — the dataset (train + test)
- `requirements.txt` — Python dependencies

## Adjusting sample size
Open `sentiment_analysis.py` and change `TRAIN_SAMPLE_SIZE` / `TEST_SAMPLE_SIZE`
(currently 40,000 / 8,000) to use more data for higher accuracy at the cost of
longer training time.

## Web App (no command line needed)
`sentiment_app.html` — a self-contained, offline-capable web page. The trained
model's weights are embedded directly in the file (as JavaScript), so it needs
no server, no Python, and no internet connection.

**To run it:** just double-click `sentiment_app.html` (or open it in any browser).
Type a tweet, click "Analyze Sentiment", and see the live prediction with
confidence score and the words that influenced it.

**Live hosted version:** https://claude.ai/artifact/SQojFq3cx1NSTGQZNorsjz

**To deploy it yourself for free (optional):**
- GitHub Pages: create a repo, upload `sentiment_app.html` as `index.html`, enable
  Pages in repo Settings → Pages → deploy from main branch.
- Netlify Drop: go to app.netlify.com/drop and drag the file in — instant public link.
- Vercel: `vercel deploy` after adding the file to a project folder.
