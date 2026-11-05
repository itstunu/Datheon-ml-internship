# Task4_NLP_Sentiment/task4_nlp_sentiment.py
"""Task 4: TF-IDF sentiment classification on a synthetic review dataset.

Pipeline: generate 200 reviews -> preprocess -> TF-IDF -> train Naive Bayes
and Logistic Regression -> evaluate -> confusion matrix -> predict helper.
"""

import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-GUI backend.
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

OUTPUT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Hardcoded building blocks for the synthetic dataset.
# Reviews are random (seeded) combinations: opener + core + closer.
# ---------------------------------------------------------------------------
POS_OPENERS = [
    "I love this product", "Absolutely love this", "What a fantastic purchase",
    "Really happy with this", "This is amazing", "Highly recommend this",
    "Great quality overall", "Excellent value", "Totally satisfied", "Wonderful experience",
    "Best purchase I have made", "Truly impressive",
]
POS_CORES = [
    "the quality is excellent", "it works perfectly", "very easy to use",
    "it exceeded my expectations", "the design is beautiful", "great value for the price",
    "delivery was fast and the packaging was great", "customer service was friendly and helpful",
    "it feels sturdy and well made", "performance is outstanding",
    "everything about it is wonderful", "the battery lasts forever and I love it",
]
POS_CLOSERS = [
    "I would buy it again.", "Five stars!", "Definitely worth every penny.",
    "My family loves it too.", "Will recommend to all my friends.",
    "Couldn't be happier.", "Superb!", "Great job by the team.",
]

NEG_OPENERS = [
    "Terrible quality", "Absolutely hate this", "What a disappointing purchase",
    "Really unhappy with this", "This is awful", "Do not buy this",
    "Poor quality overall", "Complete waste of money", "Totally dissatisfied", "Horrible experience",
    "Worst purchase I have made", "Truly disappointing",
]
NEG_CORES = [
    "the quality is bad", "it broke after two days", "very hard to use",
    "it failed to meet my expectations", "the design is ugly", "terrible value for the price",
    "delivery was slow and the packaging was damaged", "customer service was rude and useless",
    "it feels cheap and flimsy", "performance is the worst ever",
    "everything about it is awful", "the battery dies quickly and I hate it",
]
NEG_CLOSERS = [
    "I want a refund.", "One star.", "Definitely not worth the money.",
    "My family hates it too.", "Will warn all my friends.",
    "Couldn't be more disappointed.", "Garbage!", "Shame on the team.",
]

# Populated by main(); read by predict_sentiment().
_STATE = {}


def generate_dataset(n_per_class=100, seed=RANDOM_STATE):
    """Build a balanced synthetic dataset of unique reviews.

    Returns a DataFrame with columns 'text' and 'label' (1 = positive, 0 = negative),
    shuffled, containing n_per_class reviews of each class.
    """
    rng = random.Random(seed)

    def sample(openers, cores, closers):
        combos = [f"{o}, {c}. {cl}" for o in openers for c in cores for cl in closers]
        return rng.sample(combos, n_per_class)

    pos = sample(POS_OPENERS, POS_CORES, POS_CLOSERS)
    neg = sample(NEG_OPENERS, NEG_CORES, NEG_CLOSERS)
    df = pd.DataFrame({"text": pos + neg, "label": [1] * len(pos) + [0] * len(neg)})
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_stopwords():
    """Return the English stopword set from NLTK.

    Downloads the corpus if missing. If NLTK data cannot be downloaded (e.g. no
    internet), falls back to scikit-learn's built-in English stopword list.
    """
    from nltk.corpus import stopwords

    try:
        return set(stopwords.words("english"))
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))
        except Exception:
            print("[warn] NLTK stopwords unavailable; using scikit-learn's stopword list.")
            return set(ENGLISH_STOP_WORDS)


def preprocess_text(text, stop_words):
    """Lowercase, strip punctuation/digits via regex, and remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in stop_words]
    return " ".join(tokens)


def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Save a confusion-matrix heatmap to the given filename."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Negative", "Positive"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)


def predict_sentiment(text_input):
    """Predict and print the sentiment of a custom string.

    Uses the trained TfidfVectorizer and Naive Bayes model stored by main().
    Returns the label string ('Positive' or 'Negative').
    """
    if not _STATE:
        raise RuntimeError("Models are not trained yet. Run main() first.")
    cleaned = preprocess_text(text_input, _STATE["stop_words"])
    vec = _STATE["vectorizer"].transform([cleaned])
    model = _STATE["model"]
    pred = int(model.predict(vec)[0])
    confidence = float(model.predict_proba(vec)[0][pred])
    label = "Positive" if pred == 1 else "Negative"
    print(f'Input: "{text_input}"\n  -> Sentiment: {label} (confidence: {confidence:.2%})')
    return label


def main():
    """Train and evaluate both models, then register the NB model for predictions."""
    df = generate_dataset()
    print(f"Generated {len(df)} reviews "
          f"({(df.label == 1).sum()} positive, {(df.label == 0).sum()} negative)\n")

    stop_words = load_stopwords()
    df["clean"] = df["text"].apply(lambda t: preprocess_text(t, stop_words))

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train_txt)
    X_test = vectorizer.transform(X_test_txt)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    }

    print("=" * 50)
    print("MODEL PERFORMANCE (Test Set)")
    print("=" * 50)
    print(f"{'Model':<28}{'Accuracy':>10}{'F1':>10}")
    print("-" * 48)

    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds[name] = model.predict(X_test)
        acc = accuracy_score(y_test, preds[name])
        f1 = f1_score(y_test, preds[name])
        print(f"{name:<28}{acc:>10.4f}{f1:>10.4f}")

    plot_confusion_matrix(y_test, preds["Multinomial Naive Bayes"],
                          "Confusion Matrix - Multinomial Naive Bayes",
                          "nlp_confusion_matrix.png")
    print("\nSaved: nlp_confusion_matrix.png\n")

    _STATE.update(vectorizer=vectorizer, model=models["Multinomial Naive Bayes"],
                  stop_words=stop_words)


if __name__ == "__main__":
    main()
    print("=" * 50)
    print("DEMO PREDICTIONS")
    print("=" * 50)
    predict_sentiment("I absolutely love this!")
    predict_sentiment("This is the worst thing ever.")