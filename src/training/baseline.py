"""Simple baselines to check the transformer earned its complexity: a
majority-class predictor (the floor -- anything below this is broken) and
TF-IDF + logistic regression (the real bar -- a transformer that can't beat
this isn't learning anything a bag-of-words model couldn't).
"""

import collections
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.training.finetune_data import load_examples

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "finetune"


def majority_class_accuracy(train_examples, val_examples):
    majority = collections.Counter(ex["label"] for ex in train_examples).most_common(1)[0][0]
    preds = [majority] * len(val_examples)
    truth = [ex["label"] for ex in val_examples]
    return accuracy_score(truth, preds), majority


def tfidf_logreg_accuracy(train_examples, val_examples):
    train_texts = [ex["text"] for ex in train_examples]
    train_labels = [ex["label"] for ex in train_examples]
    val_texts = [ex["text"] for ex in val_examples]
    val_labels = [ex["label"] for ex in val_examples]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_val)
    return accuracy_score(val_labels, preds)


def main():
    train_examples = load_examples(DATA_DIR / "train.jsonl")
    val_examples = load_examples(DATA_DIR / "val.jsonl")

    maj_acc, majority_label = majority_class_accuracy(train_examples, val_examples)
    print(f"majority-class baseline ('{majority_label}' always): {maj_acc:.3f}")

    tfidf_acc = tfidf_logreg_accuracy(train_examples, val_examples)
    print(f"TF-IDF + logistic regression: {tfidf_acc:.3f}")


if __name__ == "__main__":
    main()
