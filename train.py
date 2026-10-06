"""Train, compare and save the best spam classifier."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data_loader import load_data


def make_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english",
                                  ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", clf),
    ])


def main():
    df = load_data()
    df["y"] = (df["label"] == "spam").astype(int)
    print(f"Dataset: {len(df)} messages | spam ratio: {df['y'].mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["y"], test_size=0.2, stratify=df["y"], random_state=42)

    models = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced"),
    }

    best_name, best_model, best_f1 = None, None, -1
    for name, clf in models.items():
        pipe = make_pipeline(clf).fit(X_train, y_train)
        f1 = f1_score(y_test, pipe.predict(X_test))
        print(f"{name:<22} F1 (spam) = {f1:.4f}")
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, pipe, f1

    print(f"\nBest model: {best_name}\n")
    preds = best_model.predict(X_test)
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))

    ConfusionMatrixDisplay.from_predictions(
        y_test, preds, display_labels=["ham", "spam"], cmap="Blues")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")

    joblib.dump(best_model, "spam_model.joblib")
    print("Saved model -> spam_model.joblib, plot -> confusion_matrix.png")


if __name__ == "__main__":
    main()
