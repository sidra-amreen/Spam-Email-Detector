"""Classify messages with the trained model.

Usage:
    python predict.py "You won a free prize, click now!"
    python predict.py            # interactive mode
"""
import sys
import joblib

model = joblib.load("spam_model.joblib")


def classify(text: str):
    label = model.predict([text])[0]
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        conf = model.predict_proba([text])[0][1]
    else:  # LinearSVC: squash decision score into 0-1
        import math
        conf = 1 / (1 + math.exp(-model.decision_function([text])[0]))
    return ("SPAM" if label == 1 else "HAM"), conf


if __name__ == "__main__":
    if len(sys.argv) > 1:
        label, conf = classify(" ".join(sys.argv[1:]))
        print(f"{label} (spam score: {conf:.2%})")
    else:
        print("Type a message (empty line to quit):")
        while (msg := input("> ").strip()):
            label, conf = classify(msg)
            print(f"  -> {label} (spam score: {conf:.2%})")
