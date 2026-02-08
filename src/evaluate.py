"""
Evaluate the trained model on a separate test set (realistic: data the model never saw).
Use a different --seed when generating the test CSV so it's truly out-of-sample.

Usage:
  1. Generate test data (different seed):
     python src/generate_data.py -n 10000 --fraud-ratio 0.005 -o data/test_messages.csv --seed 123
  2. Evaluate:
     python src/evaluate.py data/test_messages.csv
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

from src.features import add_high_risk_pair, FEATURE_COLUMNS
from src.predict import load_model


def encode_test(df, encoders):
    """Encode categorical columns using saved encoders; unseen categories get -1 or 0."""
    df = df.copy()
    df["sender_country_encoded"] = df["sender_country"].astype(str).apply(
        lambda v: encoders["sender"].transform([v])[0] if v in encoders["sender"].classes_ else -1
    )
    df["receiver_country_encoded"] = df["receiver_country"].astype(str).apply(
        lambda v: encoders["receiver"].transform([v])[0] if v in encoders["receiver"].classes_ else -1
    )
    df["message_type_encoded"] = df["message_type"].astype(str).apply(
        lambda v: encoders["message_type"].transform([v])[0] if v in encoders["message_type"].classes_ else 0
    )
    return df


def evaluate(test_path="data/test_messages.csv", model_path="models/fraud_detector.pkl"):
    """Load test CSV and trained model; report metrics."""
    test_path = Path(test_path)
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test file not found: {test_path}\n"
            "Generate it with: python src/generate_data.py -n 10000 --fraud-ratio 0.005 -o data/test_messages.csv --seed 123"
        )
    df = pd.read_csv(test_path)
    df = add_high_risk_pair(df)
    model, encoders, _ = load_model(model_path)
    df = encode_test(df, encoders)
    X = df[FEATURE_COLUMNS]
    y = df["is_fraud"]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    print("=== Evaluation on held-out test set (unseen data) ===\n")
    print(f"Test size: {len(y)}, Fraud: {y.sum()}, Normal: {len(y) - y.sum()}\n")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("\nClassification report:")
    print(classification_report(y, y_pred, target_names=["Normal", "Fraud"]))
    print("Confusion matrix:\n", confusion_matrix(y, y_pred))
    try:
        print("\nROC-AUC:", roc_auc_score(y, y_proba))
    except ValueError:
        print("\nROC-AUC: (skipped - need both classes in test set)")
    return {"accuracy": accuracy_score(y, y_pred), "y_true": y, "y_pred": y_pred, "y_proba": y_proba}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Evaluate model on a separate test CSV")
    p.add_argument("test_csv", nargs="?", default="data/test_messages.csv", help="Path to test CSV")
    p.add_argument("--model", default="models/fraud_detector.pkl", help="Path to model pickle")
    args = p.parse_args()
    evaluate(args.test_csv, args.model)
