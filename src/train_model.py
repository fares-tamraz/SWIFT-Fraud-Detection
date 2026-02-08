"""Train Random Forest model for SWIFT fraud detection."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


def load_and_prepare_data(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load CSV and convert to feature matrix X and labels y."""
    df = pd.read_csv(csv_path)

    # Expect columns: amount, currency, sender_country, receiver_country,
    # message_type, value_date (optional), is_fraud
    required = {"amount", "currency", "sender_country", "receiver_country", "is_fraud"}
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    from src.features import extract_features, FEATURE_COLUMNS, features_to_array

    rows = []
    for _, r in df.iterrows():
        parsed = {
            "amount": float(r.get("amount", 0) or 0),
            "currency": str(r.get("currency", "USD"))[:3],
            "sender_country": str(r.get("sender_country", "US"))[:2],
            "receiver_country": str(r.get("receiver_country", "US"))[:2],
            "message_type": str(r.get("message_type", "MT103")),
            "value_date": str(r.get("value_date", "230101")),
        }
        feat = extract_features(parsed)
        rows.append([feat[c] for c in FEATURE_COLUMNS])

    X = np.array(rows, dtype=np.float64)
    y = np.array(df["is_fraud"].astype(int), dtype=np.int32)

    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train SWIFT fraud detection model")
    parser.add_argument("--data", required=True, help="Path to training CSV")
    parser.add_argument("--output", default="models/rf_model.pkl", help="Output model path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--n-estimators", type=int, default=100, help="Random Forest trees")
    parser.add_argument("--max-depth", type=int, default=10, help="Max tree depth")
    args = parser.parse_args()

    X, y = load_and_prepare_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    main()
