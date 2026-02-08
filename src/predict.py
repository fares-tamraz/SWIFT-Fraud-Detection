"""Predict fraud probability for a single SWIFT transaction."""

import sys
from pathlib import Path
import pickle
import pandas as pd

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.features import FEATURE_COLUMNS, HIGH_RISK_PAIRS


def load_model(model_path="models/fraud_detector.pkl"):
    """Load the saved model, encoders, and feature column names."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["encoders"], data["feature_columns"]

def transaction_to_row(tx, encoders):
    """Turn one transaction dict into a feature row (same order as FEATURE_COLUMNS)."""
    sender = str(tx.get("sender_country", ""))
    receiver = str(tx.get("receiver_country", ""))
    msg_type = str(tx.get("message_type", "pacs.008"))
    amount = float(tx.get("amount", 0))
    if amount <= 0:
        amount = 1.0

    try:
        sender_enc = encoders["sender"].transform([sender])[0]
    except ValueError:
        sender_enc = -1
    try:
        receiver_enc = encoders["receiver"].transform([receiver])[0]
    except ValueError:
        receiver_enc = -1
    try:
        msg_enc = encoders["message_type"].transform([msg_type])[0]
    except ValueError:
        msg_enc = 0

    high_risk = 1 if (sender, receiver) in HIGH_RISK_PAIRS else 0
    is_round = 1 if amount % 10000 == 0 else 0

    row = {
        "amount": amount,
        "hour_of_day": int(tx.get("hour_of_day", 12)),
        "day_of_week": int(tx.get("day_of_week", 2)),
        "account_age_days": int(tx.get("account_age_days", 365)),
        "transaction_velocity": int(tx.get("transaction_velocity", 1)),
        "ip_country_matches_sender": int(tx.get("ip_country_matches_sender", 1)),
        "message_has_typos": int(tx.get("message_has_typos", 0)),
        "is_round_number": is_round,
        "sender_country_encoded": sender_enc,
        "receiver_country_encoded": receiver_enc,
        "message_type_encoded": msg_enc,
        "high_risk_country_pair": high_risk,
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]

def get_explanation_reasons(tx):
    """Return a list of human-readable reasons why this transaction might be risky (for explainability)."""
    reasons = []
    sender = str(tx.get("sender_country", "")).strip()
    receiver = str(tx.get("receiver_country", "")).strip()
    if (sender, receiver) in HIGH_RISK_PAIRS:
        reasons.append("High-risk country pair (sender → receiver route is flagged)")
    hour = int(tx.get("hour_of_day", 12))
    if hour in (0, 1, 2, 3, 4, 5, 23):
        reasons.append("Off-hours transaction (outside typical 9am–5pm)")
    day = int(tx.get("day_of_week", 2))
    if day >= 5:
        reasons.append("Weekend transaction (higher risk period)")
    age = int(tx.get("account_age_days", 365))
    if age < 90:
        reasons.append("New or young account (under 90 days)")
    vel = int(tx.get("transaction_velocity", 1))
    if vel > 10:
        reasons.append("High transaction velocity (unusual activity volume)")
    if int(tx.get("ip_country_matches_sender", 1)) == 0:
        reasons.append("IP country does not match sender (suspicious login)")
    if int(tx.get("message_has_typos", 0)) == 1:
        reasons.append("Message contains typos (common in fraud)")
    amount = float(tx.get("amount", 0))
    if amount > 100_000:
        reasons.append("Large amount (above 100k)")
    if amount > 50_000 and amount % 10000 != 0:
        reasons.append("Amount just under round number (possible threshold avoidance)")
    return reasons if reasons else ["No strong risk indicators identified"]


def predict(tx, model_path="models/fraud_detector.pkl", threshold=0.5):
    """Return fraud probability (0–1), binary is_fraud at given threshold, and reasons."""
    model, encoders, _ = load_model(model_path)
    X = transaction_to_row(tx, encoders)
    proba = model.predict_proba(X)[0, 1]   # probability of class 1 (fraud)
    is_fraud = 1 if proba >= threshold else 0
    reasons = get_explanation_reasons(tx)
    return {"fraud_probability": float(proba), "is_fraud": is_fraud, "threshold_used": threshold, "reasons": reasons}


if __name__ == "__main__":
    model, encoders, _ = load_model()

    normal = {
        "amount": 25000,
        "hour_of_day": 14,
        "day_of_week": 2,
        "sender_country": "USA",
        "receiver_country": "Canada",
        "account_age_days": 500,
        "transaction_velocity": 3,
        "ip_country_matches_sender": 1,
        "message_has_typos": 0,
        "message_type": "pacs.008",
    }
    suspicious = {
        "amount": 999999,
        "hour_of_day": 3,
        "day_of_week": 5,
        "sender_country": "USA",
        "receiver_country": "Nigeria",
        "account_age_days": 5,
        "transaction_velocity": 25,
        "ip_country_matches_sender": 0,
        "message_has_typos": 1,
        "message_type": "pacs.008",
    }

    for name, tx in [("Normal", normal), ("Suspicious", suspicious)]:
        out = predict(tx)
        print(f"{name}: fraud_prob={out['fraud_probability']:.2%}, is_fraud={out['is_fraud']}")