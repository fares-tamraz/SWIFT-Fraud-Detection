"""Feature engineering for SWIFT fraud detection."""

import pandas as pd

# Country pairs often linked to fraud (sender -> receiver).
# We'll flag these so the model can learn "this route is riskier."
HIGH_RISK_PAIRS = frozenset({
    ("USA", "Nigeria"), ("UK", "Nigeria"), ("Germany", "Russia"),
    ("USA", "Iran"), ("France", "Panama"), ("Japan", "Cayman Islands"),
    ("Canada", "Nigeria"), ("Australia", "Cayman Islands"),
})

def add_high_risk_pair(df):
    """Add a column: 1 if (sender_country, receiver_country) is high-risk, else 0."""
    pairs = list(zip(df["sender_country"], df["receiver_country"]))
    df = df.copy()
    df["high_risk_country_pair"] = [1 if p in HIGH_RISK_PAIRS else 0 for p in pairs]
    return df

# Column names the model will use (after we encode categories in train.py).
# Must match the order we build the training matrix in.
FEATURE_COLUMNS = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "account_age_days",
    "transaction_velocity",
    "ip_country_matches_sender",
    "message_has_typos",
    "is_round_number",
    "sender_country_encoded",
    "receiver_country_encoded",
    "message_type_encoded",
    "high_risk_country_pair",
]