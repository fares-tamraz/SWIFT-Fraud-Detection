"""Train the Random Forest fraud detection model."""

import sys
from pathlib import Path

# So "from src.xxx" works when running: python src/train.py
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

from src.features import add_high_risk_pair, FEATURE_COLUMNS

def train_fraud_detector(data_path="data/swift_messages.csv", model_path="models/fraud_detector.pkl"):
    df = pd.read_csv(data_path)
    df = add_high_risk_pair(df)

    le_sender = LabelEncoder()
    le_receiver = LabelEncoder()
    le_msg = LabelEncoder()
    df["sender_country_encoded"] = le_sender.fit_transform(df["sender_country"].astype(str))
    df["receiver_country_encoded"] = le_receiver.fit_transform(df["receiver_country"].astype(str))
    df["message_type_encoded"] = le_msg.fit_transform(df["message_type"].astype(str))

    X = df[FEATURE_COLUMNS]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "encoders": {"sender": le_sender, "receiver": le_receiver, "message_type": le_msg},
                "feature_columns": FEATURE_COLUMNS,
            },
            f,
        )
    print(f"Model saved to {model_path}")
    return model

if __name__ == "__main__":
    train_fraud_detector()