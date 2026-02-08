# SWIFT Fraud Detection

Machine learning system that analyzes SWIFT messages and uses a Random Forest classifier to predict whether a transaction is likely fraudulent.

## Overview

- **Model:** Random Forest (scikit-learn)
- **Input:** SWIFT-style transaction messages
- **Output:** Fraud probability / binary classification (fraud / not fraud)
- **Scope:** Local inference for now; Flask web app planned

## Project Structure

```
SWIFT-Fraud-Detection/
├── data/               # Training data (CSV with SWIFT-like fields)
├── models/             # Saved Random Forest model
├── src/
│   ├── swift_parser.py # Parse SWIFT message format
│   ├── features.py     # Feature engineering
│   ├── train.py        # Train the model
│   └── predict.py      # Run predictions
├── app/                # (Future) Flask web app
├── requirements.txt
└── README.md
```

## Setup

```bash
cd SWIFT-Fraud-Detection
pip install -r requirements.txt
```

## Usage

### 1. Prepare data

Place training data in `data/` as CSV with columns such as:
`message_type`, `amount`, `currency`, `sender_country`, `receiver_country`, `timestamp`, `sender_bic`, `receiver_bic`, `is_fraud`, etc.

See `data/sample_data.csv` for the expected format.

### 2. Train the model

```bash
python -m src.train --data data/train.csv --output models/rf_model.pkl
```

### 3. Predict on a single message

```bash
python -m src.predict --model models/rf_model.pkl --message "MT103..."
```

Or from Python:

```python
from src.predict import predict_fraud

result = predict_fraud("MT103:...")
print(result)  # {"fraud_probability": 0.12, "is_fraud": False}
```

## SWIFT Message Format (Reference)

SWIFT messages use MT (Message Type) codes, e.g.:
- **MT103** – Single Customer Credit Transfer
- **MT202** – Bank Transfer
- **MT700** – Documentary Credit

Key fields for fraud detection: amount, currency, sender/receiver BICs, countries, timestamps, and message type.

## License

MIT
