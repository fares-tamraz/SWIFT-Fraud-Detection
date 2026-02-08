# SWIFT Fraud Detection

Machine learning prototype that scores SWIFT-style transactions for fraud using a Random Forest classifier. Includes synthetic data generation, training, evaluation, and a **web app** for single and batch scoring with explainability and configurable threshold.

## Overview

- **Model:** Random Forest (scikit-learn), trained on 200k synthetic transactions with optional noise
- **Input:** Transaction fields (amount, sender/receiver country, timing, account age, velocity, etc.)
- **Output:** Fraud probability, binary flag at a chosen threshold, and human-readable reasons
- **Web app:** Flask UI — score one transaction or upload a CSV; adjust detection threshold; see why a transaction was flagged

## Features

- **Synthetic data generation** — Build training/test sets with configurable size, fraud ratio, and noise (overlapping normal/fraud patterns)
- **High-risk country pairs** — Feature and explainability for sender→receiver routes
- **Explainability** — Each prediction returns reasons (e.g. off-hours, high-risk pair, new account)
- **Threshold** — Tune strictness (e.g. 0.5 = balanced; 0.7 = fewer false alarms)
- **Batch CSV** — Upload a CSV; download a scored CSV with `fraud_probability`, `predicted_fraud`, and `reasons`
- **Validation** — Positive amounts, required fields, sensible ranges; clear errors in the UI and API

## Project structure

```
SWIFT-Fraud-Detection/
├── app.py                 # Flask app (single + batch API, serves web UI)
├── templates/             # HTML for the web app
├── static/                # CSS, JS
├── data/                  # Training/test CSVs (generated)
├── models/                # Saved model (fraud_detector.pkl)
├── src/
│   ├── generate_data.py   # Generate synthetic SWIFT-style data
│   ├── features.py        # Feature engineering, high_risk_country_pair
│   ├── train.py           # Train Random Forest, save model + encoders
│   ├── predict.py         # Predict + explainability
│   └── evaluate.py        # Evaluate on held-out test CSV
├── docs/
│   ├── REFERENCE.md       # Full reference (fraud knowledge, pipeline, concepts)
│   └── DEPLOY.md          # How to deploy the app to a public URL
├── requirements.txt
└── README.md
```

## Setup

```bash
cd SWIFT-Fraud-Detection
pip install -r requirements.txt
```

## Usage

### 1. Generate data

```bash
# Default: 1000 rows, 10% fraud
python src/generate_data.py

# Large training set (e.g. 200k rows, 0.5% fraud, 20% noise)
python src/generate_data.py -n 200000 --fraud-ratio 0.005 --noise 0.2 -o data/swift_messages.csv

# Separate test set (different seed)
python src/generate_data.py -n 10000 --fraud-ratio 0.005 -o data/test_messages.csv --seed 123 --noise 0.3
```

### 2. Train

```bash
python src/train.py
```

Uses `data/swift_messages.csv` by default and writes `models/fraud_detector.pkl` (model + encoders).

### 3. Evaluate (optional)

```bash
python src/evaluate.py data/test_messages.csv
```

Reports accuracy, precision/recall, confusion matrix on the test set.

### 4. Run the web app locally

```bash
python app.py
```

Open **http://127.0.0.1:5000**. You can score a single transaction (form), adjust the threshold (slider), or upload a CSV for batch scoring and download the results.

### 5. Deploy so others can use a link

See **[docs/DEPLOY.md](docs/DEPLOY.md)** for deploying to Render (or Railway / Hugging Face) so anyone can use the app via a public URL without installing anything.

## Data schema

Training and batch CSVs use these columns:

- **Required:** `amount`, `sender_country`, `receiver_country`
- **Optional (with defaults):** `hour_of_day`, `day_of_week`, `account_age_days`, `transaction_velocity`, `ip_country_matches_sender`, `message_has_typos`, `message_type`
- **Label:** `is_fraud` (0 or 1) for training data

The web app’s single-transaction form uses dropdowns for countries (the set the model was trained on). Batch CSV can use any country strings; unknown values are encoded with a fallback.

## License

MIT
