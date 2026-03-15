"""
Flask web app for SWIFT fraud detection: single transaction, batch CSV, explainability, threshold.
Run from project root: python app.py   or   flask --app app run
"""

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import pandas as pd

from src.predict import load_model, transaction_to_row, get_explanation_reasons
from src.features import add_high_risk_pair, FEATURE_COLUMNS

app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
def _cors_origins(origin):
    if not origin:
        return True
    if origin == "http://localhost:3000":
        return True
    if origin.startswith("https://") and origin.endswith(".vercel.app"):
        return True
    return False


CORS(app, origins=_cors_origins)

# Load model once at startup
MODEL_PATH = ROOT / "models" / "fraud_detector.pkl"
_model, _encoders, _ = load_model(str(MODEL_PATH)) if MODEL_PATH.exists() else (None, None, None)


REQUIRED_SINGLE = {"amount", "sender_country", "receiver_country"}
OPTIONAL_DEFAULTS = {
    "hour_of_day": 12, "day_of_week": 2, "account_age_days": 365,
    "transaction_velocity": 1, "ip_country_matches_sender": 1, "message_has_typos": 0,
    "message_type": "pacs.008",
}


def validate_single(data):
    """Validate single transaction JSON. Returns (None, None) if OK, else (error_message, 400)."""
    if not isinstance(data, dict):
        return "Body must be a JSON object", 400
    missing = REQUIRED_SINGLE - set(data.keys())
    if missing:
        return f"Missing required fields: {', '.join(sorted(missing))}", 400
    try:
        amount = float(data.get("amount", 0))
    except (TypeError, ValueError):
        return "amount must be a number", 400
    if amount <= 0:
        return "amount must be positive", 400
    for key in ("hour_of_day", "day_of_week", "account_age_days", "transaction_velocity"):
        if key in data and data[key] is not None:
            try:
                v = int(data[key])
                if key == "hour_of_day" and not (0 <= v <= 23):
                    return "hour_of_day must be 0–23", 400
                if key == "day_of_week" and not (0 <= v <= 6):
                    return "day_of_week must be 0–6", 400
                if key == "account_age_days" and v < 0:
                    return "account_age_days must be non-negative", 400
                if key == "transaction_velocity" and v < 0:
                    return "transaction_velocity must be non-negative", 400
            except (TypeError, ValueError):
                return f"{key} must be an integer", 400
    return None, None


def build_tx(data):
    """Build full transaction dict with defaults for optional fields."""
    tx = dict(data)
    for k, v in OPTIONAL_DEFAULTS.items():
        if k not in tx or tx[k] is None or tx[k] == "":
            tx[k] = v
    return tx


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if _model is None:
        return jsonify({"error": "Model not loaded. Train first: python src/train.py"}), 503
    err, status = validate_single(request.get_json(silent=True))
    if err:
        return jsonify({"error": err}), status
    data = request.get_json()
    threshold = float(request.args.get("threshold", 0.5))
    if not (0.01 <= threshold <= 0.99):
        threshold = 0.5
    tx = build_tx(data)
    import src.predict as pred
    out = pred.predict(tx, model_path=str(MODEL_PATH), threshold=threshold)
    return jsonify(out)


def validate_batch_df(df):
    """Validate batch CSV. Returns (None, None) if OK, else (error_message, 400)."""
    required = {"amount", "sender_country", "receiver_country"}
    cols = set(df.columns)
    missing = required - cols
    if missing:
        return f"CSV must have columns: {', '.join(sorted(required))}. Missing: {', '.join(sorted(missing))}", 400
    if df["amount"].isna().any():
        return "amount cannot be missing in any row", 400
    try:
        amt = pd.to_numeric(df["amount"], errors="coerce")
    except Exception:
        return "amount must be numeric", 400
    if (amt <= 0).any():
        return "amount must be positive in every row", 400
    if df["sender_country"].isna().any() or df["receiver_country"].isna().any():
        return "sender_country and receiver_country cannot be missing", 400
    return None, None


@app.route("/api/batch", methods=["POST"])
def api_batch():
    if _model is None:
        return jsonify({"error": "Model not loaded. Train first: python src/train.py"}), 503
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file uploaded. Use form field 'file' with a CSV."}), 400
    threshold = float(request.form.get("threshold", 0.5))
    if not (0.01 <= threshold <= 0.99):
        threshold = 0.5
    try:
        df = pd.read_csv(io.BytesIO(f.read()))
        df.columns = df.columns.str.strip()
    except Exception as e:
        return jsonify({"error": f"Invalid CSV: {e}"}), 400
    err, status = validate_batch_df(df)
    if err:
        return jsonify({"error": err}), status
    for k, v in OPTIONAL_DEFAULTS.items():
        if k not in df.columns:
            df[k] = v
    out_df = df.copy()
    df = add_high_risk_pair(df)
    from src.evaluate import encode_test
    df = encode_test(df, _encoders)
    X = df[FEATURE_COLUMNS]
    proba = _model.predict_proba(X)[:, 1]
    pred_fraud = (proba >= threshold).astype(int)
    out_df["fraud_probability"] = proba
    out_df["predicted_fraud"] = pred_fraud
    reasons_list = []
    for _, row in out_df.iterrows():
        tx = row.to_dict()
        for k, v in OPTIONAL_DEFAULTS.items():
            if k not in tx or pd.isna(tx.get(k)):
                tx[k] = v
        reasons_list.append("; ".join(get_explanation_reasons(tx)))
    out_df["reasons"] = reasons_list
    buf = io.BytesIO()
    out_df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        buf,
        mimetype="text/csv",
        as_attachment=True,
        download_name="fraud_scores.csv",
    )


# --- Stub endpoints for fraud-ui (mock data until real implementation) ---

MOCK_EXPLAIN = {
    "score": 0.87,
    "verdict": "fraud",
    "features": [
        {"name": "transaction_velocity", "humanLabel": "Transaction velocity", "value": 8, "normalizedValue": 0.72, "contribution": 0.31, "severity": "critical"},
        {"name": "account_age_days", "humanLabel": "Account age (days)", "value": 3, "normalizedValue": 0.02, "contribution": 0.22, "severity": "high"},
        {"name": "amount", "humanLabel": "Amount", "value": 187500, "normalizedValue": 0.65, "contribution": 0.18, "severity": "high"},
        {"name": "hour_of_day", "humanLabel": "Hour of day", "value": 14, "normalizedValue": 0.58, "contribution": 0.08, "severity": "medium"},
        {"name": "high_risk_country_pair", "humanLabel": "High-risk country pair", "value": 1, "normalizedValue": 1.0, "contribution": 0.08, "severity": "medium"},
    ],
}

MOCK_GRAPH_NODES = [
    {"id": "n1", "label": "Acme Corp", "type": "corporate", "riskLevel": "clean", "country": "USA", "isSelected": False, "isFocused": False},
    {"id": "n2", "label": "First Bank", "type": "bank", "riskLevel": "clean", "country": "USA", "isSelected": False, "isFocused": False},
    {"id": "n3", "label": "Offshore LLC", "type": "shell", "riskLevel": "fraud", "country": "Cayman Islands", "isSelected": False, "isFocused": False},
    {"id": "n4", "label": "John Doe", "type": "individual", "riskLevel": "suspicious", "country": "UK", "isSelected": False, "isFocused": False},
    {"id": "n5", "label": "Euro Bank", "type": "bank", "riskLevel": "clean", "country": "Germany", "isSelected": False, "isFocused": False},
    {"id": "n6", "label": "Unknown Entity", "type": "unknown", "riskLevel": "suspicious", "country": "Panama", "isSelected": False, "isFocused": False},
    {"id": "n7", "label": "Payments Inc", "type": "corporate", "riskLevel": "clean", "country": "Canada", "isSelected": False, "isFocused": False},
    {"id": "n8", "label": "Trust Fund", "type": "individual", "riskLevel": "clean", "country": "USA", "isSelected": False, "isFocused": False},
]

MOCK_GRAPH_EDGES = [
    {"source": "n1", "target": "n2", "amount": 50000, "timestamp": "2024-01-15T10:30:00Z", "riskLevel": "clean", "messageType": "pacs.008"},
    {"source": "n2", "target": "n3", "amount": 45000, "timestamp": "2024-01-15T10:35:00Z", "riskLevel": "fraud", "messageType": "pacs.008"},
    {"source": "n1", "target": "n4", "amount": 12000, "timestamp": "2024-01-14T14:00:00Z", "riskLevel": "suspicious", "messageType": "pacs.009"},
    {"source": "n4", "target": "n5", "amount": 8000, "timestamp": "2024-01-14T16:00:00Z", "riskLevel": "clean", "messageType": "pacs.008"},
    {"source": "n5", "target": "n6", "amount": 75000, "timestamp": "2024-01-13T09:00:00Z", "riskLevel": "suspicious", "messageType": "pacs.004"},
    {"source": "n7", "target": "n1", "amount": 25000, "timestamp": "2024-01-12T11:00:00Z", "riskLevel": "clean", "messageType": "pacs.008"},
    {"source": "n8", "target": "n2", "amount": 100000, "timestamp": "2024-01-11T15:30:00Z", "riskLevel": "clean", "messageType": "pacs.008"},
    {"source": "n3", "target": "n6", "amount": 30000, "timestamp": "2024-01-15T11:00:00Z", "riskLevel": "fraud", "messageType": "pacs.009"},
    {"source": "n2", "target": "n7", "amount": 15000, "timestamp": "2024-01-10T08:00:00Z", "riskLevel": "clean", "messageType": "pacs.008"},
    {"source": "n6", "target": "n3", "amount": 20000, "timestamp": "2024-01-14T17:00:00Z", "riskLevel": "fraud", "messageType": "pacs.008"},
]

MOCK_COUNTERFACTUAL = {
    "originalScore": 0.87,
    "achievedScore": 0.38,
    "changes": {
        "accountAgeDays": {"from": 3, "to": 90, "impact": 0.22},
        "transactionVelocity": {"from": 8, "to": 2, "impact": 0.31},
        "hourOfDay": {"from": 14, "to": 10, "impact": 0.08},
    },
}


@app.route("/api/explain", methods=["POST"])
def api_explain():
    return jsonify(MOCK_EXPLAIN)


@app.route("/api/graph", methods=["POST"])
def api_graph():
    return jsonify({"nodes": MOCK_GRAPH_NODES, "edges": MOCK_GRAPH_EDGES})


@app.route("/api/counterfactual", methods=["POST"])
def api_counterfactual():
    return jsonify(MOCK_COUNTERFACTUAL)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
