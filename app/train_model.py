# app/train_model.py
import pandas as pd
import joblib
import json
from app.fraud_model import HybridFraudDetectionModel

# ==========================
# Config
# ==========================
DATA_PATH = "dataset/paysim.csv"    # replace with your dataset
MODEL_DIR = "model/"
MODEL_PATH = MODEL_DIR + "hybrid_fraud_detection_model.pkl"
XGB_PATH = MODEL_DIR + "xgb_model.json"
META_PATH = MODEL_DIR + "metadata.json"

THRESHOLD = 0.23  # replace with F1-optimal threshold

# ==========================
# Training
# ==========================
def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Required features
    feature_cols = [
        "step","type","amount","nameOrig","oldbalanceOrg",
        "newbalanceOrig","nameDest","oldbalanceDest","newbalanceDest"
    ]
    target_col = "isFraud"

    X = df[feature_cols].copy()
    y = df[target_col]

    print("Training hybrid model...")
    # Convert categorical/string columns to category codes
    categorical_cols = ["type", "nameOrig", "nameDest"]
    for col in categorical_cols:
        X.loc[:, col] = X[col].astype("category").cat.codes

    model = HybridFraudDetectionModel(rule_weight=0.5)
    model.fit(X, y)

    print("Saving artifacts...")

    # 1) Save full hybrid model (pickle)
    joblib.dump(model, MODEL_PATH)

    # 2) Save XGB model separately
    model.xgb_model.save_model(XGB_PATH)

    # 3) Save metadata (features + threshold)
    metadata = {
        "features": feature_cols,
        "threshold": THRESHOLD
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f)

    print(f"Saved hybrid model → {MODEL_PATH}")
    print(f"Saved XGBoost core → {XGB_PATH}")
    print(f"Saved metadata → {META_PATH}")

if __name__ == "__main__":
    main()

