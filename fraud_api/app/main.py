from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import io

# Import model classes so joblib can resolve them
from app.fraud_model import HybridFraudDetectionModel, EnhancedAdaptiveRiskScorer

# ==========================
# Config
# ==========================
MODEL_PATH = "model/hybrid_fraud_detection_model.pkl"
THRESHOLD = 0.23  # replace with your F1-optimal threshold

# ==========================
# Load model
# ==========================
try:
    hybrid_model: HybridFraudDetectionModel = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load model from {MODEL_PATH}: {e}")

# ==========================
# FastAPI app
# ==========================
app = FastAPI(title="Hybrid Fraud Detection API")

# Enable CORS (so frontend can call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Pydantic schema
# ==========================
class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float


# ==========================
# Routes
# ==========================

@app.get("/healthz")
def health_check():
    return {"status": "ok", "message": "Hybrid Fraud Detection API is running."}


@app.post("/predict")
def predict_single(txn: Transaction):
    try:
        # Convert request into DataFrame
        df = pd.DataFrame([txn.dict()])

        # Run prediction
        prob = float(hybrid_model.predict_proba(df)[0])
        label = int(prob >= THRESHOLD)

        return {"fraud_probability": prob, "fraud_label": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from fastapi.responses import StreamingResponse
import io

@app.post("/batch")
def predict_batch(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Ensure required columns
        required_cols = [
            "step","type","amount","nameOrig","oldbalanceOrg",
            "newbalanceOrig","nameDest","oldbalanceDest","newbalanceDest"
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

        # Run predictions
        probs = hybrid_model.predict_proba(df)
        labels = (probs >= THRESHOLD).astype(int)

        df["fraud_probability"] = probs
        df["fraud_label"] = labels

        # Write output CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        # Return as downloadable file
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()

