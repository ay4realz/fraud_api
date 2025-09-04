# 🛡️ Hybrid Adaptive Fraud Detection API

This project implements a **Hybrid Adaptive Fraud Detection Model** and serves it through a **FastAPI** backend.  
It combines **rule-based detection** and a **machine learning model (XGBoost)** to predict fraudulent financial transactions.  

The API supports:
- **Single transaction prediction** (via JSON input).  
- **Batch transaction prediction** (via CSV upload → downloadable CSV output).  

---

## 🚀 Features
- Hybrid model (`rule-based + XGBoost`) for robust fraud detection.
- REST API built with **FastAPI**.
- `/predict` endpoint → predict fraud for a single transaction.
- `/batch` endpoint → upload a CSV of multiple transactions and download results.
- Pre-trained model artifacts stored in `/model/`.

---

## 🗂️ Project Structure
```
fraud_api/
├── app/
│   ├── main.py              # FastAPI app entrypoint
│   ├── fraud_model.py       # Hybrid model implementation
│   └── train_model.py       # Script to train/retrain the model
├── model/
│   ├── hybrid_fraud_detection_model.pkl  # Saved hybrid model
│   ├── xgb_model.json                     # XGBoost core model
│   └── metadata.json                      # Model metadata (features + threshold)
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container definition (for Railway/Cloud Run)
└── README.md                # Project documentation
```

---

## ⚙️ Installation (Local)

1. Clone the repository:
   ```bash
   git clone https://github.com/ay4realz/fraud_api.git
   cd fraud_api
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the API locally:
   ```bash
   uvicorn app.main:app --reload
   ```

5. Open in browser:
   ```
   http://127.0.0.1:8000/docs
   ```

---

## 🔮 API Endpoints

### 1. Health Check
```http
GET /healthz
```

---

### 2. Single Transaction Prediction
```http
POST /predict
```

**Request body (JSON):**
```json
{
  "step": 1,
  "type": "TRANSFER",
  "amount": 1000,
  "nameOrig": "C12345",
  "oldbalanceOrg": 2000,
  "newbalanceOrig": 1000,
  "nameDest": "M54321",
  "oldbalanceDest": 0,
  "newbalanceDest": 1000
}
```

**Response:**
```json
{
  "fraud_probability": 0.51,
  "fraud_label": 1
}
```

**Test with curl:**
```bash
curl -X POST "http://127.0.0.1:8000/predict"   -H "Content-Type: application/json"   -d '{"step":1,"type":"TRANSFER","amount":1000,"nameOrig":"C12345","oldbalanceOrg":2000,"newbalanceOrig":1000,"nameDest":"M54321","oldbalanceDest":0,"newbalanceDest":1000}'
```

---

### 3. Batch Transaction Prediction
```http
POST /batch
```

**Request:** Upload a `.csv` file with columns:
```
step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest
```

**Response:** A downloadable `predictions.csv` with added columns:
```
fraud_probability,fraud_label
```

**Test with curl:**
```bash
curl -X POST "http://127.0.0.1:8000/batch"   -F "file=@transactions.csv" -OJ
```

---

## 📦 Deployment

### Deploy on [Railway](https://railway.app):
1. Connect your GitHub repo to Railway.  
2. Add a new service → select this repo.  
3. Railway will build using `Dockerfile` and run FastAPI automatically.  
4. Get your live API URL (e.g. `https://fraud-api.up.railway.app`).  

---

## 📊 Model Details
- **HybridFraudDetectionModel**:
  - Combines:
    - Rule-based engine (simple heuristics for transaction anomalies).
    - XGBoost classifier trained on PaySim dataset.
  - Uses an **F1-optimal threshold** (`0.23`) to assign fraud labels.  
- Saved artifacts are portable and reloadable via `joblib`.

---

## 🙌 Acknowledgements
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework.  
- [XGBoost](https://xgboost.ai/) for fraud classification.  
- [PaySim Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) for training.  

---

## ✨ Author
**Ayomide Adenowo**  
Fraud detection system powered by machine learning & FastAPI 🚀
