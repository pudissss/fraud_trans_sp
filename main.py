from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics import classification_report
import joblib
import os

import config
from src.detector import FraudDetector

app = FastAPI(title="Real-Time Fraud Detection & Analytics API")

# Initialize detector outside the endpoint for speed
detector = FraudDetector()

class Transaction(BaseModel):
    Transaction_ID: str
    User_ID: str
    Transaction_Amount: float
    Merchant_Category: str
    Location: str
    Timestamp: str

@app.get("/")
def home():
    return {"status": "Fraud Detection API is Online", "model_mode": detector.mode}

@app.get("/metrics")
def get_metrics():
    if not os.path.exists(config.RAW_DATA_PATH):
        raise HTTPException(status_code=404, detail="Dataset not found for evaluation.")

    # 1. Load Ground Truth Data
    df = pd.read_csv(config.RAW_DATA_PATH)
    y_true = df[config.TARGET_COL]
    
    # 2. Extract Features
    X = df.drop(columns=[config.TARGET_COL] + config.DROP_COLS, errors='ignore')

    # --- CRITICAL FIX: PREPROCESSING ---
    # We must convert strings to numbers using the encoders you saved during training
    try:
        # Load the encoders (Ensure these filenames match what you saved in src/preprocess.py)
        le_merchant = joblib.load(os.path.join(config.MODEL_DIR, "le_Merchant_Category.pkl"))
        le_location = joblib.load(os.path.join(config.MODEL_DIR, "le_Location.pkl"))

        # Transform the string columns into numeric codes
        X['Merchant_Category'] = le_merchant.transform(X['Merchant_Category'])
        X['Location'] = le_location.transform(X['Location'])
    except Exception as e:
        print(f"Preprocessing Error: {e}")
        raise HTTPException(status_code=500, detail="Could not encode data for metrics calculation.")
    # -----------------------------------

    # 3. Get Supervised Metrics (XGBoost)
    xgb_model = joblib.load(config.SUPERVISED_MODEL_PATH)
    xgb_preds = xgb_model.predict(X) # Now X contains only numbers!
    xgb_report = classification_report(y_true, xgb_preds, output_dict=True)

    # 4. Get Unsupervised Metrics (Isolation Forest)
    iso_model = joblib.load(config.UNSUPERVISED_MODEL_PATH)
    iso_raw = iso_model.predict(X)
    iso_preds = [1 if p == -1 else 0 for p in iso_raw]
    iso_report = classification_report(y_true, iso_preds, output_dict=True)

    return {
        "supervised_xgboost": xgb_report["1"],
        "unsupervised_iso_forest": iso_report["1"]
    }

@app.post("/predict")
def predict_fraud(data: Transaction):
    try:
        transaction_dict = data.model_dump()
        # Detector handles the logic routing between XGBoost and IsoForest
        result = detector.detect(transaction_dict)
        
        # Real-time console log for your demo
        status = "FRAUD" if result['is_fraud'] else "CLEAN"
        print(f"[{result['model_used'].upper()}] ID: {data.Transaction_ID} | Result: {status}")
        
        return result
    except Exception as e:
        # Fallback if preprocessing fails on 'Impossible' data
        print(f"WARN: Handled outlier {data.Location}. Error: {str(e)}")
        return {
            "is_fraud": True, 
            "confidence": "Anomaly", 
            "model_used": "emergency_filter",
            "reason": "Unknown Location/Pattern Detected"
        }