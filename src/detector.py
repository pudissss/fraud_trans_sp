import joblib
import pandas as pd
import os
import config
from src.preprocess import preprocess_data

class FraudDetector:
    def __init__(self):
        # Determine if we should use Supervised or Unsupervised based on what exists
        if os.path.exists(config.SUPERVISED_MODEL_PATH):
            self.model = joblib.load(config.SUPERVISED_MODEL_PATH)
            self.mode = "supervised"
        else:
            self.model = joblib.load(config.UNSUPERVISED_MODEL_PATH)
            self.mode = "unsupervised"
        
        # Load the essential preprocessors
        self.scaler = joblib.load(os.path.join(config.MODEL_DIR, "scaler_amount.pkl"))
        self.le_merchant = joblib.load(os.path.join(config.MODEL_DIR, "le_Merchant_Category.pkl"))
        self.le_location = joblib.load(os.path.join(config.MODEL_DIR, "le_Location.pkl"))

    def detect(self, transaction_data: dict):
        """
        Takes a raw dictionary of transaction data and returns a fraud prediction.
        """
        df = pd.DataFrame([transaction_data])
        
        # Preprocess the single row using the saved states
        df['Merchant_Category'] = self.le_merchant.transform(df['Merchant_Category'].astype(str))
        df['Location'] = self.le_location.transform(df['Location'].astype(str))
        df[config.NUMERIC_COLS] = self.scaler.transform(df[config.NUMERIC_COLS])
        
        # Drop IDs as we did in training
        X = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns], errors='ignore')
        if config.TARGET_COL in X.columns:
            X = X.drop(columns=[config.TARGET_COL])

        # Predict
        prediction = self.model.predict(X)[0]
        
        # Isolation Forest returns -1 for anomalies, XGBoost returns 1 for fraud
        is_fraud = bool(prediction == 1 if self.mode == "supervised" else prediction == -1)
        
        return {
            "is_fraud": is_fraud,
            "confidence": "High" if is_fraud else "Normal",
            "model_used": self.mode
        }

if __name__ == "__main__":
    # Test with a suspicious transaction
    detector = FraudDetector()
    sample = {
        "Transaction_ID": "TXN_TEST_99",
        "User_ID": "USER_999",
        "Transaction_Amount": 5000.0, # High amount
        "Merchant_Category": "electronics",
        "Location": "Unknown",
        "Timestamp": "2026-02-13 23:00:00"
    }
    print(detector.detect(sample))