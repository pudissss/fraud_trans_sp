import joblib
import pandas as pd
import os
import config

class FraudDetector:
    def __init__(self):
        # Load BOTH models for a hybrid search approach
        self.supervised_model = None
        self.unsupervised_model = None
        
        if os.path.exists(config.SUPERVISED_MODEL_PATH):
            self.supervised_model = joblib.load(config.SUPERVISED_MODEL_PATH)
        
        if os.path.exists(config.UNSUPERVISED_MODEL_PATH):
            self.unsupervised_model = joblib.load(config.UNSUPERVISED_MODEL_PATH)
        
        # Load essential preprocessors
        self.scaler = joblib.load(os.path.join(config.MODEL_DIR, "scaler_amount.pkl"))
        self.le_merchant = joblib.load(os.path.join(config.MODEL_DIR, "le_Merchant_Category.pkl"))
        self.le_location = joblib.load(os.path.join(config.MODEL_DIR, "le_Location.pkl"))

    def _safe_encode(self, encoder, value):
        val = str(value).strip().lower()
        if val in encoder.classes_:
            return encoder.transform([val])[0]
        return 0 

    def detect(self, transaction_data: dict):
        try:
            df = pd.DataFrame([transaction_data])
            
            # Robust Preprocessing
            df['Merchant_Category'] = df['Merchant_Category'].apply(lambda x: self._safe_encode(self.le_merchant, x))
            df['Location'] = df['Location'].apply(lambda x: self._safe_encode(self.le_location, x))
            df[config.NUMERIC_COLS] = self.scaler.transform(df[config.NUMERIC_COLS])
            
            X = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns], errors='ignore')
            if config.TARGET_COL in X.columns:
                X = X.drop(columns=[config.TARGET_COL])

            # --- HYBRID SEARCH LOGIC ---
            # 1. Check Supervised (XGBoost)
            sup_fraud = False
            if self.supervised_model:
                sup_fraud = bool(self.supervised_model.predict(X)[0] == 1)

            # 2. Check Unsupervised (Isolation Forest)
            unsup_fraud = False
            if self.unsupervised_model:
                unsup_fraud = bool(self.unsupervised_model.predict(X)[0] == -1)

            # Determine final verdict and which model "caught" it
            is_fraud = sup_fraud or unsup_fraud
            
            # Label which model took the lead for the UI
            used_mode = "supervised" if sup_fraud else ("unsupervised" if unsup_fraud else "supervised")

            return {
                "is_fraud": is_fraud,
                "confidence": "High Risk" if is_fraud else "Normal",
                "model_used": used_mode,
                "details": {
                    "xgboost_flag": sup_fraud,
                    "iso_forest_flag": unsup_fraud
                }
            }
        except Exception as e:
            raise e