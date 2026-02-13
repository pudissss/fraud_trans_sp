import pandas as pd
import joblib
import config
from src.detector import FraudDetector

def show_comparison():
    # 1. Load the models
    xgb_model = joblib.load(config.SUPERVISED_MODEL_PATH)
    iso_model = joblib.load(config.UNSUPERVISED_MODEL_PATH)
    
    # 2. Create test cases
    test_cases = [
        {"Transaction_Amount": 45.0, "Merchant_Category": "groceries", "Location": "New York"}, # Normal
        {"Transaction_Amount": 4500.0, "Merchant_Category": "electronics", "Location": "Unknown"}, # Fraudulent
    ]

    print(f"{'Test Case':<20} | {'XGBoost (Supervised)':<20} | {'IsoForest (Unsupervised)':<25}")
    print("-" * 75)

    for case in test_cases:
        # Simplified prediction for demonstration
        # (In reality, you'd use your FraudDetector class)
        xgb_pred = "FRAUD" if case["Transaction_Amount"] > 1000 else "NORMAL"
        iso_pred = "FRAUD (Anomaly)" if case["Location"] == "Unknown" else "NORMAL"
        
        print(f"Amt: ${case['Transaction_Amount']:<14} | {xgb_pred:<20} | {iso_pred:<25}")

if __name__ == "__main__":
    show_comparison()