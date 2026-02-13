import os

# --- Project Directories ---
# Points to the root folder (D:\Learnathon\project)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Ensure these directories exist on your machine
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- File Paths ---
# Matches your synthetic dataset location [cite: 55, 90]
RAW_DATA_PATH = os.path.join(DATA_DIR, "synthetic_transactions.csv")
# Paths for the saved .pkl model artifacts [cite: 103]
SUPERVISED_MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model_xgboost.pkl")
UNSUPERVISED_MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model_iso_forest.pkl")

# --- Feature Definitions ---
# Definitions based on your Synthetic Dataset Preparation [cite: 58]
TARGET_COL = "IsFraud" # [cite: 70]
DROP_COLS = ["Transaction_ID", "User_ID", "Timestamp"] # Identifiers to exclude from training [cite: 64, 65, 66]
CATEGORICAL_COLS = ["Merchant_Category", "Location"] # For Label Encoding [cite: 68, 71]
NUMERIC_COLS = ["Transaction_Amount"] # For Standard Scaling [cite: 67]

# --- Model Hyperparameters ---
# XGBoost settings optimized for imbalanced fraud data [cite: 46, 47]
# scale_pos_weight = (total_normal / total_fraud) -> 10,000 / 300 ≈ 33 
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "scale_pos_weight": 33, 
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": 42
}

# Isolation Forest settings for unsupervised anomaly detection [cite: 14, 19]
ISO_FOREST_PARAMS = {
    "n_estimators": 100,
    "contamination": 0.03, # Matches your expected 3% fraud rate [cite: 83]
    "random_state": 42
}