import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

import config
from preprocess import preprocess_data

def train_system():
    # 1. Load Data
    if not os.path.exists(config.RAW_DATA_PATH):
        print(f"Error: Dataset not found at {config.RAW_DATA_PATH}")
        return
    
    df = pd.read_csv(config.RAW_DATA_PATH)
    print(f"Loaded dataset with {len(df)} rows.")

    # 2. Routing Logic: Check for Labels
    is_supervised = config.TARGET_COL in df.columns and df[config.TARGET_COL].notnull().any()

    if is_supervised:
        print("--- Mode: Supervised Learning (XGBoost) ---")
        
        # Preprocess for training
        processed_df = preprocess_data(df, is_training=True)
        
        X = processed_df.drop(columns=[config.TARGET_COL])
        y = processed_df[config.TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Train XGBoost - Optimized for classification of known fraud cases
        model = xgb.XGBClassifier(**config.XGB_PARAMS)
        model.fit(X_train, y_train)

        # Evaluate using precision, recall, and F1-score
        y_pred = model.predict(X_test)
        print("\nEvaluation Metrics:")
        print(classification_report(y_test, y_pred))
        
        # Save Model for the production pipeline
        joblib.dump(model, config.SUPERVISED_MODEL_PATH)
        print(f"Model saved to: {config.SUPERVISED_MODEL_PATH}")

    else:
        print("--- Mode: Unsupervised Learning (Isolation Forest) ---")
        
        # Preprocess (IsFraud won't be in columns)
        processed_df = preprocess_data(df, is_training=True)
        
        # Train Isolation Forest - Best for detecting unknown anomalies
        model = IsolationForest(contamination=0.03, random_state=42) 
        model.fit(processed_df)

        # Save Model
        joblib.dump(model, config.UNSUPERVISED_MODEL_PATH)
        print(f"Model saved to: {config.UNSUPERVISED_MODEL_PATH}")

if __name__ == "__main__":
    train_system()