import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import config # Importing our centralized config

def preprocess_data(df, is_training=True):
    """
    Cleans and transforms raw transaction data.
    """
    # 1. Handle missing values (industry best practice)
    df = df.copy()
    df = df.dropna()

    # 2. Drop unnecessary ID columns [cite: 64, 65]
    # We keep 'IsFraud' if it exists for training
    cols_to_drop = [c for c in config.DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 3. Label Encoding for Categorical Features [cite: 22, 68, 71]
    # We save encoders to ensure the same mapping during real-time API inference
    for col in config.CATEGORICAL_COLS:
        le = LabelEncoder()
        if is_training:
            df[col] = le.fit_transform(df[col].astype(str))
            joblib.dump(le, f"models/le_{col}.pkl")
        else:
            # Load existing encoder for prediction
            le = joblib.load(f"models/le_{col}.pkl")
            df[col] = le.transform(df[col].astype(str))

    # 4. Standard Scaling for Transaction Amount [cite: 22, 67]
    scaler = StandardScaler()
    if is_training:
        df[config.NUMERIC_COLS] = scaler.fit_transform(df[config.NUMERIC_COLS])
        joblib.dump(scaler, "models/scaler_amount.pkl")
    else:
        scaler = joblib.load("models/scaler_amount.pkl")
        df[config.NUMERIC_COLS] = scaler.transform(df[config.NUMERIC_COLS])

    return df

if __name__ == "__main__":
    # Test the preprocessor
    if os.path.exists(config.RAW_DATA_PATH):
        raw_df = pd.read_csv(config.RAW_DATA_PATH)
        processed_df = preprocess_data(raw_df)
        print("Data preprocessed successfully!")
        print(processed_df.head())
    else:
        print(f"File not found: {config.RAW_DATA_PATH}. Please run your data generation script first.")