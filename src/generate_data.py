import pandas as pd
import numpy as np
import os
import config # Uses RAW_DATA_PATH from your config

def generate_synthetic_data():
    np.random.seed(42)
    
    # 1. Generate Normal Transactions (~10,000 rows) [cite: 75]
    n_normal = 10000
    normal_data = {
        'Transaction_ID': [f'TXN{i:05d}' for i in range(n_normal)],
        'User_ID': [f'USER{np.random.randint(1000, 5000)}' for _ in range(n_normal)],
        # Use 'min' for minute frequency
        'Timestamp': pd.date_range(start='2026-01-01', periods=n_normal, freq='min'), 
        'Transaction_Amount': np.random.uniform(10, 500, n_normal), 
        'Merchant_Category': np.random.choice(['groceries', 'electronics', 'restaurant', 'clothing'], n_normal), 
        'Location': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_normal), 
        'IsFraud': 0
    }
    df_normal = pd.DataFrame(normal_data)

    # 2. Inject Fraudulent Transactions (~300 rows) [cite: 83]
    n_fraud = 300
    fraud_data = {
        'Transaction_ID': [f'FRAUD{i:05d}' for i in range(n_fraud)],
        'User_ID': [f'USER{np.random.randint(1000, 5000)}' for _ in range(n_fraud)],
        # CHANGE: Use 'h' instead of 'H' for pandas 3.0+ compatibility
        'Timestamp': pd.date_range(start='2026-01-01', periods=n_fraud, freq='h'), 
        'Transaction_Amount': np.random.uniform(1000, 5000, n_fraud), # Unusual Amounts [cite: 84]
        'Merchant_Category': np.random.choice(['electronics', 'travel'], n_fraud),
        'Location': np.random.choice(['Moscow', 'London', 'Unknown'], n_fraud),
        'IsFraud': 1
    }
    df_fraud = pd.DataFrame(fraud_data)

    # 3. Combine and Shuffle [cite: 88, 90]
    df = pd.concat([df_normal, df_fraud]).sample(frac=1).reset_index(drop=True)
    
    # Save to the path defined in config.py
    df.to_csv(config.RAW_DATA_PATH, index=False)
    print(f"Dataset successfully created at: {config.RAW_DATA_PATH}")

if __name__ == "__main__":
    generate_synthetic_data()