import requests
import time
import random
from datetime import datetime

API_URL = "http://127.0.0.1:8000/predict"

def generate_chaotic_data():
    """Generates a mix of normal, high-risk, and anomalous data."""
    # Possible values
    cats = ["groceries", "electronics", "travel", "pharmacy", "gaming"]
    locs = ["Mumbai", "New York", "London", "Hyderabad", "Dubai"]
    anomalies = ["Proxy_Server_7", "VPN_Tunnel_Alpha", "Dark_Net_Relay", "Unknown_Origin"]
    
    # Randomly decide what type of transaction to send
    dice_roll = random.random()

    if dice_roll < 0.70:
        # 70% chance: Normal transaction (Clean)
        amount = random.uniform(10, 500)
        location = random.choice(locs)
        desc = "Normal"
    elif dice_roll < 0.90:
        # 20% chance: High-Value Attack (Triggers XGBoost/Supervised)
        amount = random.uniform(8000, 15000)
        location = random.choice(locs)
        desc = "High-Value"
    else:
        # 10% chance: Statistical Anomaly (Triggers IsoForest/Unsupervised)
        amount = random.uniform(5, 50)
        location = random.choice(anomalies)
        desc = "Anomaly"

    return {
        "Transaction_ID": f"LIVE_{random.randint(1000, 9999)}",
        "User_ID": f"USER_{random.randint(10, 99)}",
        "Transaction_Amount": round(amount, 2),
        "Merchant_Category": random.choice(cats),
        "Location": location,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }, desc

print("🛡️  FRAUD SHIELD LIVE MONITORING STARTED...")
print(f"Targeting API: {API_URL}\n")

try:
    while True:
        tx_data, intent = generate_chaotic_data()
        
        # Hit the Search Engine
        response = requests.post(API_URL, json=tx_data)
        
        if response.status_code == 200:
            res = response.json()
            status_icon = "🚨 [FRAUD]" if res['is_fraud'] else "✅ [CLEAN]"
            
            print(f"{status_icon} ID: {tx_data['Transaction_ID']} | Intent: {intent:<10} | "
                  f"Model: {res.get('model_used', 'N/A'):<12} | Strategy: {res.get('strategy', 'N/A')}")
        
        time.sleep(2) # 2-second heartbeat
except KeyboardInterrupt:
    print("\nStopping Live Monitor...")