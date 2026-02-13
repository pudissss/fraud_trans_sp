import requests
import random
import time

URL = "http://127.0.0.1:8000/predict"

def run_stress_test():
    print("\n" + "="*60)
    print("LIVE PERFORMANCE SEARCH: SUPERVISED vs UNSUPERVISED")
    print("="*60)
    
    scenarios = [
        {"desc": "Massive Amount Attack", "amt": 99999.0, "loc": "Unknown", "cat": "travel"},
        {"desc": "Micro-Transaction Spam", "amt": 0.01, "loc": "New York", "cat": "groceries"},
        {"desc": "Impossible Location", "amt": 250.0, "loc": "Mars_Base_1", "cat": "electronics"}
    ]

    for scenario in scenarios:
        payload = {
            "Transaction_ID": f"STRESS_{random.randint(100, 999)}",
            "User_ID": "USER_STRESS",
            "Transaction_Amount": scenario["amt"],
            "Merchant_Category": scenario["cat"],
            "Location": scenario["loc"],
            "Timestamp": "2026-02-14 12:00:00"
        }
        
        start = time.time()
        try:
            response = requests.post(URL, json=payload, timeout=5)
            latency = (time.time() - start) * 1000
            res = response.json()
            
            # Print formatted results
            fraud_status = str(res.get('is_fraud', 'Error'))
            model_used = res.get('model_used', 'unknown')
            
            print(f"[{scenario['desc']:<22}] | Fraud: {fraud_status:<6} | Model: {model_used:<15} | {latency:>6.2f}ms")
        except Exception as e:
            print(f"[{scenario['desc']:<22}] | Request Failed: {str(e)}")

if __name__ == "__main__":
    run_stress_test()