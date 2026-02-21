from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import json
import asyncio

import config
from src.detector import FraudDetector

app = FastAPI(title="Fraud Shield: Live Monitoring Console")
detector = FraudDetector()

# --- REAL-TIME CONNECTION MANAGER ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass # Clean up stale connections if necessary

manager = ConnectionManager()

class Transaction(BaseModel):
    Transaction_ID: str
    User_ID: str
    Transaction_Amount: float
    Merchant_Category: str
    Location: str
    Timestamp: str

@app.get("/")
def get_dashboard():
    """Serves the Live UI Dashboard."""
    return FileResponse("index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """The 'Pipe' that pushes live data to your browser."""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # Keeps connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/predict")
async def predict_fraud(data: Transaction):
    try:
        transaction_dict = data.model_dump()
        result = detector.detect(transaction_dict)
        
        # Add the strategy description for the UI
        if result['model_used'] == 'supervised':
            result['strategy'] = "Pattern Match Search (XGBoost)"
        else:
            result['strategy'] = "Anomaly Isolation Search (IsoForest)"

        # PREPARE DATA FOR THE LIVE BOX
        full_payload = {
            "tx": transaction_dict,
            "analysis": result
        }

        # BROADCAST TO INTERFACE IN REAL-TIME
        await manager.broadcast(json.dumps(full_payload))
        
        return result
    except Exception as e:
        # Emergency filter for 'Impossible' data (like Mars_Station)
        error_result = {
            "is_fraud": True,
            "confidence": "Anomaly",
            "model_used": "emergency_filter",
            "strategy": "Handling Unknown Location Exception"
        }
        # Even errors get broadcasted as flagged anomalies!
        await manager.broadcast(json.dumps({"tx": data.model_dump(), "analysis": error_result}))
        return error_result

# Keep your /metrics endpoint below this for the PPT stats...