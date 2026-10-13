from pathlib import Path
 
import joblib
import pandas as pd
 
from fastapi import FastAPI
from pydantic import BaseModel
 
 
# --------------------------------------------------
# Paths
# --------------------------------------------------
 
BASE_DIR = Path(__file__).resolve().parent.parent
 
MODEL_PATH = (
    BASE_DIR
    / "models"
    / "fraud_detection_model.pkl"
)
 
 
# --------------------------------------------------
# Load trained model
# --------------------------------------------------
 
model = None

def load_model():
    global model
 
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )
 
        model = joblib.load(MODEL_PATH)
 
    return model
 
 
 
# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------
 
app = FastAPI(
    title="Online Fraud Detection API",
    description="API for detecting fraudulent online transactions",
    version="1.0.0"
)
 
 
# --------------------------------------------------
# Input schema
# --------------------------------------------------
 
class Transaction(BaseModel):
 
    step: float
    type: str
    amount: float
 
    oldbalanceOrg: float
    newbalanceOrig: float
 
    oldbalanceDest: float
    newbalanceDest: float
 
    nameDest: str
 
 
# --------------------------------------------------
# Health check
# --------------------------------------------------
 
@app.get("/")
def home():
 
    return {
        "message": "Online Fraud Detection API is running"
    }
 
 
# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
 
@app.post("/predict")
def predict(transaction: Transaction):
 
    data = pd.DataFrame(
        [transaction.model_dump()]
    )
 
    model_instance = load_model()
 
    prediction = model_instance.predict(data)[0]
 
    probability = model_instance.predict_proba(
    data
)[0][1]
 
    return {
        "prediction": int(prediction),
        "fraud": bool(prediction == 1),
        "fraud_probability": float(probability)
    }