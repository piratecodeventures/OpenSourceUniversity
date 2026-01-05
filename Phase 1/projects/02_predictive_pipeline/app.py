from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# 1. Load Model
try:
    model = joblib.load("model.pkl")
except:
    model = None # Handle missing model during dev

# 2. Define Input Schema
class HouseFeatures(BaseModel):
    feat_0: float
    feat_1: float
    feat_2: float
    feat_3: float
    feat_4: float
    feat_5: float
    feat_6: float
    feat_7: float
    feat_8: float
    feat_9: float

# 3. Create App
app = FastAPI(title="House Price Predictor", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(features: HouseFeatures):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert Pydantic -> DataFrame (Model expects names)
    data = pd.DataFrame([features.dict()])
    
    try:
        prediction = model.predict(data)
        return {"predicted_price": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    print("Run with: uvicorn app:app --reload")
