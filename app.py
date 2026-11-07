import os, joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="House Price Prediction API", version="1.0.0")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=500, detail="Model artifact missing. Run train.py first.")
        _model = joblib.load(MODEL_PATH)
    return _model

class HouseFeatures(BaseModel):
    square_feet: float = Field(..., ge=200.0, le=15000.0, example=1750.0)
    bedrooms: int = Field(..., ge=1, le=10, example=3)
    bathrooms: int = Field(..., ge=1, le=8, example=2)
    property_age_years: int = Field(..., ge=0, le=100, example=5)
    location_tier: int = Field(..., ge=1, le=3, description="1: Prime, 2: Suburban, 3: Rural", example=1)
    garage_spaces: int = Field(..., ge=0, le=6, example=2)

@app.get("/")
def root():
    return {"status": "online", "service": "House Price Prediction API", "docs_url": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_artifact_loaded": os.path.exists(MODEL_PATH)}

@app.post("/predict")
def predict_price(payload: HouseFeatures):
    model = get_model()
    df = pd.DataFrame([payload.model_dump()])
    price = float(model.predict(df)[0])
    return {"predicted_price": round(price, 2), "currency": "USD", "status": "Success"}