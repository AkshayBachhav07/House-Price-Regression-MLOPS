import os, pytest
from fastapi.testclient import TestClient

if not os.path.exists("models/model.pkl"):
    import train

from app import app
client = TestClient(app)

def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "online"

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_model_artifact_exists():
    assert os.path.isfile("models/model.pkl")

def test_predict_price_success():
    payload = {
        "square_feet": 2000.0,
        "bedrooms": 3,
        "bathrooms": 2,
        "property_age_years": 5,
        "location_tier": 1,
        "garage_spaces": 2
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_price" in data
    assert data["predicted_price"] > 0
    assert data["status"] == "Success"

def test_validation_failure():
    bad_payload = {
        "square_feet": 2000.0,
        "bedrooms": 3,
        "bathrooms": 2,
        "property_age_years": 5,
        "location_tier": 99,
        "garage_spaces": 2
    }
    res = client.post("/predict", json=bad_payload)
    assert res.status_code == 422