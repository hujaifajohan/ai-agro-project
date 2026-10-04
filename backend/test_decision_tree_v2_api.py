"""
AgroAI Decision Tree V2 Comprehensive Test Suite
Validates dataset, model training artifacts, prediction service, input handling, and FastAPI endpoints.
"""

import os
import json
import pytest
from fastapi.testclient import TestClient

from main import app
from ai.dtree.decision_tree_v2_service import dtree_v2_service

client = TestClient(app)

def test_model_artifacts_exist():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "models", "decision_tree_v2")
    
    assert os.path.exists(os.path.join(model_dir, "water_need_pipeline.joblib"))
    assert os.path.exists(os.path.join(model_dir, "status_pipeline.joblib"))
    assert os.path.exists(os.path.join(model_dir, "metadata.json"))
    assert os.path.exists(os.path.join(model_dir, "metrics.json"))
    assert os.path.exists(os.path.join(model_dir, "feature_info.json"))

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "decision_tree_v2" in data
    assert data["decision_tree_v2"]["loaded"] is True

def test_decision_v2_service_case_stress():
    # High temperature, low moisture, abnormal pH
    res = dtree_v2_service.predict(
        crop="rice",
        N=30,
        P=20,
        K=15,
        temperature=35.0,
        humidity=90.0,
        ph=4.5,
        rainfall=10.0,
        soil_moisture=12.0
    )
    assert res["success"] is True
    assert res["status"] in ["Critical", "Attention"]
    assert res["water_need"] in ["High", "Urgent"]
    assert res["action"] in ["Pathogen AI", "Queue Run"]

def test_decision_v2_service_case_healthy():
    # Ideal parameters for rice
    res = dtree_v2_service.predict(
        crop="rice",
        N=90,
        P=42,
        K=43,
        temperature=20.8,
        humidity=82.0,
        ph=6.5,
        rainfall=202.9,
        soil_moisture=46.9
    )
    assert res["success"] is True
    assert res["status"] == "Healthy"
    assert res["water_need"] == "Low"
    assert res["action"] == "Inspect"
    assert res["action_route"] == "/fields"
    assert res["status_confidence"] > 50.0
    assert res["water_need_confidence"] > 50.0

def test_decision_v2_service_unknown_crop():
    # Preprocessor must safely handle unknown crop string via OneHotEncoder(handle_unknown='ignore')
    res = dtree_v2_service.predict(
        crop="exotic_dragonfruit_unseen",
        N=50,
        P=30,
        K=30,
        temperature=28.0,
        humidity=65.0,
        ph=6.5,
        rainfall=100.0,
        soil_moisture=40.0
    )
    assert res["success"] is True
    assert "status" in res
    assert "water_need" in res

def test_decision_v2_api_post():
    payload = {
        "crop": "tomato",
        "N": 40.0,
        "P": 30.0,
        "K": 35.0,
        "temperature": 28.0,
        "humidity": 65.0,
        "ph": 6.5,
        "rainfall": 100.0,
        "soil_moisture": 45.0
    }
    response = client.post("/api/decision", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "status" in data
    assert "water_need" in data
    assert "action" in data
    assert "status_confidence" in data
    assert "water_need_confidence" in data

def test_decision_v2_api_get():
    response = client.get("/api/decision?crop=rice&N=90&P=42&K=43&temperature=20.8&humidity=82.0&ph=6.5&rainfall=202.9&soil_moisture=46.9")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["status"] == "Healthy"
    assert data["water_need"] == "Low"

def test_decision_v2_api_validation_error():
    payload = {
        "crop": "rice",
        "N": -999.0,  # Invalid N
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9,
        "soil_moisture": 46.9
    }
    response = client.post("/api/decision", json=payload)
    assert response.status_code == 400

def test_existing_endpoints_unbroken():
    # Verify existing endpoints are preserved and fully working
    res_crop = client.get("/api/crop-recommendation?N=90&P=42&K=43&temperature=20.8&humidity=82.0&ph=6.5&rainfall=202.9")
    assert res_crop.status_code == 200
    
    res_fields = client.get("/api/fields")
    assert res_fields.status_code == 200
    
    res_farms = client.get("/api/farms")
    assert res_farms.status_code == 200

if __name__ == "__main__":
    pytest.main(["-v", __file__])
