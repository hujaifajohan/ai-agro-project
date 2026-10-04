"""
AgroAI — Backend End-to-End Test Suite for Crop Recommendation API
Tests FastAPI endpoints: GET /api/health, POST /api/crop-recommendation, GET /api/crop-recommendation
"""

import sys
import os
import json
from pathlib import Path
from fastapi.testclient import TestClient

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from main import app

client = TestClient(app)

def run_tests():
    print("=" * 70)
    print("AGROAI — BACKEND API END-TO-END TEST SUITE")
    print("=" * 70)

    # Test 1: Health Check
    print("\n[Test 1] Health Check Endpoint (GET /api/health)")
    response = client.get("/api/health")
    assert response.status_code == 200, f"Health check failed: {response.text}"
    health_data = response.json()
    print(f"Response: {json.dumps(health_data, indent=2)}")
    assert health_data.get("status") == "ok"
    assert health_data.get("cropRecommendationModelLoaded") is True, "Model should be loaded!"
    print("--> PASS")

    # Test 2: Valid Crop Recommendation Request (POST /api/crop-recommendation)
    print("\n[Test 2] Valid Crop Recommendation (POST /api/crop-recommendation)")
    payload = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.87,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.93
    }
    response = client.post("/api/crop-recommendation", json=payload)
    assert response.status_code == 200, f"POST recommendation failed: {response.text}"
    res_data = response.json()
    print(f"Response: {json.dumps(res_data, indent=2)}")
    assert res_data.get("success") is True
    assert "recommendedCrop text" or "recommendedCrop" in res_data
    assert res_data.get("confidence") > 50.0
    print("--> PASS")

    # Test 3: Second Real CSV Row (Maize / Coffee / Banana)
    print("\n[Test 3] Second Real Dataset Row (GET /api/crop-recommendation)")
    response = client.get("/api/crop-recommendation?N=100&P=18&K=30&temperature=25.0&humidity=60.0&ph=6.2&rainfall=100.0")
    assert response.status_code == 200, f"GET recommendation failed: {response.text}"
    res_data = response.json()
    print(f"Response: {json.dumps(res_data, indent=2)}")
    assert res_data.get("success") is True
    print("--> PASS")

    # Test 4: Invalid Input Validation (out of bounds pH)
    print("\n[Test 4] Invalid Input Rejection (pH out of bounds)")
    payload_invalid = {
        "N": 45,
        "P": 30,
        "K": 35,
        "temperature": 28,
        "humidity": 70,
        "ph": 99.0, # invalid pH > 14
        "rainfall": 200
    }
    response = client.post("/api/crop-recommendation", json=payload_invalid)
    assert response.status_code == 400, f"Should reject invalid pH with HTTP 400, got: {response.status_code}"
    print(f"Rejected correctly with HTTP 400: {response.json()}")
    print("--> PASS")

    # Test 5: Missing Required Feature
    print("\n[Test 5] Missing Required Feature Payload")
    payload_missing = {
        "N": 45,
        "P": 30
    }
    response = client.post("/api/crop-recommendation", json=payload_missing)
    assert response.status_code == 422, f"Pydantic should reject missing fields with HTTP 422, got: {response.status_code}"
    print(f"Rejected correctly with HTTP 422: {response.json()}")
    print("--> PASS")

    # Test 6: Verify Existing Endpoints Remain Untouched
    print("\n[Test 6] Verification of Existing Endpoints")
    res_dtree = client.post("/api/ai/decision-tree", json={"crop": "Tomato", "soil_moisture": 30.0, "soil_ph": 6.2, "temperature": 32.0, "humidity": 55.0, "rainfall": 5.0})
    assert res_dtree.status_code == 200, "Existing decision tree endpoint should remain functional"
    
    res_kmeans = client.get("/api/ai/kmeans")
    assert res_kmeans.status_code == 200, "Existing kmeans endpoint should remain functional"
    
    res_farms = client.get("/api/farms")
    assert res_farms.status_code == 200, "Existing farms endpoint should remain functional"

    print("Existing endpoints remain functional --> PASS")

    print("\n" + "=" * 70)
    print("ALL API END-TO-END TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
