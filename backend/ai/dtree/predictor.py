"""
AgroAI — Decision Tree Crop Recommendation Predictor
Module: backend/ai/dtree/predictor.py
Loads trained DecisionTree model and provides inference with input validation and confidence calculation.
"""

import os
import math
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "crop_recommendation_model.pkl"
META_PATH = MODEL_DIR / "model_metadata.json"

FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Pre-defined display titles for crops
CROP_DISPLAY_NAMES = {
    "rice": "Rice 🌾",
    "maize": "Maize / Corn 🌽",
    "chickpea": "Chickpea 🫘",
    "kidneybeans": "Kidney Beans 🫘",
    "pigeonpeas": "Pigeon Peas 🫛",
    "mothbeans": "Moth Beans 🫘",
    "mungbean": "Mung Bean 🫛",
    "blackgram": "Black Gram 🫘",
    "lentil": "Lentil 🫘",
    "pomegranate": "Pomegranate 🍎",
    "banana": "Banana 🍌",
    "mango": "Mango 🥭",
    "grapes": "Grapes 🍇",
    "watermelon": "Watermelon 🍉",
    "muskmelon": "Muskmelon 🍈",
    "apple": "Apple 🍎",
    "orange": "Orange 🍊",
    "papaya": "Papaya 🍈",
    "coconut": "Coconut 🥥",
    "cotton": "Cotton ☁️",
    "jute": "Jute 🌿",
    "coffee": "Coffee ☕"
}

class CropRecommendationPredictor:
    def __init__(self):
        self.model = None
        self.metadata = {}
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        """Loads trained scikit-learn model once on startup."""
        if MODEL_PATH.exists():
            try:
                self.model = joblib.load(MODEL_PATH)
                if META_PATH.exists():
                    with open(META_PATH, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                self.is_loaded = True
                print(f"[CropRecommendationPredictor] Model loaded successfully from {MODEL_PATH}")
            except Exception as e:
                print(f"[CropRecommendationPredictor] Error loading model: {e}")
                self.is_loaded = False
        else:
            print(f"[CropRecommendationPredictor] Model file not found at {MODEL_PATH}")
            self.is_loaded = False

    def validate_inputs(self, N: float, P: float, K: float, temperature: float, humidity: float, ph: float, rainfall: float) -> tuple[bool, str]:
        """Validates numerical feature inputs against sane agricultural bounds."""
        vals = {"N": N, "P": P, "K": K, "temperature": temperature, "humidity": humidity, "ph": ph, "rainfall": rainfall}
        
        for key, val in vals.items():
            if val is None:
                return False, f"Feature '{key}' is required and cannot be null."
            try:
                num_val = float(val)
                if math.isnan(num_val) or math.isinf(num_val):
                    return False, f"Feature '{key}' contains invalid numeric value (NaN or Inf)."
            except (ValueError, TypeError):
                return False, f"Feature '{key}' must be a valid numeric value."

        if N < 0 or N > 300:
            return False, f"Nitrogen (N) value ({N}) out of realistic soil range [0, 300]."
        if P < 0 or P > 300:
            return False, f"Phosphorus (P) value ({P}) out of realistic soil range [0, 300]."
        if K < 0 or K > 300:
            return False, f"Potassium (K) value ({K}) out of realistic soil range [0, 300]."
        if temperature < -20 or temperature > 60:
            return False, f"Temperature value ({temperature}°C) out of valid atmospheric range [-20, 60]."
        if humidity < 0 or humidity > 100:
            return False, f"Humidity value ({humidity}%) must be between 0% and 100%."
        if ph < 0 or ph > 14:
            return False, f"pH value ({ph}) must be between 0 and 14."
        if rainfall < 0 or rainfall > 1000:
            return False, f"Rainfall value ({rainfall} mm) out of valid range [0, 1000]."

        return True, ""

    def predict(self, N: float, P: float, K: float, temperature: float, humidity: float, ph: float, rainfall: float) -> dict:
        """Runs crop recommendation prediction using trained Decision Tree."""
        is_valid, err_msg = self.validate_inputs(N, P, K, temperature, humidity, ph, rainfall)
        if not is_valid:
            return {"success": False, "error": err_msg}

        if not self.is_loaded or self.model is None:
            # Fallback re-attempt load
            self._load_model()
            if not self.is_loaded or self.model is None:
                return {"success": False, "error": "Crop recommendation model is currently unavailable."}

        try:
            # Arrange features in EXACT order used during training
            features_row = [float(N), float(P), float(K), float(temperature), float(humidity), float(ph), float(rainfall)]
            df_input = pd.DataFrame([features_row], columns=FEATURE_NAMES)

            # Predict label & probability distribution
            raw_pred = self.model.predict(df_input)[0]
            raw_crop = str(raw_pred).lower().strip()
            
            proba = self.model.predict_proba(df_input)[0]
            confidence = round(float(np.max(proba)) * 100, 2)

            display_crop = CROP_DISPLAY_NAMES.get(raw_crop, raw_crop.capitalize())
            importances = self.metadata.get("feature_importances", {})

            return {
                "success": True,
                "recommendedCrop": display_crop,
                "rawCrop": raw_crop,
                "confidence": confidence,
                "model": self.metadata.get("algorithm", "Decision Tree Classifier"),
                "features": {
                    "N": float(N),
                    "P": float(P),
                    "K": float(K),
                    "temperature": float(temperature),
                    "humidity": float(humidity),
                    "ph": float(ph),
                    "rainfall": float(rainfall)
                },
                "featureImportances": importances,
                "modelAccuracy": self.metadata.get("accuracy", 0.9886)
            }
        except Exception as exc:
            print(f"[CropRecommendationPredictor] Prediction exception: {exc}")
            return {"success": False, "error": f"Model inference error: {str(exc)}"}

# Global singleton instance for efficient zero-cost request handling
crop_predictor = CropRecommendationPredictor()
