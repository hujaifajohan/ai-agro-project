"""
AgroAI Decision Tree V2 Prediction Service
Loads trained pipelines from backend/models/decision_tree_v2/
Executes inference for Water Need and Field Status, and computes deterministic Action policy.
"""

import os
import json
import pandas as pd
import numpy as np
import joblib

class DecisionTreeV2Service:
    def __init__(self):
        self.is_loaded = False
        self.water_pipeline = None
        self.status_pipeline = None
        self.metadata = {}
        self.feature_info = {}
        self.load_models()

    def _resolve_model_dir(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
        
        model_dir = os.path.join(project_root, "backend", "models", "decision_tree_v2")
        if not os.path.exists(model_dir):
            model_dir = os.path.abspath("backend/models/decision_tree_v2")
        return model_dir

    def load_models(self):
        try:
            model_dir = self._resolve_model_dir()
            water_path = os.path.join(model_dir, "water_need_pipeline.joblib")
            status_path = os.path.join(model_dir, "status_pipeline.joblib")
            meta_path = os.path.join(model_dir, "metadata.json")
            feature_path = os.path.join(model_dir, "feature_info.json")

            if os.path.exists(water_path) and os.path.exists(status_path):
                self.water_pipeline = joblib.load(water_path)
                self.status_pipeline = joblib.load(status_path)
                
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                if os.path.exists(feature_path):
                    with open(feature_path, "r", encoding="utf-8") as f:
                        self.feature_info = json.load(f)

                self.is_loaded = True
                print("[DecisionTreeV2Service] Decision Tree V2 models successfully loaded.")
            else:
                print(f"[DecisionTreeV2Service] Model files not found in {model_dir}. Please run train_decision_tree_v2.py.")
                self.is_loaded = False
        except Exception as e:
            print(f"[DecisionTreeV2Service] Error loading Decision Tree V2 models: {e}")
            self.is_loaded = False

    def determine_action_policy(self, status: str, water_need: str) -> tuple:
        """
        Deterministic Action policy mapping Status + Water Need -> (Action, ActionRoute).
        
        Matrix:
        - Critical + Urgent/High/Moderate/Low -> Pathogen AI
        - Urgent (any Status) -> Pathogen AI
        - Critical / High Water Need -> Pathogen AI or Queue Run
        - Attention + High/Moderate/Urgent -> Queue Run
        - Attention + Low -> Inspect (or Queue Run if dry)
        - Healthy + Moderate/High -> Queue Run
        - Healthy + Low -> Inspect
        """
        status_norm = str(status).strip().capitalize()
        water_norm = str(water_need).strip().capitalize()

        if status_norm == "Critical" or water_norm == "Urgent":
            return "Pathogen AI", "/disease-detection"
        elif status_norm == "Attention" or water_norm == "High":
            return "Queue Run", "/irrigation-planner"
        elif water_norm == "Moderate":
            return "Inspect", "/fields"
        else:
            # Healthy + Low
            return "Inspect", "/fields"

    def predict(
        self,
        crop: str,
        N: float,
        P: float,
        K: float,
        temperature: float,
        humidity: float,
        ph: float,
        rainfall: float,
        soil_moisture: float
    ) -> dict:
        if not self.is_loaded:
            # Try reloading if models weren't ready at import time
            self.load_models()
            if not self.is_loaded:
                return {
                    "success": False,
                    "error": "Decision Tree V2 model is not loaded. Please run model training script."
                }

        # Input Validation
        try:
            crop_str = str(crop).strip().lower() if crop else "rice"
            n_val = float(N)
            p_val = float(P)
            k_val = float(K)
            temp_val = float(temperature)
            hum_val = float(humidity)
            ph_val = float(ph)
            rain_val = float(rainfall)
            sm_val = float(soil_moisture)
        except (ValueError, TypeError) as e:
            return {
                "success": False,
                "error": f"Invalid numeric input parameters: {e}"
            }

        # Range Validations
        if not (0 <= n_val <= 300 and 0 <= p_val <= 300 and 0 <= k_val <= 300):
            return {"success": False, "error": "NPK values out of realistic range [0, 300]"}
        if not (-20 <= temp_val <= 60):
            return {"success": False, "error": "Temperature out of realistic range [-20, 60] °C"}
        if not (0 <= hum_val <= 100):
            return {"success": False, "error": "Humidity out of realistic range [0, 100] %"}
        if not (0 <= ph_val <= 14):
            return {"success": False, "error": "pH out of realistic range [0, 14]"}
        if not (0 <= rain_val <= 1000):
            return {"success": False, "error": "Rainfall out of realistic range [0, 1000] mm"}
        if not (0 <= sm_val <= 100):
            return {"success": False, "error": "Soil Moisture out of realistic range [0, 100] %"}

        # Construct input DataFrame with exact features expected by preprocessor
        input_data = pd.DataFrame([{
            "crop": crop_str,
            "N": n_val,
            "P": p_val,
            "K": k_val,
            "temperature": temp_val,
            "humidity": hum_val,
            "ph": ph_val,
            "rainfall": rain_val,
            "Soil Moisture": sm_val
        }])

        try:
            # Predict Status
            status_pred = str(self.status_pipeline.predict(input_data)[0])
            status_proba = self.status_pipeline.predict_proba(input_data)[0]
            status_classes = list(self.status_pipeline.classes_)
            status_conf = float(np.max(status_proba) * 100)

            # Predict Water Need
            water_pred = str(self.water_pipeline.predict(input_data)[0])
            water_proba = self.water_pipeline.predict_proba(input_data)[0]
            water_classes = list(self.water_pipeline.classes_)
            water_conf = float(np.max(water_proba) * 100)

            # Compute Policy Action
            action, action_route = self.determine_action_policy(status_pred, water_pred)

            # Low confidence check
            is_low_confidence = status_conf < 50.0 or water_conf < 50.0

            reasons = [
                f"Field status predicted as '{status_pred}' with {status_conf:.1f}% model confidence.",
                f"Water requirement predicted as '{water_pred}' with {water_conf:.1f}% model confidence."
            ]
            if is_low_confidence:
                reasons.append("AI confidence is low. Please inspect the field manually.")

            return {
                "success": True,
                "status": status_pred,
                "water_need": water_pred,
                "action": action,
                "action_route": action_route,
                "status_confidence": round(status_conf, 1),
                "water_need_confidence": round(water_conf, 1),
                "status_probabilities": {cls: round(float(prob * 100), 1) for cls, prob in zip(status_classes, status_proba)},
                "water_need_probabilities": {cls: round(float(prob * 100), 1) for cls, prob in zip(water_classes, water_proba)},
                "is_low_confidence": is_low_confidence,
                "reasons": reasons,
                "model_version": self.metadata.get("model_version", "v2.0.0"),
                "soil_moisture_provenance": "Dataset V2 contains synthetic/derived Soil Moisture values."
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Inference execution failed: {e}"
            }

# Global Singleton Instance
dtree_v2_service = DecisionTreeV2Service()
