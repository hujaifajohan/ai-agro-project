"""
AgroAI — Decision Tree Model Evaluation & Verification Script
Reads saved model and evaluates on test inputs and sample dataset rows.
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

MODEL_PATH = PROJECT_ROOT / "backend" / "ai" / "dtree" / "crop_recommendation_model.pkl"
META_PATH = PROJECT_ROOT / "backend" / "ai" / "dtree" / "model_metadata.json"
DATA_PATH = PROJECT_ROOT / "ai_training" / "data" / "Crop_recommendation.csv"

FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

def evaluate():
    print("=" * 70)
    print("AGROAI — DECISION TREE MODEL EVALUATION")
    print("=" * 70)

    if not MODEL_PATH.exists() or not META_PATH.exists():
        print("Model or metadata artifact missing!")
        return

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print(f"Algorithm: {meta.get('algorithm')}")
    print(f"Accuracy : {meta.get('accuracy') * 100:.2f}%")
    print(f"F1 Macro : {meta.get('f1_macro'):.4f}")
    print(f"Classes  : {len(meta.get('classes', []))} crop categories")

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    sample = df.sample(n=5, random_state=42)

    print("\n[Sample Predictions]")
    for idx, row in sample.iterrows():
        input_data = pd.DataFrame([[row[c] for c in FEATURE_NAMES]], columns=FEATURE_NAMES)
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        conf = float(np.max(proba)) * 100
        actual = row["label"]
        match = "MATCH" if pred == actual else "MISMATCH"
        print(f"Row {idx:4d} | Actual: {actual:12s} | Pred: {pred:12s} | Conf: {conf:5.1f}% | {match}")

if __name__ == "__main__":
    evaluate()
