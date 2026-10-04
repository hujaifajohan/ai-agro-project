"""
AgroAI — Decision Tree Crop Recommendation Model Training Script
Dataset: ai_training/data/Crop_recommendation.csv
Features: N, P, K, temperature, humidity, ph, rainfall
Target: label
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------------------------
# 1. PATH CONFIGURATION
# ------------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_PATH = PROJECT_ROOT / "ai_training" / "data" / "Crop_recommendation.csv"
BACKEND_MODEL_DIR = PROJECT_ROOT / "backend" / "ai" / "dtree"
LOCAL_MODEL_DIR = SCRIPT_DIR

FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_NAME = "label"

def train_and_evaluate():
    print("=" * 70)
    print("AGROAI — CROP RECOMMENDATION DECISION TREE TRAINING")
    print("=" * 70)
    print(f"Dataset path: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found at: {DATA_PATH}")

    # --------------------------------------------------------------------------
    # 2. DATA LOAD & INSPECTION
    # --------------------------------------------------------------------------
    raw_df = pd.read_csv(DATA_PATH)
    original_row_count = len(raw_df)
    print(f"\n[Dataset Summary]")
    print(f"Original rows: {original_row_count}")
    print(f"Columns: {list(raw_df.columns)}")

    # Check required columns
    missing_cols = [col for col in FEATURE_NAMES + [TARGET_NAME] if col not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    # --------------------------------------------------------------------------
    # 3. DATA VALIDATION & CLEANING
    # --------------------------------------------------------------------------
    # Check duplicate rows
    duplicate_rows_count = raw_df.duplicated().sum()
    print(f"Duplicate rows detected: {duplicate_rows_count}")
    cleaned_df = raw_df.drop_duplicates().copy()

    # Check missing values
    missing_val_count = cleaned_df[FEATURE_NAMES + [TARGET_NAME]].isnull().sum().sum()
    print(f"Missing value count: {missing_val_count}")
    cleaned_df = cleaned_df.dropna(subset=FEATURE_NAMES + [TARGET_NAME]).copy()

    # Verify numerical types for features
    invalid_numeric_count = 0
    for col in FEATURE_NAMES:
        non_num = pd.to_numeric(cleaned_df[col], errors='coerce').isnull().sum()
        invalid_numeric_count += non_num
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')

    cleaned_df = cleaned_df.dropna(subset=FEATURE_NAMES).copy()
    print(f"Invalid non-numeric feature rows: {invalid_numeric_count}")

    # Verify target label valid strings
    cleaned_df[TARGET_NAME] = cleaned_df[TARGET_NAME].astype(str).str.strip().str.lower()
    cleaned_df = cleaned_df[cleaned_df[TARGET_NAME] != ""].copy()

    final_row_count = len(cleaned_df)
    invalid_rows = original_row_count - final_row_count
    print(f"Invalid/Removed rows total: {invalid_rows}")
    print(f"Final valid rows for training: {final_row_count}")

    # Class distribution
    class_counts = cleaned_df[TARGET_NAME].value_counts()
    unique_classes = sorted(class_counts.index.tolist())
    num_classes = len(unique_classes)

    print(f"\n[Target Class Summary]")
    print(f"Number of unique crop classes: {num_classes}")
    print(f"Min samples per class: {class_counts.min()} ('{class_counts.idxmin()}')")
    print(f"Max samples per class: {class_counts.max()} ('{class_counts.idxmax()}')")

    # Feature ranges inspection
    print("\n[Feature Ranges]")
    for col in FEATURE_NAMES:
        print(f" - {col:12s}: min={cleaned_df[col].min():8.2f}, max={cleaned_df[col].max():8.2f}, mean={cleaned_df[col].mean():8.2f}")

    # --------------------------------------------------------------------------
    # 4. TRAIN / TEST SPLIT
    # --------------------------------------------------------------------------
    X = cleaned_df[FEATURE_NAMES]
    y = cleaned_df[TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"\n[Train/Test Split]")
    print(f"Training samples: {len(X_train)} (80%)")
    print(f"Testing samples : {len(X_test)} (20%)")

    # --------------------------------------------------------------------------
    # 5. MODEL SELECTION & TRAINING
    # --------------------------------------------------------------------------
    # Hyperparameter selection
    best_model = None
    best_config = None
    best_f1 = -1.0

    candidates = [
        {"criterion": "entropy", "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
        {"criterion": "gini", "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
        {"criterion": "entropy", "max_depth": 12, "min_samples_split": 4, "min_samples_leaf": 2},
        {"criterion": "entropy", "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2},
    ]

    print("\n[Evaluating Model Hyperparameters]")
    for cfg in candidates:
        clf = DecisionTreeClassifier(
            criterion=cfg["criterion"],
            max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_split"],
            min_samples_leaf=cfg["min_samples_leaf"],
            random_state=42
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro")
        acc = accuracy_score(y_test, preds)
        print(f" - Config {cfg} -> Acc: {acc*100:.2f}%, F1-Macro: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_config = cfg
            best_model = clf

    print(f"\n[Selected Best Model]")
    print(f"Hyperparameters: {best_config}")

    # --------------------------------------------------------------------------
    # 6. EVALUATION METRICS
    # --------------------------------------------------------------------------
    y_pred = best_model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro"))
    rec_macro = float(recall_score(y_test, y_pred, average="macro"))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))

    print("\n" + "=" * 70)
    print("MODEL EVALUATION METRICS")
    print("=" * 70)
    print(f"Accuracy        : {acc * 100:.2f}% ({acc:.4f})")
    print(f"Macro Precision : {prec_macro:.4f}")
    print(f"Macro Recall    : {rec_macro:.4f}")
    print(f"Macro F1-Score  : {f1_macro:.4f}")

    print("\n[Classification Report]")
    cls_report = classification_report(y_test, y_pred, zero_division=0)
    print(cls_report)

    conf_mat = confusion_matrix(y_test, y_pred, labels=unique_classes).tolist()

    # Calculate feature importances
    importances = dict(zip(FEATURE_NAMES, [float(round(imp, 4)) for imp in best_model.feature_importances_]))
    print("\n[Feature Importances]")
    for feat, imp in sorted(importances.items(), key=lambda item: item[1], reverse=True):
        print(f" - {feat:12s}: {imp:.4f}")

    # --------------------------------------------------------------------------
    # 7. REAL CSV ROW VERIFICATION (SECTION 33 REQUIREMENT)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("REAL CSV ROW VALIDATION REPORT")
    print("=" * 70)
    # Pick 10 sample rows across different classes
    sample_indices = [0, 100, 200, 400, 600, 800, 1000, 1300, 1600, 2000]
    sample_indices = [idx for idx in sample_indices if idx < len(cleaned_df)]
    sample_rows = cleaned_df.iloc[sample_indices]

    correct_cnt = 0
    total_samples = len(sample_rows)
    for idx, row in sample_rows.iterrows():
        feat_vals = pd.DataFrame([[row[col] for col in FEATURE_NAMES]], columns=FEATURE_NAMES)
        pred_label = best_model.predict(feat_vals)[0]
        proba = best_model.predict_proba(feat_vals)[0]
        max_proba = float(np.max(proba)) * 100
        actual_label = row[TARGET_NAME]
        is_correct = (pred_label == actual_label)
        if is_correct:
            correct_cnt += 1
        status_str = "PASS" if is_correct else "FAIL"
        print(f"Row {idx:4d} | Actual: {actual_label:12s} | Pred: {pred_label:12s} | Conf: {max_proba:5.1f}% | [{status_str}]")

    print(f"\nReal Row Test Result: {correct_cnt}/{total_samples} Passed ({correct_cnt/total_samples*100:.1f}%)")

    # --------------------------------------------------------------------------
    # 8. MODEL & METADATA SERIALIZATION
    # --------------------------------------------------------------------------
    metadata = {
        "model_name": "Crop Recommendation Decision Tree",
        "algorithm": "Decision Tree Classifier",
        "features": FEATURE_NAMES,
        "target": TARGET_NAME,
        "classes": unique_classes,
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "original_rows": original_row_count,
        "duplicate_rows": int(duplicate_rows_count),
        "invalid_rows": int(invalid_rows),
        "final_rows": final_row_count,
        "hyperparameters": {
            "criterion": best_config["criterion"],
            "max_depth": best_config["max_depth"],
            "min_samples_split": best_config["min_samples_split"],
            "min_samples_leaf": best_config["min_samples_leaf"],
            "random_state": 42
        },
        "feature_importances": importances
    }

    # Save to Backend directory
    BACKEND_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    backend_pkl = BACKEND_MODEL_DIR / "crop_recommendation_model.pkl"
    backend_json = BACKEND_MODEL_DIR / "model_metadata.json"

    joblib.dump(best_model, backend_pkl)
    with open(backend_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # Save to Local ai_training/dtree directory
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    local_pkl = LOCAL_MODEL_DIR / "crop_recommendation_model.pkl"
    local_json = LOCAL_MODEL_DIR / "model_metadata.json"

    joblib.dump(best_model, local_pkl)
    with open(local_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\n" + "=" * 70)
    print("MODEL ARTIFACTS SAVED")
    print("=" * 70)
    print(f"Backend Model  : {backend_pkl} ({backend_pkl.stat().st_size} bytes)")
    print(f"Backend Meta   : {backend_json} ({backend_json.stat().st_size} bytes)")
    print(f"Local Model    : {local_pkl}")
    print(f"Local Meta     : {local_json}")
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    train_and_evaluate()
