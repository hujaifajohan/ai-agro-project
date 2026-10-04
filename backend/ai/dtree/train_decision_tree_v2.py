"""
AgroAI Decision Tree V2 Training Script
Trains separate Decision Tree models for Water Need and Field Status prediction
using the uploaded agroai_labeled_dataset_v2.csv dataset.
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
import joblib

import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error
)

# Ordinal mapping for Water Need diagnostic MAE
WATER_NEED_ORDINAL_MAP = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
    "Urgent": 3
}

def resolve_paths():
    """
    Returns absolute paths relative to project root or current directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Look up to find AgroAI project root
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    
    csv_path = os.path.join(project_root, "backend", "data", "agroai_labeled_dataset_v2.csv")
    if not os.path.exists(csv_path):
        # Fallback to current working directory relative path
        csv_path = os.path.abspath("backend/data/agroai_labeled_dataset_v2.csv")
        
    model_dir = os.path.join(project_root, "backend", "models", "decision_tree_v2")
    if not os.path.exists(os.path.dirname(model_dir)):
        model_dir = os.path.abspath("backend/models/decision_tree_v2")
        
    return csv_path, model_dir

def load_and_validate_dataset(csv_path):
    print(f"Loading dataset from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    required_cols = [
        "N", "P", "K", "temperature", "humidity", "ph", "rainfall",
        "label", "Water Need", "Status", "Soil Moisture"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in V2 dataset: {col}")
            
    print(f"Dataset successfully loaded. Total rows: {len(df)}, Total columns: {len(df.columns)}")
    return df

def build_pipeline():
    numeric_features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "Soil Moisture"]
    categorical_features = ["crop"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])

    return pipeline, numeric_features, categorical_features

def evaluate_baselines(X_train, y_train, X_test, y_test, target_name):
    print(f"\n--- Baseline Evaluation for {target_name} ---")
    # 1. Majority Class Baseline
    dummy_majority = DummyClassifier(strategy="most_frequent")
    dummy_majority.fit(X_train, y_train)
    y_pred_majority = dummy_majority.predict(X_test)
    
    maj_acc = accuracy_score(y_test, y_pred_majority)
    maj_macro_f1 = f1_score(y_test, y_pred_majority, average="macro", zero_division=0)
    maj_weighted_f1 = f1_score(y_test, y_pred_majority, average="weighted", zero_division=0)
    
    print(f"Majority Class Baseline -> Accuracy: {maj_acc:.4f}, Macro F1: {maj_macro_f1:.4f}, Weighted F1: {maj_weighted_f1:.4f}")

    # 2. Crop-Majority Baseline
    df_tr = X_train.assign(target=y_train)
    crop_majority_map = df_tr.groupby("crop")["target"].agg(lambda x: x.mode()[0] if not x.empty else y_train.mode()[0]).to_dict()
    global_majority = y_train.mode()[0]
    
    y_pred_crop_maj = X_test["crop"].map(lambda c: crop_majority_map.get(c, global_majority))
    
    crop_maj_acc = accuracy_score(y_test, y_pred_crop_maj)
    crop_maj_macro_f1 = f1_score(y_test, y_pred_crop_maj, average="macro", zero_division=0)
    crop_maj_weighted_f1 = f1_score(y_test, y_pred_crop_maj, average="weighted", zero_division=0)
    
    print(f"Crop-Majority Baseline -> Accuracy: {crop_maj_acc:.4f}, Macro F1: {crop_maj_macro_f1:.4f}, Weighted F1: {crop_maj_weighted_f1:.4f}")
    
    return {
        "majority_baseline": {
            "accuracy": round(float(maj_acc), 4),
            "macro_f1": round(float(maj_macro_f1), 4),
            "weighted_f1": round(float(maj_weighted_f1), 4)
        },
        "crop_majority_baseline": {
            "accuracy": round(float(crop_maj_acc), 4),
            "macro_f1": round(float(crop_maj_macro_f1), 4),
            "weighted_f1": round(float(crop_maj_weighted_f1), 4)
        }
    }

def train_target_model(X_train, y_train, X_test, y_test, target_name):
    print(f"\n==================================================")
    print(f"TRAINING DECISION TREE MODEL: {target_name}")
    print(f"==================================================")
    
    # Class distribution in training and test sets
    print(f"Train class distribution:\n{y_train.value_counts()}")
    print(f"Test class distribution:\n{y_test.value_counts()}")

    # Baselines
    baselines = evaluate_baselines(X_train, y_train, X_test, y_test, target_name)

    base_pipeline, num_cols, cat_cols = build_pipeline()

    # Hyperparameter Grid
    param_grid = {
        "classifier__criterion": ["gini", "entropy"],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 10],
        "classifier__min_samples_leaf": [1, 5],
        "classifier__class_weight": [None, "balanced"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_macro_f1 = grid_search.best_score_

    print(f"\nBest Hyperparameters for {target_name}: {best_params}")
    print(f"Best 5-Fold CV Macro F1: {best_cv_macro_f1:.4f}")

    # Evaluate on final Test Set
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)
    classes = list(best_pipeline.classes_)

    test_acc = accuracy_score(y_test, y_pred)
    test_macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    test_precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    test_recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)

    per_class_precision = precision_score(y_test, y_pred, average=None, labels=classes, zero_division=0)
    per_class_recall = recall_score(y_test, y_pred, average=None, labels=classes, zero_division=0)
    per_class_f1 = f1_score(y_test, y_pred, average=None, labels=classes, zero_division=0)

    per_class_metrics = {}
    for cls, p, r, f in zip(classes, per_class_precision, per_class_recall, per_class_f1):
        per_class_metrics[cls] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4)
        }

    cm = confusion_matrix(y_test, y_pred, labels=classes)

    print(f"\nTest Evaluation for {target_name}:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro F1: {test_macro_f1:.4f}")
    print(f"Weighted F1: {test_weighted_f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("\nConfusion Matrix (Rows=True, Cols=Predicted):\n", pd.DataFrame(cm, index=classes, columns=classes))

    # Ordinal MAE if Water Need
    ordinal_mae = None
    if target_name == "Water Need":
        y_test_ord = np.array([WATER_NEED_ORDINAL_MAP.get(c, 0) for c in y_test])
        y_pred_ord = np.array([WATER_NEED_ORDINAL_MAP.get(c, 0) for c in y_pred])
        ordinal_mae = float(mean_absolute_error(y_test_ord, y_pred_ord))
        print(f"Water Need Ordinal Mean Absolute Error (MAE): {ordinal_mae:.4f}")

    results = {
        "target_name": target_name,
        "best_params": {k.replace("classifier__", ""): v for k, v in best_params.items()},
        "cv_metrics": {
            "best_cv_macro_f1": round(float(best_cv_macro_f1), 4)
        },
        "test_metrics": {
            "accuracy": round(float(test_acc), 4),
            "macro_f1": round(float(test_macro_f1), 4),
            "weighted_f1": round(float(test_weighted_f1), 4),
            "precision_macro": round(float(test_precision_macro), 4),
            "recall_macro": round(float(test_recall_macro), 4),
            "ordinal_mae": round(ordinal_mae, 4) if ordinal_mae is not None else None,
            "per_class": per_class_metrics
        },
        "confusion_matrix": {
            "classes": classes,
            "matrix": cm.tolist()
        },
        "baselines": baselines
    }

    return best_pipeline, results

def train_and_save_decision_tree_v2():
    csv_path, model_dir = resolve_paths()
    cm_dir = os.path.join(model_dir, "confusion_matrices")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)

    df = load_and_validate_dataset(csv_path)

    # Feature matrix X
    # Rename 'label' column to 'crop' internally for clear representation
    df_features = df.copy()
    df_features["crop"] = df_features["label"].astype(str)

    features = ["crop", "N", "P", "K", "temperature", "humidity", "ph", "rainfall", "Soil Moisture"]
    X = df_features[features]

    # Target 1: Water Need
    y_water = df_features["Water Need"].astype(str)
    # Target 2: Status
    y_status = df_features["Status"].astype(str)

    # Train/Test Split (80/20, random_state=42, stratified)
    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
        X, y_water, test_size=0.20, random_state=42, stratify=y_water
    )

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_status, test_size=0.20, random_state=42, stratify=y_status
    )

    # Train Water Need model
    water_pipeline, water_results = train_target_model(
        X_train_w, y_train_w, X_test_w, y_test_w, "Water Need"
    )

    # Train Status model
    status_pipeline, status_results = train_target_model(
        X_train_s, y_train_s, X_test_s, y_test_s, "Status"
    )

    # Save Pipeline artifacts
    water_model_path = os.path.join(model_dir, "water_need_pipeline.joblib")
    status_model_path = os.path.join(model_dir, "status_pipeline.joblib")

    joblib.dump(water_pipeline, water_model_path)
    joblib.dump(status_pipeline, status_model_path)
    print(f"\nSaved Water Need Pipeline to {water_model_path}")
    print(f"Saved Status Pipeline to {status_model_path}")

    # Feature Info
    feature_info = {
        "features": features,
        "numerical_features": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "Soil Moisture"],
        "categorical_features": ["crop"],
        "unique_crops": sorted(df["label"].unique().tolist()),
        "soil_moisture_provenance": "Training dataset agroai_labeled_dataset_v2.csv contains synthetic/derived Soil Moisture values."
    }

    feature_info_path = os.path.join(model_dir, "feature_info.json")
    with open(feature_info_path, "w", encoding="utf-8") as f:
        json.dump(feature_info, f, indent=2)

    # Confusion Matrices artifacts
    cm_water_path = os.path.join(cm_dir, "water_need_confusion_matrix.json")
    with open(cm_water_path, "w", encoding="utf-8") as f:
        json.dump(water_results["confusion_matrix"], f, indent=2)

    cm_status_path = os.path.join(cm_dir, "status_confusion_matrix.json")
    with open(cm_status_path, "w", encoding="utf-8") as f:
        json.dump(status_results["confusion_matrix"], f, indent=2)

    # Consolidated Metrics JSON
    metrics = {
        "water_need": water_results,
        "status": status_results
    }
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Detailed Metadata JSON
    metadata = {
        "model_version": "v2.0.0",
        "system_name": "AgroAI Decision Tree V2 System",
        "dataset_name": "agroai_labeled_dataset_v2.csv",
        "dataset_path": csv_path,
        "dataset_total_rows": len(df),
        "training_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "random_seed": 42,
        "scikit_learn_version": sklearn.__version__,
        "features": features,
        "targets": {
            "water_need": {
                "classes": list(water_pipeline.classes_),
                "class_distribution_dataset": y_water.value_counts().to_dict(),
                "selected_hyperparameters": water_results["best_params"],
                "best_cv_macro_f1": water_results["cv_metrics"]["best_cv_macro_f1"],
                "test_macro_f1": water_results["test_metrics"]["macro_f1"],
                "test_accuracy": water_results["test_metrics"]["accuracy"],
                "test_weighted_f1": water_results["test_metrics"]["weighted_f1"],
                "ordinal_mae": water_results["test_metrics"]["ordinal_mae"]
            },
            "status": {
                "classes": list(status_pipeline.classes_),
                "class_distribution_dataset": y_status.value_counts().to_dict(),
                "selected_hyperparameters": status_results["best_params"],
                "best_cv_macro_f1": status_results["cv_metrics"]["best_cv_macro_f1"],
                "test_macro_f1": status_results["test_metrics"]["macro_f1"],
                "test_accuracy": status_results["test_metrics"]["accuracy"],
                "test_weighted_f1": status_results["test_metrics"]["weighted_f1"]
            }
        },
        "soil_moisture_provenance": "Dataset V2 contains synthetic/derived Soil Moisture values.",
        "artifacts": [
            "water_need_pipeline.joblib",
            "status_pipeline.joblib",
            "metadata.json",
            "metrics.json",
            "feature_info.json",
            "confusion_matrices/water_need_confusion_matrix.json",
            "confusion_matrices/status_confusion_matrix.json"
        ]
    }

    metadata_path = os.path.join(model_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n==================================================")
    print("DECISION TREE V2 TRAINING & ARTIFACT GENERATION COMPLETE!")
    print(f"Artifacts saved in: {model_dir}")
    print("==================================================")

if __name__ == "__main__":
    train_and_save_decision_tree_v2()
