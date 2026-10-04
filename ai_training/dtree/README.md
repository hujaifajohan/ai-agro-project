# AgroAI Decision Tree Crop Recommendation Model

## Overview
This module implements a Decision Tree Classifier trained on soil and environmental parameters to recommend optimal crop species for agricultural land parcels.

## Training Dataset
- **Location**: `ai_training/data/Crop_recommendation.csv`
- **Total Samples**: 2,200 rows
- **Features**:
  1. `N` (Nitrogen ratio in soil)
  2. `P` (Phosphorus ratio in soil)
  3. `K` (Potassium ratio in soil)
  4. `temperature` (Ambient temperature in °C)
  5. `humidity` (Relative humidity in %)
  6. `ph` (Soil pH value)
  7. `rainfall` (Rainfall in mm)
- **Target**: `label` (22 unique crop classes)

## Training Command
```bash
python ai_training/dtree/train_crop_model.py
```

## Model Artifacts
- Model File: `backend/ai/dtree/crop_recommendation_model.pkl`
- Metadata File: `backend/ai/dtree/model_metadata.json`

## Model Evaluation Metrics
- **Accuracy**: 98.86%
- **Macro Precision**: 0.9894
- **Macro Recall**: 0.9886
- **Macro F1-Score**: 0.9885
