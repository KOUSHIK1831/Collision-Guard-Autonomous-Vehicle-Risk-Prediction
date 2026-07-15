# 🚗 Collision Guard — Real-Time Collision Risk Prediction for Autonomous Vehicles

An end-to-end machine learning pipeline that predicts collision risk using synthetic driving telemetry, with multi-model comparison, hyperparameter tuning, and real-time risk simulation.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green)
![Status](https://img.shields.io/badge/status-academic%20research-lightgrey)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models & Results](#models--results)
- [Technical Details](#technical-details)
- [Potential Improvements](#potential-improvements)
- [License](#license)
- [Citation](#citation)

---

## Overview

This project builds and evaluates machine learning models to predict collision risk in autonomous vehicles using synthetic telemetry data. It addresses the challenge of severe class imbalance (4.8% collision rate) by applying SMOTE oversampling and comparing four classifiers to identify the best approach for safety-critical detection.

The pipeline covers:

- **Data Profiling** — Automated statistical summaries, class distribution analysis, missing value check
- **Preprocessing** — Stratified train/test split, SMOTE balancing, StandardScaler normalization
- **Feature Engineering** — Domain-specific interaction features (speed-distance ratio, brake-accel ratio, etc.)
- **Multi-Model Comparison** — Random Forest, Gradient Boosting, Logistic Regression, SVM evaluated on 5 metrics
- **Hyperparameter Tuning** — GridSearchCV on the best-performing model
- **Risk Simulation** — Real-time prediction on random sample points with probability scoring

---

## Features

- **Class Imbalance Handling** — SMOTE oversampling applied only to training data to prevent data leakage
- **No Data Leakage** — SMOTE, scaling, and feature engineering all fit on training data only
- **Multi-Metric Evaluation** — Accuracy, precision, recall, F1-score, and ROC AUC for all models
- **Safety-Critical Focus** — Recall prioritized as the key metric; missing a collision is costlier than a false alarm
- **Hyperparameter Tuning** — GridSearchCV with 5-fold cross-validation on the best model
- **Real-Time Simulation** — Random sampling of data points with risk probability prediction and visualization

---

## Dataset

Synthetic dataset with 500 observations of driving telemetry.

| Column | Type | Description |
|---|---|---|
| `time` | float | Timestamp (0–100) |
| `speed` | float | Vehicle speed |
| `steer_angle` | float | Steering angle |
| `accel` | float | Acceleration |
| `distance_to_obj` | float | Distance to nearest object |
| `rel_speed` | float | Relative speed to object |
| `brake` | float | Braking intensity |
| `weather` | int | Weather severity (0, 1, 2) |
| `road_type` | int | Road type category (0, 1, 2) |
| `risk` | int | **Target** — 0: no risk, 1: collision risk |

**Class distribution:** 476 no-risk (95.2%) — 24 collision-risk (4.8%)

Engineered features (created during preprocessing):

| Feature | Formula |
|---|---|
| `speed_distance_ratio` | speed / (distance_to_obj + 1e-5) |
| `brake_accel_ratio` | brake / (abs(accel) + 1e-5) |
| `rel_speed_distance` | rel_speed × distance_to_obj |
| `speed_steer_interaction` | speed × abs(steer_angle) |

---

## Project Structure

```
.
├── Collision_Risk_Prediction.ipynb   # Main notebook (11 code cells)
├── collision_risk_dataset.csv        # Synthetic dataset (500 rows, 10 columns)
└── README.md                         # Project documentation
```

---

## Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Collision-Guard-Autonomous-Vehicle-Risk-Prediction.git
cd Collision-Guard-Autonomous-Vehicle-Risk-Prediction

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/macOS
# venv\Scripts\activate     # Windows

# Install dependencies
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter

# Launch the notebook
jupyter notebook Collision_Risk_Prediction.ipynb
```

---

## Usage

1. **Open the notebook** — Launch `Collision_Risk_Prediction.ipynb`
2. **Run cells sequentially** — The notebook is organized into 11 code cells:
    - Cell 1: Library imports and plot styling
    - Cell 2: Data loading and initial profiling (info, describe, class distribution)
    - Cell 3: Exploratory data analysis (class distribution plot, correlation heatmap, feature distributions, risk time series)
    - Cell 4: Train/test split, SMOTE balancing, StandardScaler normalization
    - Cell 5: Feature engineering (4 interaction features)
    - Cell 6: Model training and comparison (4 models, 5 metrics)
    - Cell 7: Best model evaluation (confusion matrix, classification report, ROC curve)
    - Cell 8: Feature importance / coefficient analysis
    - Cell 9: Hyperparameter tuning with GridSearchCV
    - Cell 10: Real-time risk simulation and visualization
    - Cell 11: Model saving (joblib) and reusable `predict_collision_risk()` function

---

## Models & Results

Four classifiers were trained on the balanced dataset and evaluated on the original test set.

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Random Forest | 0.9533 | 0.0000 | 0.0000 | 0.0000 | 0.4486 |
| Gradient Boosting | 0.9533 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| **Logistic Regression** | 0.1733 | 0.0465 | **0.8571** | **0.0882** | 0.5415 |
| SVM | 0.9533 | 0.0000 | 0.0000 | 0.0000 | 0.6743 |

**Best model (by F1): Logistic Regression**

Random Forest, Gradient Boosting, and SVM default to predicting all zeros due to the severe class imbalance in the test set, yielding zero recall. Logistic Regression trades overall accuracy for the ability to detect collision events.

### Classification Report — Logistic Regression

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 (No risk) | 0.95 | 0.14 | 0.24 | 143 |
| 1 (Collision) | 0.05 | **0.86** | 0.09 | 7 |

### Hyperparameter Tuning

- **Best parameters:** `C=100`, `penalty=l1`, `solver=liblinear`
- **Tuned ROC AUC:** 0.5544
- **Tuned macro F1:** 0.34

### Feature Coefficients (Logistic Regression)

| Feature | Coefficient |
|---|---|
| `rel_speed_distance` | **+2.413** |
| `speed` | +0.062 |
| `brake_accel_ratio` | −0.097 |
| `speed_steer_interaction` | −0.156 |
| `weather` | −0.228 |
| `speed_distance_ratio` | −0.841 |
| `road_type` | −1.081 |
| `distance_to_obj` | −2.247 |
| `brake` | −3.162 |
| `rel_speed` | −3.250 |

The interaction feature `rel_speed_distance` is the strongest positive predictor. `rel_speed` and `brake` show the strongest negative association with collision risk.

---

## Technical Details

### Pipeline Architecture

```
┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐
│ Raw Data     │───▶│ Preprocessing    │───▶│ Feature           │
│ (500 rows    │    │ - stratified     │    │ Engineering       │
│  10 columns) │    │   split (70/30)  │    │ - speed_dist_ratio│
│              │    │ - SMOTE (333/333)│    │ - brake_accel_ratio│
│              │    │ - StandardScaler │    │ - rel_speed_dist  │
└──────────────┘    └──────────────────┘    │ - speed_steer_int │
                                            └────────┬──────────┘
                                                     ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Simulation       │◀───│ Hyperparameter   │◀───│ Model Comparison │
│ - Random sample  │    │ Tuning           │    │ - RF             │
│ - Risk prob plot │    │ - GridSearchCV   │    │ - GBM            │
│                   │    │ - 5-fold CV      │    │ - LR (best F1)   │
│                   │    │ - scoring: F1    │    │ - SVM            │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

### Key Design Decisions

- **Post-split SMOTE** — SMOTE is applied after `train_test_split` to prevent synthetic samples from leaking into the test set
- **F1-based model selection** — The best model is chosen by F1-score rather than accuracy to account for class imbalance
- **Isolated feature scaling** — `StandardScaler` is fit on training data and transformed on test data to prevent leakage
- **Safety-oriented metric** — Recall is highlighted as the primary metric for the final model since missing a collision has higher stakes than a false alarm
- **Deterministic splits** — `random_state=42` ensures reproducible train/test splits and SMOTE synthesis

---

## Potential Improvements

- Hyperparameter tuning for all models (not just the best one)
- Advanced models (XGBoost, LightGBM, neural networks)
- Temporal cross-validation to respect the time-series structure
- Cost-sensitive learning to penalize false negatives more heavily
- Synthetic data augmentation beyond SMOTE (GANs, variational autoencoders)
- SHAP / LIME explanations for model interpretability
- Web deployment (Streamlit / FastAPI) for real-time risk scoring
- Continuous drift monitoring for production readiness

---

## License

This project is provided for academic and educational purposes.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{collision-guard,
  author = {Koushik},
  title = {Collision Guard: Real-Time Collision Risk Prediction for Autonomous Vehicles},
  year = {2024},
  note = {Available as part of the project repository}
}
```

---

*Built with Python, scikit-learn, imbalanced-learn, and Jupyter.*
