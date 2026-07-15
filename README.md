# 🚗 Collision Guard – Real-Time Collision Risk Prediction for Autonomous Vehicles

A Machine Learning Approach to Enhance Autonomous Driving Safety

This project presents a machine learning-based system that estimates real-time collision risks in autonomous vehicles (AVs) using synthetic telemetry data. The goal is to enhance AV safety by providing early alerts or triggering preventive actions based on risk assessments derived from vehicle dynamics.

This project was developed as part of an academic ML research initiative focusing on synthetic driving data, feature engineering, and risk prediction models.

## 🎯 Project Objective

To build a collision risk prediction system that can:
- Analyze driving behavior and environmental conditions
- Predict the probability of a collision
- Support autonomous vehicles in making safer navigation decisions

## 📌 Overview

The rapid development of autonomous vehicles promises to revolutionize transportation, offering benefits such as increased safety, reduced traffic congestion, and improved fuel efficiency. However, ensuring the safety of AVs is paramount, with collision risk prediction being a critical component. This project addresses the challenge of accurately assessing collision probabilities by building a robust predictive model. The synthetic dataset used for training and evaluation is designed to mimic the complexities and nuances of actual driving data, allowing for effective model development in a controlled environment.

## 💡 Features

- **Collision Risk Prediction:** Develops and evaluates multiple machine learning models to predict the probability of a collision in real-time.
- **Synthetic Data Utilization:** Leverages a synthetic dataset specifically generated to replicate real-world autonomous vehicle telemetry data, addressing data privacy concerns associated with proprietary or sensitive driving information.
- **Comprehensive Model Evaluation:** Compares the performance of various machine learning algorithms, including Random Forest, Gradient Boosting, Logistic Regression, and Support Vector Machines (SVM), based on a range of metrics such as accuracy, precision, recall, F1-score, and ROC AUC.
- **Safety-Critical Focus:** Prioritizes recall as a key metric for model selection, emphasizing the importance of identifying actual collision risks to enhance AV safety.
- **Hyperparameter Tuning:** Uses GridSearchCV to optimize the best-performing model for improved results.
- **Real-Time Simulation:** Includes a function to simulate real-time collision risk predictions on sample data points.

## 🧠 How It Works

### 1. Data Collection

We use synthetic driving data including:
- Vehicle speed & acceleration
- Distance to nearby objects
- Traffic density
- Weather conditions
- Road type
- Time and location data

### 2. Exploratory Data Analysis (EDA)

EDA revealed important insights:
- Class imbalance in collision vs non-collision cases
- Strong feature correlations affecting model behavior
- Risk trends over time
- Outliers and unusual driving events
- Feature distributions guiding model selection

### 3. Data Preprocessing

The following steps were applied:
- Removed unnecessary columns (e.g., time)
- Isolated the target variable (collision risk)
- Detected class imbalance (4.8% collision cases)
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) to balance the dataset

### 4. Feature Engineering

Features include:
- Speed
- Acceleration
- Distance to other vehicles
- Weather severity
- Traffic density
- Road conditions
- Time of day
- **Engineered features:** speed_distance_ratio, brake_accel_ratio, rel_speed_distance, speed_steer_interaction

### 5. Models Used

Four models were trained and compared:
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **Logistic Regression**
- **Support Vector Machine (SVM)**

### 6. Model Evaluation

Models were evaluated using precision, recall, F1-score, and ROC AUC.

## 🧾 Dataset

- **Type**: Synthetic
- **Source**: Generated to simulate real-world autonomous vehicle telemetry data
- **Purpose**: Bypass data privacy issues associated with proprietary or sensitive driving datasets
- **Size**: 500 rows, 10 columns
- **Class Distribution**: 476 no-risk (0), 24 collision-risk (1) — 4.8% positive class

### Key Features in the Dataset:

- Distance to objects
- Relative speed
- Braking intensity
- Steering angle
- Vehicle speed
- Speed-to-distance ratio (engineered)
- Time to collision (engineered)

## 🧠 Models Evaluated

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|:---|---:|---:|---:|---:|---:|
| Random Forest | 0.9533 | 0.0000 | 0.0000 | 0.0000 | 0.4486 |
| Gradient Boosting | 0.9533 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression | 0.1733 | 0.0465 | 0.8571 | 0.0882 | 0.5415 |
| SVM | 0.9533 | 0.0000 | 0.0000 | 0.0000 | 0.6743 |

> ✅ **Best Performing Model (in Recall): Logistic Regression**

Despite the lower overall accuracy of Logistic Regression, it achieved a **recall of 0.8571**, indicating its effectiveness in identifying high-risk collision events — essential for safety-critical applications where missing a risk is more costly than issuing a false alarm. This model's strong recall is crucial for proactive risk management in AVs.

After hyperparameter tuning with GridSearchCV, the best parameters were `{'C': 100, 'penalty': 'l1', 'solver': 'liblinear'}`, improving accuracy to 0.43 and F1-score to 0.10.

## 🔍 Key Collision Risk Drivers

Based on Logistic Regression coefficients, the most influential features were:
- **Positive correlation:** rel_speed_distance, speed
- **Negative correlation:** rel_speed, brake, distance_to_obj, road_type

## ⚙️ Project Structure

```
.
├── Collision_Risk_Prediction.ipynb   # Main Jupyter notebook with preprocessing, modeling, and evaluation
├── collision_risk_dataset.csv        # Synthetic dataset used for model training and testing
├── README.md                         # Project overview and documentation
```

## 🛠️ Technologies and Libraries

- **Python**
- **Pandas, NumPy** — Data manipulation
- **Scikit-learn** — ML models (Logistic Regression, SVM, Random Forest, Gradient Boosting), preprocessing, metrics
- **SMOTE (imbalanced-learn)** — Class imbalance handling
- **Matplotlib / Seaborn** — EDA & Visualization
- **Jupyter Notebook** — Interactive development

Dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`, `jupyter`

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
   ```

4. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook Collision_Risk_Prediction.ipynb
   ```

5. **Run all cells** — The notebook guides you through data loading, preprocessing, model training, evaluation, and result presentation.

## 📈 Why This Project Matters

Collision Guard demonstrates how AI + data can:
- Predict dangerous driving situations
- Reduce accidents
- Improve trust in autonomous vehicles
- Enable safer urban mobility
