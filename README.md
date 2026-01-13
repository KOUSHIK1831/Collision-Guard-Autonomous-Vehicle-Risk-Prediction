🚗 Collision Guard – Autonomous Vehicle Risk Prediction

A Machine Learning Approach to Enhance Autonomous Driving Safety

Collision Guard is a machine learning–based system designed to predict collision risk in autonomous vehicles using driving telemetry, environmental data, and sensor-based features. The goal is to enable real-time risk assessment and proactive safety decisions for autonomous driving systems.

This project was developed as part of an academic ML research initiative focusing on synthetic driving data, feature engineering, and risk prediction models.

👥 Team Members

G. Agraz (22STUCHH010713)

K. Suradhya Reddy (22STUCHH010682)

B. Koushik Reddy (22STUCHH010819)

M. Charan Teja (22STUCHH010666) 

🎯 Project Objective

To build a collision risk prediction system that can:

Analyze driving behavior and environmental conditions

Predict the probability of a collision

Support autonomous vehicles in making safer navigation decisions

🧠 How It Works
1. Data Collection

We use synthetic and real-world inspired driving data including:

Vehicle speed & acceleration

Distance to nearby objects

Traffic density

Weather conditions

Road type

Time and location data

Data acquisition began in 2020 and expanded with multi-sensor telemetry (camera, radar, lidar) and diverse driving conditions.

2. Exploratory Data Analysis (EDA)

EDA revealed important insights:

Class imbalance in collision vs non-collision cases

Strong feature correlations affecting model behavior

Risk trends over time

Outliers and unusual driving events

Feature distributions guiding model selection 

3. Data Preprocessing

The following steps were applied:

Removed unnecessary columns (e.g., time)

Isolated the target variable (collision risk)

Detected class imbalance

Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset 

4. Feature Engineering

Features include:

Speed

Acceleration

Distance to other vehicles

Weather severity

Traffic density

Road conditions

Time of day

Feature engineering evolved from manual selection to advanced machine-learning-driven feature extraction.

5. Model Used

We trained a Random Forest Classifier to predict collision risk.

Why Random Forest?

Handles nonlinear data well

Works with mixed feature types

Resistant to overfitting

Provides feature importance for risk analysis 

6. Model Evaluation

The model was evaluated using:

Precision

Recall

F1-Score

This ensured the model performs reliably on unseen driving data and avoids bias toward the majority class.

🔍 Key Collision Risk Drivers

The most influential risk factors identified were:

Vehicle speed

Distance to nearby vehicles

Weather conditions

Road geometry

Traffic density

Machine learning models were used to quantify feature importance and improve real-time risk estimation.

🚀 Deployment Vision

Future deployment will integrate this model into:

Autonomous vehicle control systems

Traffic management platforms

Real-time risk dashboards

The goal is to provide live collision risk alerts and enable preventive actions before accidents occur.

🧪 Technologies Used

Python

Pandas, NumPy

Scikit-learn

Random Forest Classifier

SMOTE (imbalanced-learn)

Matplotlib / Seaborn (EDA & Visualization)

📈 Why This Project Matters

Collision Guard demonstrates how AI + data can:

Predict dangerous driving situations

Reduce accidents

Improve trust in autonomous vehicles

Enable safer urban mobility
