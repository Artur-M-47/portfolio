# 🧠 Portfolio Overview
This portfolio showcases two applied machine learning projects focused on classification and anomaly detection, combining regulatory awareness with technical precision.

## 1. 🍄 Mushroom Edibility Classification (Kaggle: PS S4E8)
A competition-grade pipeline for classifying mushrooms as edible or poisonous using 22 categorical features. The project compares Logistic Regression, a Tuned CatBoost classifier, and a Deep Neural Network (DNN), achieving:

Accuracy: 0.98481

Leaderboard Rank: 555 / 2422

Tech stack: Python, CatBoost, Keras, Scikit-learn

Highlights:

Advanced preprocessing with One-Hot Encoding (107 features)

Feature selection via Chi-Square Test

Model comparison with MCC scoring

Reproducible environment via Conda

### 📁 kaggle_competitions/Binary_Prediction_of_Poisonous_Mushrooms/

## 2. 📡 Telecom KPI Anomaly Detection (Isolation Forest)
An unsupervised anomaly detection pipeline for synthetic telecom KPI time series (RTT and SR). The project includes data fabrication, contamination tuning, and visual validation.

Model: Isolation Forest

Detected anomalies:

Success Rate: 685

Round-Trip Time: 13

Tech stack: Python, Scikit-learn, Pandas, Matplotlib

Highlights:

Synthetic KPI generation

Visual anomaly tagging

Contamination-level tuning

Modular notebooks for fabrication and detection

### 📁 telecom_KPI_anomaly_IForest_project/