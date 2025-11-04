# 🧠 Portfolio Overview
This portfolio showcases two applied machine learning projects focused on classification and anomaly detection, combining regulatory awareness with technical precision.

## 1. 🍄 Mushroom Edibility Classification (Kaggle: PS S4E8)
A competition-grade pipeline for classifying mushrooms as edible or poisonous using 22 categorical features. The project compares Logistic Regression, a Tuned CatBoost classifier, and a Deep Neural Network (DNN), achieving:

Accuracy: 0.98481  
Leaderboard Rank: 555 / 2422

Tech stack:  

### Python, Pandas, NumPy, CatBoost, Keras, Scikit-learn 

Highlights:

- Used ML models : **Deep Neural Network (DNN)**,**Tuned CatBoost classifier**, **Logistic Regression**
- Advanced preprocessing with **One-Hot Encoding** (107 features)
- Feature selection via **Chi-Square Test**
- Model comparison with MCC scoring
- Reproducible environment via Conda

### 📁 kaggle_competitions/Binary_Prediction_of_Poisonous_Mushrooms/

## 2. 🚗 Insurance Marketing Campaigns with Predictive Modeling (Kaggle: PS S4E7)
A binary classification challenge focused on predicting which clients will respond positively to an automobile insurance offer. The project utilized advanced ensemble methods and robust validation techniques to maximize prediction stability.  

ROC AUC Score: 0.89515  
Leaderboard Rank: 344 / 2234  

Tech Stack:  

### Python, Pandas, NumPy, CatBoost, Scikit-learn, xgboost, lightgbm

Highlights:  

- Used ML models : **XGBoost**,**Tuned CatBoost classifier**, **LightGBM**, **RandomForest**
- Tiered Modeling Strategy comparing four ensemble classifiers.  
- Hyperparameter Tuning with GPU acceleration for CatBoost.  
- Imbalance handling via scale_pos_weight.  
- Final submission secured using Stratified **K-Fold Ensembling** for variance reduction.  

### 📁 Insurance_Marketing_Campaigns/

## 3. 📡 Telecom KPI Anomaly Detection (Isolation Forest)
An unsupervised anomaly detection pipeline for synthetic telecom KPI time series (RTT and SR). The project includes data fabrication, contamination tuning, and visual validation.

Detected anomalies:  

- Success Rate: 685  
- Round-Trip Time: 13

Tech stack: 
### Python, Pandas, NumPy, Scikit-learn, Pandas, Matplotlib

Highlights:  

- Used ML models : **Isolation Forest**
- Synthetic KPI generation  
- Visual anomaly tagging  
- Contamination-level tuning  
- Modular notebooks for fabrication and detection  

### 📁 telecom_KPI_anomaly_IForest_project/