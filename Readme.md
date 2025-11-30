# 🧠 Portfolio Overview
This portfolio showcases applied machine learning and AI application projects, ranging from competition-grade classification pipelines to production-ready API deployments.  
It also includes **business analytics and automation case studies**, demonstrating end-to-end data workflows and Power BI dashboards for real-world processes.  

---
## 1. 📊 Business Analytics & Automatisations (Power BI + Excel + Word + PDF)  
An end-to-end business automation project integrating document workflows, financial tracking, and interactive dashboards.  
The project demonstrates:  
⏹️ payroll extraction from PDFs,  
⏹️ cost aggregation from Excel,  
⏹️ automated contract generation (JSON + PDF),  
⏹️ dynamic Word templates with mail merge,  
⏹️ Power BI dashboards for contracts, rents, deposits, utilities, and payments,  
⏹️ CSV revenue tracking and bank statement reconciliation,  
⏹️ automated invoice proposals with alerts for unpaid invoices.  

Tech Stack:  
### Power BI, Python, Excel, Word, PDF Automation  

Highlights:  
- Complete **data flow architecture** illustrated via mind map.  
- Interactive **Power BI dashboard** with multiple pages (Costs, Revenues, Tenant Agreements, Invoice Management, Lump-Sum Settlement).  
- Demonstrates **business impact**: time efficiency, financial transparency, and scalable automation.  
  
### 📁 Business_Analytics_&_Automatisations/property_management_dashboard/  
---
## 1. 💰 Loan Default Risk API (FastAPI + CatBoost)
A production-grade FastAPI application designed to predict loan default risk using a trained CatBoost model. 
The project demonstrates:  
⏹️ end-to-end ML application development,  
⏹️ including model training,  
⏹️ data transformation,  
⏹️ API serving,  
⏹️ simulated testing 

Prediction Performance: ROC AUC = 0.922  

Tech Stack:  
### Python, FastAPI, CatBoost, Pydantic, Pandas, Docker-ready

Highlights:  
- Built and deployed a **FastAPI REST API** with `/predict` endpoint.  
- Integrated **CatBoostClassifier** for loan default risk estimation.  
- Used **Pydantic schemas** for JSON validation and structured input/output.  
- Modular design with separate layers for schema, transformation, and prediction.  
- Simulated API testing with batch requests from `test.csv`.  
- Deployment-ready structure (conda environment, Docker-friendly).  

📊 Example Response:
```json
{
  "default_risk": "low",
  "probability": 0.92
}
```
### 📁 kaggle_competitions/Predicting_Loan_Payback/
---
## 2. 🍄 Mushroom Edibility Classification (Kaggle: PS S4E8)
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

## 3. 🚗 Insurance Marketing Campaigns with Predictive Modeling (Kaggle: PS S4E7)
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

## 4. 📡 Telecom KPI Anomaly Detection (Isolation Forest)
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