import pandas as pd
import os
import pickle
from fastapi import FastAPI
from app.schemas import LoanApplication
from app.predict import predict_default_risk
from app.transform import cat_columns_to_category
from catboost import Pool
# Loan Default Risk API – Overview
# This FastAPI application provides a simple REST API for predicting the risk of loan 
# default based on user-submitted application data. 
# It exposes two endpoints:

base_dir = os.path.dirname(__file__)
cat_col_path = os.path.join(base_dir, 'app', "list_cat_columns.pkl")
num_col_path = os.path.join(base_dir, 'app', "list_num_columns.pkl")

print("💾 Load cat columns")
with open(cat_col_path, "rb") as f:
    list_cat_columns = pickle.load(f)
print("✅ list_cat_columns:", list_cat_columns)

print("💾 Load num columns")
with open(num_col_path, "rb") as f:
    list_num_columns = pickle.load(f)
print("✅ list_num_columns:", list_num_columns)

if __name__ == "__main__":
    print("🚀 Starting test prediction...")
    sample = {
        "gender": "Male",
        "marital_status": "Single",
        "education_level": "Bachelor",
        "employment_status": "Employed",
        "loan_purpose": "Debt Consolidation",
        "grade_subgrade": "B2",
        "annual_income": 55000,
        "debt_to_income_ratio": 0.25,
        "credit_score": 720,
        "loan_amount": 12000,
        "interest_rate": 13.5
    }

    application = LoanApplication(**sample)
    input_df = pd.DataFrame([application.dict()])
    # --- Feature Engineering ---
    input_df["calc_ratio"] = input_df["loan_amount"] / input_df["annual_income"]
    
    print("🔄 Transform categorical columns")
    df_cat = cat_columns_to_category(input_df, list_cat_columns)
    print("✅ Transformed DataFrame:\n", df_cat)
    X_test_pool  = Pool(df_cat, cat_features=list_cat_columns)

    print("🔮 Predicting default risk")
    result = predict_default_risk(X_test_pool)
    print("✅ Prediction result:", result)


app = FastAPI()
# GET /
# Purpose:
# Health check endpoint to confirm that the API is running.
@app.get("/")
def read_root():
    return {"message": "Loan Default Risk API is running."}

# POST /predict
# Purpose:
# Accepts a loan application in JSON format and returns a prediction about the default risk.

# Request Body:
# An object matching the LoanApplication schema, defined in app.schemas.
@app.post("/predict")
def predict(application: LoanApplication):

    input_df = pd.DataFrame([application.dict()])
    input_df["calc_ratio"] = input_df["loan_amount"] / input_df["annual_income"]
    
    df_cat = cat_columns_to_category(input_df, list_cat_columns)
    
    X_test_pool  = Pool(df_cat, cat_features=list_cat_columns)
    result = predict_default_risk(X_test_pool)
    return result