"""
📊 Prediction Module

This module loads the trained CatBoost model and provides a function
to predict loan default risk. It takes a CatBoost Pool object as input,
computes the probability of loan repayment (class 1), and returns
a simple risk classification ("low" or "high") along with the probability.
"""

import pickle
import numpy as np
import os
from .schemas import LoanApplication 

# Load the trained model
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "model.pkl")
print("model_path -> ",model_path)
with open(model_path , "rb") as f:
    model = pickle.load(f)

def predict_default_risk(X_test_pool):#application: LoanApplication
    """
    Transforms input data and returns default risk prediction.
    """
    probability = model.predict_proba(X_test_pool)[0][1]
    risk = "low" if probability < 0.5 else "high"

    return {
        "default_risk": risk,
        "probability": round(probability, 2)
    }