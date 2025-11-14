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
    #employment = 1 if application.employment_status.lower() == "employed" else 0
    #features = np.array([[application.income, application.loan_amount, employment]])
    # features = np.array(df) #application
    probability = model.predict_proba(X_test_pool)[0][1]
    risk = "high" if probability > 0.5 else "low"

    return {
        "default_risk": risk,
        "probability": round(probability, 2)
    }