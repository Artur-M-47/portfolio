import pickle
import numpy as np
from .schemas import LoanApplication 

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_default_risk(application: LoanApplication):
    """
    Transforms input data and returns default risk prediction.
    """
    employment = 1 if application.employment_status.lower() == "employed" else 0
    features = np.array([[application.income, application.loan_amount, employment]])
    probability = model.predict_proba(features)[0][1]
    risk = "high" if probability > 0.5 else "low"

    return {
        "default_risk": risk,
        "probability": round(probability, 2)
    }