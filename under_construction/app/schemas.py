from pydantic import BaseModel

class LoanApplication(BaseModel):
    income: float
    loan_amount: float
    employment_status: str  # "Employed" or "Unemployed"


if __name__ == "__main__":
    # Test block to validate schema
    sample = {
        "income": 5000,
        "loan_amount": 15000,
        "employment_status": "Unemployed"
    }

    try:
        app = LoanApplication(**sample)
        print("Validation successful:", app)
    except Exception as e:
        print("Validation failed:", e)