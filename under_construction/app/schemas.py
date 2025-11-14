from pydantic import BaseModel

class LoanApplication(BaseModel):
    gender: str
    marital_status: str
    education_level: str
    employment_status: str
    loan_purpose: str
    grade_subgrade: str
    annual_income: float
    debt_to_income_ratio: float
    credit_score: float
    loan_amount: float
    interest_rate: float


if __name__ == "__main__":
    # Test block to validate schema
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

    try:
        app = LoanApplication(**sample)
        print("Validation successful:", app)
    except Exception as e:
        print("Validation failed:", e)