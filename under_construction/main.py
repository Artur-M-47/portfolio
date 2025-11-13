from fastapi import FastAPI
from app.schemas import LoanApplication
from app.predict import predict_default_risk

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Loan Default Risk API is running."}

@app.post("/predict")
def predict(application: LoanApplication):
    result = predict_default_risk(application)
    return result