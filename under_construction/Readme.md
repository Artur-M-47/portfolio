
## 💰 Loan
🧠 Loan Default Risk API – AI-Ready FastAPI App with CatBoost
This project demonstrates a production-grade FastAPI application for predicting loan default risk using a trained CatBoost model. It was built in response to a real-world AI engineering opportunity involving agentic architecture, RAG, and end-to-end deployment.

Your Goal: Predict the probability that a borrower will pay back their loan.
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

Accuracy: 0.98481  
Tech stack:  

### Python, Pandas, NumPy, CatBoost, Keras, Scikit-learn 

Highlights:

    🚀 Key Features
    FastAPI REST API with /predict endpoint

    Pydantic schema for structured JSON input

    CatBoost model integration with proper handling of categorical features via Pool

    Modular architecture with separate files for schema, transformation, and prediction logic

    Ready for CI/CD and containerization

    Designed for extension into LangChain, RAG, or Vector DB pipelines

    🧩 Tech Stack
    Layer	            Tools Used
    API Framework	    FastAPI
    Model Serving	    CatBoostClassifier
    Data Validation	    Pydantic
    Data Transformation	pandas, pickle
    Deployment Ready	Modular Python, Docker-friendly
    Future Extensions	LangChain, LangGraph, FAISS-ready


POST /predict

{
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

✅ Response
{
  "default_risk": "high",
  "probability": 0.92
}


### 📁 kaggle_competitions/Predicting Loan Payback/  
├── app/                          # główny folder aplikacji  
│   ├── __init__.py              # pusty plik inicjalizujący pakiet  
│   ├── model.pkl
│   ├── list_cat_columns.pkl
│   ├── list_num_columns.pkl
│   ├── schemas.py
│   ├── transform.py
│   └── predict.py
│  
├── data/                        # folder z danymi treningowymi  
│   └── loan_data.csv            # pobrany z Kaggle lub innego źródła  
│  
├── train_model.py              # skrypt do trenowania i zapisu modelu  
├── requirements.txt            # lista bibliotek  
├── README.md                   # dokumentacja projektu  
└── .vscode/                    # konfiguracja VS Code (opcjonalnie)  
    └── settings.json           # ustawienie interpretera Conda  


    Muszę wejsc w anaconda prompt po aktywowaniu środowiska loan_env do katalogu z projektem chwilowo under_construction  
    wpisuję komendę uruchamiającą serwer   
    uvicorn main:app --reload  
    wchodzisz na stronę w wyszukiwarce http://127.0.0.1:8000/docs#/     FAST API  

    w trakcie pracy nad kodem należy "ctrl + s" na serwerze dzięki dzięki --reload zmiany się odrazu aktualizują  

🛠️ How to Run Locally
bash
# Create virtual environment
conda create -n loan_env python=3.10
conda activate loan_env

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn main:app --reload
📌 Notes
This project is designed to be extended with LangChain agents, RAG pipelines, and vector search.

All model artifacts are preloaded and versioned for reproducibility.

Built as a response to a Randstad Digital AI Engineering opportunity.