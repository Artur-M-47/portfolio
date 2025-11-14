
## 💰 Loan
Your Goal: Predict the probability that a borrower will pay back their loan.
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

Accuracy: 0.98481  
Tech stack:  

### Python, Pandas, NumPy, CatBoost, Keras, Scikit-learn 

Highlights:

- Used ML models : **Deep Neural Network (DNN)**,**Tuned CatBoost classifier**, **Logistic Regression**
- Advanced preprocessing with **One-Hot Encoding** (107 features)
- Feature selection via **Chi-Square Test**  
- Model comparison with MCC scoring  
- Reproducible environment via Conda  

### 📁 kaggle_competitions/Predicting Loan Payback/  
├── app/                          # główny folder aplikacji  
│   ├── __init__.py              # pusty plik inicjalizujący pakiet  
│   ├── main.py                  # FastAPI app  
│   ├── schemas.py               # definicja danych wejściowych (Pydantic)  
│   ├── predict.py               # logika predykcji ML  
│   └── model.pkl                # zapisany model ML  
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