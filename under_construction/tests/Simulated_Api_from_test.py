"""
📊 Simulated API Test Client

This script loads test data from a CSV file and sends each record as a POST request
to a running FastAPI server at http://localhost:8000/predict. It is used to simulate
real-world API usage and validate the model's predictions for multiple input samples.

Usage:
1. Make sure the FastAPI server is running:
    go to the project root directory and run:
    uvicorn main:app --reload

2. Ensure the test data file exists at: ../data/test.csv

3. Run this script from the 'tests/' directory:
   python Simulated_Api_from_test.py

Each record is printed before sending, and the API response is displayed after.
This helps verify model behavior and debug input/output formats.

Author: Artur Makowski
Date: 2025-11-14
"""


import pandas as pd
import requests
import os

# Load test data
base_dir = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(base_dir, "data", "test.csv")
try :
    df = pd.read_csv(data_path) 
except:
    print(f"❌ Failed to read data from {data_path}")
    exit(1)

# Transform df to dict (JSON)
json_data = df.to_dict(orient="records")
print(json_data[20])

# Sending each record from test.csv to FastAPI endpoint for prediction
for i, row in enumerate(json_data):
    response = requests.post("http://localhost:8000/predict", json=row) 
    print(f"⬆️📤 Sending POST request {i+1}: {row}")
    print(f"⬇️📥 Receiving Response API: {response.json()}")