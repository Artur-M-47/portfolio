import pandas as pd
import requests
import os

# Load test data
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "data", "test.csv")
try :
    df = pd.read_csv(data_path) 
except:
    print(f"❌ Failed to read data from {data_path}")
    exit(1)

# Transform df to dict (JSON)
json_data = df.to_dict(orient="records")
print(json_data[20])
for i, row in enumerate(json_data[20], start=1):
    print(f"🔹 Rekord {i}: {row}")

# Wyślij każdy rekord do API
for i, row in enumerate(json_data):
    response = requests.post("http://localhost:8000/predict", json=row)
    print(f"🔹 Rekord {i+1}: {row}")
    print(f"🔸 Odpowiedź API: {response.json()}")