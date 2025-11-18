import pandas as pd
import requests
import os

# Load test data
base_dir = os.path.dirname(os.path.dirname(__file__))  # upper level directory
data_path = os.path.join(base_dir, "data", "test.csv")
print(f"💾 Load test data from {data_path}")
try :
    df = pd.read_csv(data_path) 
except:
    print(f"❌ Failed to read data from {data_path}")
    exit(1)

# Transform df to dict (JSON)
json_data = df.to_dict(orient="records")
print(json_data[20])

# Send every record to API
for i, row in enumerate(json_data):
    response = requests.post("http://localhost:8000/predict", json=row) 
    print(f"🔹 Rekord {i+1}: {row}")
    print(f"🔸 Odpowiedź API: {response.json()}")