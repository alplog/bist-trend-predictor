import os
import joblib
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

data_name = input("Enter the data csv file name (e.g., df_model_2025-08-28.csv): ")
data_path = f"data/{data_name}" #FIX!!!!!!!!!!!

model_name = input("Enter the model filename (e.g., model_2025-08-28.pkl): ").strip()
model_path = f"models/{model_name}"

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = joblib.load(model_path)
df_model = pd.read_csv(data_path, index_col=0)

X = df_model.drop("target", axis=1)
last_row = X.iloc[[-1]]
last_date = pd.to_datetime(df_model.index[-1])
next_day = last_date + timedelta(days=1)

prediction = model.predict(last_row)[0]

print(f"Last known date: {last_date.date()}")
print(f"Predicted close price for {next_day.date()}: {prediction:.2f} TL")
