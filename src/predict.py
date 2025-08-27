import pandas as pd
import joblib
from datetime import datetime, timedelta

model = joblib.load("models/model.pkl")
df_model = pd.read_csv("data/df_model.csv", index_col=0)

X = df_model.drop("target", axis=1)
last_row = X.iloc[[-1]]
last_date = pd.to_datetime(df_model.index[-1])
next_day = last_date + timedelta(days=1)

prediction = model.predict(last_row)[0]

print(f"Last known date: {last_date.date()}")
print(f"Predicted close price for {next_day.date()}: {prediction:.2f} TL")
