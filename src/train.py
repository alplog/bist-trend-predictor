import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import MACD
from datetime import datetime
from ta.momentum import RSIIndicator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ticker = "ASELS.IS"
start_date = "2022-01-01"

print(f"Downloading data for {ticker} from {start_date}...")
df = yf.download(ticker, start=start_date)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

rsi_calc = RSIIndicator(close=df["Close"].squeeze())
df["rsi"] = rsi_calc.rsi()

macd_calc = MACD(close=df["Close"].squeeze())
df["macd"] = macd_calc.macd()
df["macd_signal"] = macd_calc.macd_signal()
df["macd_diff"] = macd_calc.macd_diff()

df.dropna(inplace=True)

df_model = df[["Close", "rsi", "macd", "macd_signal", "macd_diff"]].copy()
df_model["target"] = df_model["Close"].shift(-1)
df_model.dropna(inplace=True)

df_model.to_csv("data/df_model.csv")

X = df_model.drop("target", axis=1)
y = df_model["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

today_str = datetime.today().strftime('%Y-%m-%d')
model_filename = f"models/model_{today_str}.pkl"

joblib.dump(model, model_filename)
print("Model saved to models/model.pkl")
