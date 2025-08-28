import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import glob
import os

df_model = pd.read_csv("data/df_model.csv", index_col=0)

X = df_model.drop("target", axis=1)
y = df_model["target"]

split_index = int(len(df_model) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

model_files = glob.glob("models/model_*.pkl")
latest_model = max(model_files, key=os.path.getctime)
model = joblib.load(latest_model)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation on Test Set:")
print(f"🔹 Mean Squared Error (MSE): {mse:.2f}")
print(f"🔹 Mean Absolute Error (MAE): {mae:.2f}")
print(f"🔹 R² Score: {r2:.2f}")

plt.figure(figsize=(12,6))
plt.plot(y_test.values, label="Actual", color="blue")
plt.plot(y_pred, label="Predicted", color="orange")
plt.title("Model Predictions vs Actual on Test Set")
plt.xlabel("Time Index")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("models/evaluation_plot.png")
plt.show()
