# BIST Trend Predictor

📈 Predict next-day trend (up/down) of Borsa Istanbul (BIST) stocks using ML and technical indicators.

---

## 🎯 Project Goal

This project uses stock price data and technical analysis indicators (RSI, MACD, EMA, etc.) to build a machine learning model that classifies whether a BIST stock (e.g., ASELS.IS, THYAO.IS) will go **up or down** the next day.

---

## 🧰 Tools & Libraries

- `pandas`, `numpy` — Data processing
- `scikit-learn` — ML models (Random Forest, Logistic Regression)
- `yfinance` — BIST data downloader
- `ta` — Technical indicators
- `matplotlib`, `seaborn` — Visualization
- `shap` — Model explainability
- `jupyter` — Notebook analysis
- `streamlit` *(optional)* — Interactive dashboard

---

## 📁 Project Structure

bist-trend-predictor/
│
├── data/ # raw stock data (ignored in git)
├── models/ # saved models (ignored in git)
├── notebooks/ # Jupyter notebooks (EDA, training)
├── src/ # scripts and utilities
├── tests/ # unit tests
├── requirements.txt # dependencies
├── .gitignore
└── README.md
---

## 🚀 How to Run

1. Install dependencies:
python3 -m pip install -r requirements.txt


Launch notebook:
jupyter notebook notebooks/eda.ipynb