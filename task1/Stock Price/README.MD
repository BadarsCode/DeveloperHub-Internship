# 📈 Stock Price Prediction using LSTM

## 📌 Project Overview
This project predicts **future stock closing prices** using a **Long Short-Term Memory (LSTM)** neural network trained on historical stock data fetched from **Yahoo Finance**.

---

## 🎯 Objective
To predict the **next-day or future closing price** of a stock based on historical price patterns.

---

## 📊 Dataset
- **Source:** Yahoo Finance (via `yfinance` API)
- **Stock Example:** TSLA (Tesla Inc.)
- **Feature Used:** Close price
- **Time Period:** Last 5–10 years (configurable)

---

## 🧠 Why LSTM?
- Designed for time-series data
- Learns long-term dependencies
- Handles sequential patterns better than traditional ML models

---

## 🧹 Data Preprocessing
- Selected closing price
- Normalized values using MinMaxScaler
- Created rolling windows of 60 days
- Split data into training and testing sets (time-aware split)

---

## 🤖 Model Architecture
- LSTM (50 units) × 2 layers
- Dropout for regularization
- Dense output layer
- Optimizer: Adam
- Loss function: Mean Squared Error

---

## 📈 Model Evaluation
- Training & validation loss
- Visual comparison of actual vs predicted prices
- RMSE for real-price error measurement

---

## 🖥 Streamlit Web App
Features:
- User enters stock symbol
- Selects a future date from today
- Predicts closing price for that date
- Displays historical vs predicted price graph

---

## 🔮 Prediction Method
- Recursive multi-step forecasting
- Each predicted day feeds into the next prediction
- Suitable for short-term trend forecasting

---

## 🛠 Technologies Used
- Python
- TensorFlow / Keras
- Pandas, NumPy
- yfinance
- Scikit-learn
- Matplotlib
- Streamlit

---

## ⚠ Disclaimer
Stock market predictions are probabilistic. This model does not account for news, earnings, or market events and should not be used for financial advice.

---

## 📌 Author
Subhanud Din
