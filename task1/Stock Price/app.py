import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import date
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Stock Price Prediction (LSTM)", layout="centered")

st.title("📈 Stock Price Prediction using LSTM")
st.write("Predict future closing prices based on historical trends")

# -------------------- LOAD MODEL & SCALER --------------------
model = load_model("lstm_stock_model.h5")
scaler = joblib.load("scaler.pkl")

# -------------------- USER INPUT --------------------
ticker = st.text_input("Enter Stock Symbol", value="AAPL")

future_date = st.date_input(
    "Select a future date",
    min_value=date.today()
)

# -------------------- FETCH DATA --------------------
df = yf.download(ticker, period="5y")
df = df[["Close"]]

if df.empty:
    st.error("Invalid stock symbol")
    st.stop()

# -------------------- PREPARE DATA --------------------
scaled_data = scaler.transform(df)

last_60_days = scaled_data[-60:].reshape(1, 60, 1)

# -------------------- DAYS AHEAD --------------------
days_ahead = (future_date - date.today()).days

st.write(f"📅 Predicting **{days_ahead} day(s)** ahead")

# -------------------- PREDICTION --------------------
predictions = []
current_input = last_60_days.copy()

for _ in range(days_ahead):
    next_pred = model.predict(current_input, verbose=0)
    predictions.append(next_pred[0][0])

    current_input = np.append(
        current_input[:, 1:, :],
        next_pred.reshape(1, 1, 1),
        axis=1
    )

# Inverse scaling
predicted_prices = scaler.inverse_transform(
    np.array(predictions).reshape(-1, 1)
)

# -------------------- DISPLAY RESULT --------------------
st.subheader("📊 Prediction Result")

st.success(
    f"Predicted Closing Price on {future_date}: "
    f"${predicted_prices[-1][0]:.2f}"
)

# -------------------- PLOT --------------------
st.subheader("📉 Price Trend")

plot_df = df.tail(100).copy()
future_dates = pd.date_range(df.index[-1], periods=days_ahead+1)[1:]

plt.figure(figsize=(10,5))
plt.plot(plot_df.index, plot_df["Close"], label="Historical Prices")
plt.plot(future_dates, predicted_prices, label="Predicted Prices", linestyle="--")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price")
plt.title(f"{ticker} Price Prediction")
st.pyplot(plt)

# -------------------- DISCLAIMER --------------------
st.warning(
    "⚠ This prediction is based only on historical price patterns. "
    "Stock markets are volatile and affected by news, earnings, and events."
)
