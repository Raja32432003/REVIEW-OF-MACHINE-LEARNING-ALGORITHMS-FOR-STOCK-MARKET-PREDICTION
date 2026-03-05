import streamlit as st
import requests
import pandas as pd
import time
import plotly.graph_objs as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = 'D9H4FSO9M2O2RZFV'

@st.cache_data
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    return tables[0]["Symbol"].tolist()

# Caching and adding key parameter to avoid repeated API hits
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol):
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=full"
        response = requests.get(url)
        data = response.json()

        if "Time Series (Daily)" in data:
            df = pd.DataFrame(data["Time Series (Daily)"]).T
            df = df.rename(columns={"5. adjusted close": "Close"})
            df = df[["Close"]]
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df["Close"] = df["Close"].astype(float)
            return df
        else:
            return pd.DataFrame()  # Empty
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_exchange_rates(base_currency="USD"):
    try:
        response = requests.get(f"https://api.exchangerate.host/latest?base={base_currency}")
        return response.json().get("rates", {})
    except:
        return {}

@st.cache_data(ttl=600)
def convert_currency(amount, from_currency, to_currency):
    rates = fetch_exchange_rates(from_currency)
    return amount * rates.get(to_currency, 1)

def get_future_trading_dates(n_days):
    dates = []
    current_date = datetime.now()
    while len(dates) < n_days:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:
            dates.append(current_date)
    return [d.strftime("%Y-%m-%d") for d in dates]

# Sidebar
st.sidebar.title("Stock Market Dashboard")
stocks = get_sp500_tickers()
selected_stock = st.sidebar.selectbox("Select a Stock", stocks)
currency = st.sidebar.selectbox("Currency", ["INR", "USD"])
currency_symbols = {"INR": "₹", "USD": "$"}
currency_symbol = currency_symbols.get(currency, "$")

# Fetch stock data once
df = fetch_stock_data(selected_stock)

# Live price
base_currency = "USD"
if not df.empty:
    base_price = df["Close"].iloc[-1]
    live_price = convert_currency(base_price, base_currency, currency)
    st.sidebar.markdown(
        f"Live Price: <span style='font-size:20px;font-weight:bold'>{currency_symbol}{live_price:.2f}</span>",
        unsafe_allow_html=True,
    )
else:
    st.sidebar.error("Live price unavailable or data not available.")

# User inputs
date_range = st.sidebar.selectbox("Date Range", ["6M", "1Y", "2Y"])
prediction_days = st.sidebar.slider("Prediction Days", 15, 200, 30)

# Tabs
tabs = st.tabs(["📈 Stock Predictor", "📊 Stock Summary"])

with tabs[0]:
    st.title("📈 Stock Market Predictor")

    if df.empty:
        st.error("Failed to load stock data.")
    else:
        days = {"6M": 180, "1Y": 365, "2Y": 730}[date_range]
        df_recent = df.last(f"{days}D")

        st.subheader("Stock Price Chart")
        fig = go.Figure()
        converted_prices = [convert_currency(p, base_currency, currency) for p in df_recent['Close']]
        fig.add_trace(go.Scatter(x=df_recent.index, y=converted_prices,
                                 mode='lines', name='Close', line=dict(color='blue')))
        fig.update_layout(title=f"{selected_stock} Price", xaxis_title="Date", yaxis_title=f"Price ({currency})")
        st.plotly_chart(fig)

        if st.button("Predict Future Prices"):
            st.info("Training LSTM model...")

            # LSTM model
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[['Close']])
            x_train, y_train = [], []
            for i in range(60, len(scaled_data)):
                x_train.append(scaled_data[i - 60:i, 0])
                y_train.append(scaled_data[i, 0])

            if not x_train:
                st.error("Not enough data for training.")
            else:
                x_train, y_train = np.array(x_train), np.array(y_train)
                x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

                model = Sequential([
                    LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)),
                    Dropout(0.2),
                    LSTM(128),
                    Dropout(0.2),
                    Dense(64, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

                input_seq = scaled_data[-60:]
                predicted_prices = []
                current_input = input_seq.reshape(1, 60, 1)
                for _ in range(prediction_days):
                    pred = model.predict(current_input, verbose=0)[0][0]
                    predicted_prices.append(pred)
                    current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

                predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
                predicted_prices = [convert_currency(p[0], base_currency, currency) for p in predicted_prices]

                future_dates = get_future_trading_dates(prediction_days)

                st.subheader("Predicted Price Chart")
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=future_dates, y=predicted_prices,
                                              mode='lines+markers', name='Predicted', line=dict(color='red')))
                fig_pred.update_layout(title="Future Predictions", xaxis_title="Date", yaxis_title=f"Price ({currency})")
                st.plotly_chart(fig_pred)

                st.subheader("Predicted Prices Table")
                df_pred = pd.DataFrame({"Date": future_dates, "Predicted Price": [f"{currency_symbol}{p:.2f}" for p in predicted_prices]})
                st.dataframe(df_pred, use_container_width=True)

with tabs[1]:
    st.title("📊 Stock Summary")
    if df.empty:
        st.error("No summary data available.")
    else:
        st.write("Most recent closing prices:")
        st.line_chart(df['Close'].tail(30))
