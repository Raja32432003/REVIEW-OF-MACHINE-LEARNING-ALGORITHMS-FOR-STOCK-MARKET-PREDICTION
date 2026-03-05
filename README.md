# Stock Market Predictor using LSTM

## Project Overview

A Stock Market Prediction Dashboard built using Python, Streamlit, and Deep Learning (LSTM).
This application fetches real-time stock data, visualizes historical trends, and predicts future stock prices using an LSTM neural network model.

## 🚀 Features

📊 Interactive Stock Price Dashboard<br>
📈 Historical Stock Price Visualization <br>
🤖 Future Stock Price Prediction using LSTM<br>
💱 Currency Conversion (USD ↔ INR)<br>
📅 Select different date ranges<br>
📉 Future prediction for 15 to 200 days<br>
📋 Predicted price table<br>
⚡ Fast performance with Streamlit caching<br>

## 🛠️ Technologies Used

Python<br>
Streamlit<br>
TensorFlow / Keras<br>
Pandas<br>
NumPy<br>
Plotly<br>
Scikit-learn<br>
Alpha Vantage API<br>
Exchange Rate API<br>

## 📂 Project Structure

stock-market-predictor<br>
│<br>
├── main.py<br>
├── requirements.txt<br>
└── README.md<br>

## 📊 Data Source

### Stock data is fetched from:

Alpha Vantage API<br>
Wikipedia S&P 500 companies list<br>
Exchange Rate API<br>

# 🧠 Machine Learning Model

This project uses LSTM (Long Short-Term Memory) neural networks for time-series prediction.

Steps:

Collect historical stock prices<br>
Normalize data using MinMaxScaler<br>
Train LSTM neural network<br>
Predict future prices<br>
Display results using Plotly charts<br>
