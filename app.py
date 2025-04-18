import os
import requests
import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS
from tensorflow.keras.losses import MeanSquaredError

# Suppress unnecessary TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Twelve Data API Key
API_KEY = "8149690927ae4161a977ab79a3bcd0dd"
BASE_URL = "https://api.twelvedata.com/time_series"

# Directory where trained models are stored
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")  # Ensure 'models' folder exists

# Ensure the directory exists
if not os.path.exists(MODEL_DIR):
    print(f"⚠ Warning: Model directory '{MODEL_DIR}' does not exist.")

# Home Route
@app.route("/")
def home():
    return "Stock Prediction API is running!"

# Function to Fetch Latest Stock Data
def fetch_stock_data(stock_symbol, days=8):
    params = {
        "symbol": stock_symbol,
        "interval": "1day",
        "outputsize": 50,  # Ensure we fetch at least 8 valid days
        "apikey": API_KEY
    }
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        if "values" not in data:
            return None, f"Error fetching data: {data.get('message', 'Unknown error')}"
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values(by="datetime")  # Ensure ascending order
        df = df.rename(columns={"datetime": "Date", "close": "Close"})
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["Date"] = df["Date"].dt.strftime('%Y-%m-%d')
        df = df.tail(days).dropna()
        if len(df) < days:
            return None, "⚠ Not enough historical data available."
        return df, None
    except Exception as e:
        return None, f"⚠ Error fetching data: {str(e)}"

# API Endpoint for Predicting Stock Price
@app.route("/predict", methods=["GET"])
def predict_stock():
    stock_symbol = request.args.get("symbol", "").upper()
    target_date_str = request.args.get("date", "")
    if not stock_symbol:
        return jsonify({"error": "Stock symbol is required"}), 400
    try:
        target_date = (
            datetime.datetime.strptime(target_date_str, "%Y-%m-%d").date()
            if target_date_str
            else datetime.date.today() + datetime.timedelta(days=1)
        )
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
    today = datetime.date.today()
    if target_date <= today:
        return jsonify({"error": "Target date must be in the future"}), 400
    days_ahead = (target_date - today).days
    model_path = os.path.join(MODEL_DIR, f"{stock_symbol}_model.h5")
    if not os.path.exists(model_path):
        return jsonify({"error": f"Model for {stock_symbol} not found"}), 404
    df, error = fetch_stock_data(stock_symbol, days=8)
    if error or df is None:
        return jsonify({"error": error or "Insufficient data available"}), 500
    scaler = MinMaxScaler()
    df[["Close"]] = scaler.fit_transform(df[["Close"]])
    X_input = df["Close"].values[-7:].reshape(1, 7, 1)
    try:
        model = load_model(model_path, custom_objects={"mse": MeanSquaredError()})
        predicted_price = None
        for _ in range(days_ahead):
            predicted_scaled = model.predict(X_input)
            predicted_price = scaler.inverse_transform(predicted_scaled)[0, 0]
            new_data = np.append(X_input.flatten()[1:], scaler.transform([[predicted_price]]))
            X_input = new_data.reshape(1, 7, 1)
        return jsonify({
            "stock_symbol": stock_symbol,
            "predicted_price": round(float(predicted_price), 2),
            "target_date": target_date.strftime('%Y-%m-%d')
        })
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

# Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
