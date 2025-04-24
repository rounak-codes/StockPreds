import os
import numpy as np
import pandas as pd
import joblib
from pandas.tseries.offsets import BDay
from tensorflow.keras.models import load_model # type: ignore

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))

def prepare_lstm_data(df, scaler, window=60):
    log_close = np.log(df['Close'].values).reshape(-1, 1)
    data_scaled = scaler.transform(log_close)

    X, y = [], []
    for i in range(window, len(data_scaled)):
        X.append(data_scaled[i-window:i])
        y.append(data_scaled[i])

    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y)
    aligned_df = df.iloc[window:].reset_index(drop=True)

    return X, y, aligned_df

def load_lstm_model_and_scaler(symbol, region):
    suffix = ".ns" if region == "india" else ""
    base_name = f"{region}_lstm_model_{symbol.lower()}{suffix}"
    model_path = os.path.join(MODELS_DIR, f"{base_name}.h5")
    scaler_path = os.path.join(MODELS_DIR, f"{base_name}_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_with_lstm(df, symbol, region, future_months=0, window=60):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    model, scaler = load_lstm_model_and_scaler(symbol, region)
    X, y, aligned_df = prepare_lstm_data(df, scaler, window=window)

    # In-sample predictions
    scaled_preds = model.predict(X).flatten()
    preds = scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
    preds = np.exp(preds)  # convert log-price back to real price

    in_sample = aligned_df.copy()
    in_sample['Predicted_Close'] = preds

    # Out-of-sample future prediction
    future = pd.DataFrame()
    if future_months > 0:
        history = df.copy()
        # Inside predict_with_lstm, before the future prediction loop:
        last_window = scaler.transform(np.log(history['Close'].values).reshape(-1, 1))[-window:]
        future_data = []
        for _ in range(future_months * 21):
            next_scaled = model.predict(last_window.reshape(1, window, 1)).flatten()[0]
            next_price_log = scaler.inverse_transform([[next_scaled]])[0][0]
            next_price = np.exp(next_price_log)
            next_date = history['Date'].max() + BDay(1)

            history = pd.concat([history, pd.DataFrame({'Date': [next_date], 'Close': [next_price]})], ignore_index=True)
            last_window = np.vstack([last_window[1:], [[next_scaled]]])

            future_data.append({'Date': next_date, 'Predicted_Close': next_price})

        future = pd.DataFrame(future_data)

    return in_sample, future

if __name__ == "__main__":
    df = pd.read_csv("data/global/AAPL.csv")
    hist, fut = predict_with_lstm(df, "AAPL", "global", future_months=6)
    print(hist.tail())
    print(fut.head())
