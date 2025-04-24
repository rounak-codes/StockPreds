import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

def prepare_lstm_data(series, lookback=60):
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i-lookback:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def train_lstm_model(csv_path, model_path, lookback=60):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Use log returns for realism
    df['Log_Close'] = np.log(df['Close'])
    data = df['Log_Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = prepare_lstm_data(data_scaled, lookback)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), callbacks=[es], verbose=1)

    model.save(model_path)
    scaler_path = model_path.replace('.h5', '_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✅ Model saved to {model_path} and scaler to {scaler_path}")

def batch_train(data_dir, model_dir, region, lookback=60):
    os.makedirs(model_dir, exist_ok=True)

    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            stock_name = filename.replace(".csv", "")
            csv_path = os.path.join(data_dir, filename)
            model_path = os.path.join(model_dir, f"{region}_lstm_model_{stock_name.lower()}.h5")
            try:
                train_lstm_model(csv_path, model_path, lookback)
            except Exception as e:
                print(f"❌ Failed training {stock_name}: {e}")

if __name__ == "__main__":
    batch_train("data/global", "models", "global")
    batch_train("data/india", "models", "india")
