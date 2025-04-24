import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import joblib
from preprocess import add_features

def train_model(csv_path, model_path):
    df = pd.read_csv(csv_path)
    df = add_features(df)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna()

    features = ['MA10', 'MA50', 'Volatility', 'Log_Return']
    target = 'Close'

    X, y = df[features], df[target]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n📊 Evaluation for {os.path.basename(csv_path)}")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAE     : {mae:.4f}")

    joblib.dump(model, model_path)
    print(f"✅ Model saved: {model_path}")

def batch_train(data_dir, model_dir, region):
    os.makedirs(model_dir, exist_ok=True)

    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            stock_name = filename.replace(".csv", "")
            csv_path = os.path.join(data_dir, filename)
            model_path = os.path.join(model_dir, f"{region}_model_{stock_name.lower()}.pkl")
            try:
                train_model(csv_path, model_path)
            except Exception as e:
                print(f"❌ Failed to train {stock_name}: {e}")

if __name__ == "__main__":
    batch_train("data/global", "models", "global")
    batch_train("data/india", "models", "india")
