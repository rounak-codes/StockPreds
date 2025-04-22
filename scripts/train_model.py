import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from preprocess import add_features

def train_model(csv_path, model_path):
    df = pd.read_csv(csv_path)
    df = add_features(df)
    X = df[['MA10', 'MA50', 'Volatility']]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    print(f"✅ Model trained and saved: {model_path}")

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
                print(f"❌ Failed training {stock_name}: {e}")

if __name__ == "__main__":
    batch_train("data/global", "models", "global")
    batch_train("data/india", "models", "india")
