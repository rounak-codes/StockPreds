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
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    train_model("data/global/AAPL.csv", "models/global_model_aapl.pkl")
    train_model("data/india/INFY.NS.csv", "models/india_model_infy.pkl")
