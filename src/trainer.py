import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
import os

DATA = "data/gestures.csv"
MODEL_PATH = "models/gesture_model.joblib"

def main():
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(DATA, header=None)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"Accuracy: {acc*100:.2f}%")

    dump(model, MODEL_PATH)
    print("Model saved.")

if __name__ == "__main__":
    main()
