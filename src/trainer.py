import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import os

DATA = "data/gestures.csv"
MODEL_PATH = "models/gesture_model.joblib"

def main():
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(DATA, header=None)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    model.fit(X, y)

    print("Training accuracy:", model.score(X, y))
    dump(model, MODEL_PATH)
    print("Model saved:", MODEL_PATH)

if __name__ == "__main__":
    main()
