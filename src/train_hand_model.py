import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import os

HAND_GESTURES = ["cooked", "vanish", "dicaprio", "think"]

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/gestures_hands.csv", header=None)
df = df[df[0].isin(HAND_GESTURES)]

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

model.fit(X, y)
dump(model, "models/hand_model.joblib")

print("Hand model trained:", HAND_GESTURES)
