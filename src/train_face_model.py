import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import os

FACE_GESTURES = ["speed"]

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/gestures_face.csv", header=None)
df = df[df[0].isin(FACE_GESTURES)]

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)
dump(model, "models/face_model.joblib")

print("Face model trained:", FACE_GESTURES)
