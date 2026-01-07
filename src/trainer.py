import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
import os

DATA_PATH = "data/gestures.csv"
MODEL_PATH = "models/gesture_model.joblib"

os.makedirs("models", exist_ok=True)

df = pd.read_csv(DATA_PATH, header=None)

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# Split for accuracy measurement
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

dump(model, MODEL_PATH)

print(f"✅ Model trained")
print(f"📊 Training accuracy: {acc * 100:.2f}%")
print(f"💾 Saved to: {MODEL_PATH}")
