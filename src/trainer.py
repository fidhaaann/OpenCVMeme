import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
import os

DATA_PATH = "data/gestures.csv"
MODEL_PATH = "models/gesture_model.joblib"

os.makedirs("models", exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH, header=None)

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

dump(model, MODEL_PATH)

print("✅ Model trained successfully")
print(f"📊 Training accuracy: {acc * 100:.2f}%")
print(f"💾 Model saved at: {MODEL_PATH}")
