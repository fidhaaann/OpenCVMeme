import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump

CSV_PATH = "data/gestures.csv"
MODEL_PATH = "models/gesture_model.joblib"

df = pd.read_csv(CSV_PATH)

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"🎯 Accuracy: {acc*100:.2f}%")

dump(model, MODEL_PATH)
print(f"💾 Saved → {MODEL_PATH}")
