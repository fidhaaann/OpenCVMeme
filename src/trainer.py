import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
import os
import random

HAND_DATA_PATH = "data/gestures.csv"
FACE_DATA_PATH = "data/face_none_speed.csv"

HAND_MODEL_PATH = "models/hand_gesture_model.joblib"
FACE_REF_PATH = "models/face_none_speed_ref.npz"

os.makedirs("models", exist_ok=True)

# =========================================================
# 1️⃣ HAND MODEL (UNCHANGED)
# =========================================================
print("\n=== Training Hand Gesture Model ===")

hand_df = pd.read_csv(HAND_DATA_PATH, header=None)
X_hand = hand_df.iloc[:, 1:]
y_hand = hand_df.iloc[:, 0]

Xh_train, Xh_test, yh_train, yh_test = train_test_split(
    X_hand, y_hand, test_size=0.2, random_state=42, stratify=y_hand
)

hand_model = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42)
hand_model.fit(Xh_train, yh_train)

yh_pred = hand_model.predict(Xh_test)
hand_acc = accuracy_score(yh_test, yh_pred)

dump(hand_model, HAND_MODEL_PATH)

print("✅ Hand model trained")
print(f"📊 Hand accuracy: {hand_acc * 100:.2f}%")
print(f"💾 Saved → {HAND_MODEL_PATH}")

# =========================================================
# 2️⃣ FACE: NONE vs SPEED (BALANCED + NORMALIZED)
# =========================================================
print("\n=== Building Face None vs Speed Reference ===")

face_df = pd.read_csv(FACE_DATA_PATH, header=None)
X_face = face_df.iloc[:, 1:].values
y_face = face_df.iloc[:, 0].values

none_faces = X_face[y_face == "none"]
speed_faces = X_face[y_face == "speed"]

print(f"Raw None samples  : {len(none_faces)}")
print(f"Raw Speed samples : {len(speed_faces)}")

if len(speed_faces) < 5:
    print("❌ Not enough SPEED face samples. Add more speed photos/videos.")
    exit()

# ---------- BALANCE DATA ----------
sample_size = len(speed_faces)
none_sampled = random.sample(list(none_faces), sample_size)

none_faces = np.array(none_sampled)
speed_faces = np.array(speed_faces)

print(f"Balanced None samples  : {len(none_faces)}")
print(f"Balanced Speed samples : {len(speed_faces)}")

# ---------- NORMALIZE ----------
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

none_faces = np.array([normalize(v) for v in none_faces])
speed_faces = np.array([normalize(v) for v in speed_faces])

# ---------- BUILD REFERENCE ----------
none_mean = np.mean(none_faces, axis=0)
speed_mean = np.mean(speed_faces, axis=0)

# ---------- EVALUATION ----------
def classify(vec):
    vec = normalize(vec)
    d_none = np.linalg.norm(vec - none_mean)
    d_speed = np.linalg.norm(vec - speed_mean)
    return "speed" if d_speed < 0.85 * d_none else "none"

preds = [classify(x) for x in np.vstack([none_faces, speed_faces])]
true = ["none"] * len(none_faces) + ["speed"] * len(speed_faces)

face_acc = accuracy_score(true, preds)

np.savez(FACE_REF_PATH, none_mean=none_mean, speed_mean=speed_mean)

print("✅ Face reference model built")
print(f"📊 Face none vs speed accuracy: {face_acc * 100:.2f}%")
print(f"💾 Saved → {FACE_REF_PATH}")

print("\n🎉 Training complete.")
