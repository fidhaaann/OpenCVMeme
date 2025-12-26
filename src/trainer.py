import cv2
import os
import pandas as pd
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

DATA_DIR = "data/videos"
MODEL_PATH = "models/gesture_model.joblib"

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    hands = mp_hands.Hands(max_num_hands=1)
    face = mp_face.FaceMesh(max_num_faces=1)

    features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_res = hands.process(rgb)
        f_res = face.process(rgb)

        if h_res.multi_hand_landmarks and f_res.multi_face_landmarks:
            row = []
            for lm in h_res.multi_hand_landmarks[0].landmark:
                row.extend([lm.x, lm.y, lm.z])
            for lm in f_res.multi_face_landmarks[0].landmark:
                row.extend([lm.x, lm.y, lm.z])
            features.append(row)

    cap.release()
    return features


def main():
    X, y = [], []

    for label in os.listdir(DATA_DIR):
        folder = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            for sample in extract_features(path):
                X.append(sample)
                y.append(label)

    print(f"Collected {len(X)} samples")

    model = RandomForestClassifier(n_estimators=300)
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    dump(model, MODEL_PATH)

    print("Model trained and saved.")

if __name__ == "__main__":
    main()
