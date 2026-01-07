import cv2
import os
import csv
import mediapipe as mp

DATA_DIR = "data"
OUTPUT_CSV = "data/gestures.csv"

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh


def extract_features(frame, hands, face):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h = hands.process(rgb)
    f = face.process(rgb)

    features = []

    # ---------------- HAND FEATURES (ALLOW 1 OR 2 HANDS) ----------------
    if h.multi_hand_landmarks:
        for i in range(2):
            if i < len(h.multi_hand_landmarks):
                for lm in h.multi_hand_landmarks[i].landmark:
                    features.extend([lm.x, lm.y, lm.z])
            else:
                # Pad missing second hand
                features.extend([0] * 63)
    else:
        # No hands detected
        features.extend([0] * 126)

    # ---------------- FACE FEATURES (ALWAYS INCLUDED) ----------------
    if f.multi_face_landmarks:
        for lm in f.multi_face_landmarks[0].landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        # No face detected
        features.extend([0] * (468 * 3))

    return features


def main():
    # Permissive hand detector for subtle gestures
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1,
        model_complexity=1
    )

    face = mp_face.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)

        for source in ["photos", "videos"]:
            root = os.path.join(DATA_DIR, source)
            for label in os.listdir(root):
                folder = os.path.join(root, label)
                if not os.path.isdir(folder):
                    continue

                for item in os.listdir(folder):
                    path = os.path.join(folder, item)

                    # ---------- PHOTOS ----------
                    if source == "photos":
                        frame = cv2.imread(path)
                        if frame is None:
                            continue

                        feats = extract_features(frame, hands, face)
                        writer.writerow([label] + feats)

                    # ---------- VIDEOS ----------
                    else:
                        cap = cv2.VideoCapture(path)
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            feats = extract_features(frame, hands, face)
                            writer.writerow([label] + feats)

                        cap.release()

    print("✅ Feature extraction complete → data/gestures.csv")


if __name__ == "__main__":
    main()
