import cv2
import os
import csv
import mediapipe as mp
import numpy as np

DATA_DIR = "data"
HAND_CSV = "data/gestures.csv"          # Hand gestures (no speed)
FACE_CSV = "data/speed_face.csv"        # Face data ONLY for speed

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh


def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


# ---------------- HAND SHAPE FEATURES ----------------
def hand_shape_features(hand_landmarks):
    lm = hand_landmarks.landmark
    WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 0, 4, 8, 12, 16, 20

    pts = {i: (lm[i].x, lm[i].y, lm[i].z) for i in range(21)}
    hand_size = dist(pts[WRIST], pts[MIDDLE_TIP]) + 1e-6

    features = []

    for tip in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
        features.append(dist(pts[WRIST], pts[tip]) / hand_size)

    for a, b in [
        (THUMB_TIP, INDEX_TIP),
        (INDEX_TIP, MIDDLE_TIP),
        (MIDDLE_TIP, RING_TIP),
        (RING_TIP, PINKY_TIP),
        (THUMB_TIP, PINKY_TIP),
    ]:
        features.append(dist(pts[a], pts[b]) / hand_size)

    return features


# ---------------- FACE FEATURES (RAW LANDMARKS) ----------------
def face_features(face_landmarks):
    feats = []
    for lm in face_landmarks.landmark:
        feats.extend([lm.x, lm.y, lm.z])
    return feats


# ---------------- FEATURE EXTRACTION ----------------
def extract_hand(frame, hands):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h = hands.process(rgb)

    if not h.multi_hand_landmarks:
        return None

    features = []
    for i in range(2):
        if i < len(h.multi_hand_landmarks):
            features.extend(hand_shape_features(h.multi_hand_landmarks[i]))
        else:
            features.extend([0] * 10)

    return features


def extract_face(frame, face):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    f = face.process(rgb)

    if not f.multi_face_landmarks:
        return None

    return face_features(f.multi_face_landmarks[0])


# ---------------- MAIN ----------------
def main():
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

    os.makedirs("data", exist_ok=True)

    with open(HAND_CSV, "w", newline="") as fh, open(FACE_CSV, "w", newline="") as ff:
        hand_writer = csv.writer(fh)
        face_writer = csv.writer(ff)

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

                        # ----- HAND DATA (ALL EXCEPT SPEED) -----
                        if label != "speed":
                            feats = extract_hand(frame, hands)
                            if feats is not None:
                                hand_writer.writerow([label] + feats)

                        # ----- FACE DATA (ONLY SPEED) -----
                        if label == "speed":
                            face_feats = extract_face(frame, face)
                            if face_feats is not None:
                                face_writer.writerow([label] + face_feats)

                    # ---------- VIDEOS ----------
                    else:
                        cap = cv2.VideoCapture(path)
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            if label != "speed":
                                feats = extract_hand(frame, hands)
                                if feats is not None:
                                    hand_writer.writerow([label] + feats)

                            if label == "speed":
                                face_feats = extract_face(frame, face)
                                if face_feats is not None:
                                    face_writer.writerow([label] + face_feats)

                        cap.release()

    print("✅ Feature extraction complete.")
    print("👉 Hand gestures saved to:", HAND_CSV)
    print("👉 Speed facial features saved to:", FACE_CSV)


if __name__ == "__main__":
    main()
