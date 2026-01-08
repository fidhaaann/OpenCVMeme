import cv2
import os
import csv
import mediapipe as mp
import numpy as np

DATA_DIR = "data"
OUTPUT_CSV = "data/gestures.csv"

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh


# ---------- DISTANCE HELPER ----------
def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


# ---------- SHAPE-BASED HAND FEATURES ----------
def hand_shape_features(hand_landmarks):
    """
    Convert hand landmarks into shape features:
    - Distances from wrist to fingertips
    - Distances between fingertips
    - Normalized by hand size
    """

    lm = hand_landmarks.landmark

    # Landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    pts = {i: (lm[i].x, lm[i].y, lm[i].z) for i in range(21)}

    # Hand size reference (wrist to middle finger tip)
    hand_size = dist(pts[WRIST], pts[MIDDLE_TIP]) + 1e-6

    features = []

    # Distances from wrist to each fingertip
    for tip in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
        features.append(dist(pts[WRIST], pts[tip]) / hand_size)

    # Distances between fingertips (shape relationships)
    fingertip_pairs = [
        (THUMB_TIP, INDEX_TIP),
        (INDEX_TIP, MIDDLE_TIP),
        (MIDDLE_TIP, RING_TIP),
        (RING_TIP, PINKY_TIP),
        (THUMB_TIP, PINKY_TIP)
    ]

    for a, b in fingertip_pairs:
        features.append(dist(pts[a], pts[b]) / hand_size)

    return features


# ---------- FEATURE EXTRACTION ----------
def extract_features(frame, hands, face):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h = hands.process(rgb)
    f = face.process(rgb)

    features = []

    # ---------------- HAND FEATURES ----------------
    # If no hand is detected, SKIP this frame entirely (prevents zero-poisoning)
    if not h.multi_hand_landmarks:
        return None

    # Support 1-hand or 2-hand gestures
    for i in range(2):
        if i < len(h.multi_hand_landmarks):
            shape_feats = hand_shape_features(h.multi_hand_landmarks[i])
            features.extend(shape_feats)
        else:
            # Pad if second hand is missing
            features.extend([0] * 10)

    # ---------------- FACE FEATURES ----------------
    if f.multi_face_landmarks:
        for lm in f.multi_face_landmarks[0].landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        # Pad if face not detected
        features.extend([0] * (468 * 3))

    return features


# ---------- MAIN ----------
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
                        if feats is not None:
                            writer.writerow([label] + feats)

                    # ---------- VIDEOS ----------
                    else:
                        cap = cv2.VideoCapture(path)
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            feats = extract_features(frame, hands, face)
                            if feats is not None:
                                writer.writerow([label] + feats)

                        cap.release()

    print("✅ Feature extraction complete.")
    print("👉 Shape-based hand features used.")
    print("👉 Zero/invalid frames removed.")
    print(f"📄 Output saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
