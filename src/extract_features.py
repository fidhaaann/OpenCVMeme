import cv2
import os
import csv
import mediapipe as mp
import numpy as np
from collections import defaultdict

DATA_DIR = "data"
HAND_CSV = "data/gestures.csv"
FACE_CSV = "data/face_none_speed.csv"

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


# ---------------- FACE POINTS ----------------
def face_points(face_landmarks):
    pts = []
    for lm in face_landmarks.landmark:
        pts.extend([lm.x, lm.y, lm.z])
    return pts


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

    return face_points(f.multi_face_landmarks[0])


def main():
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.1, min_tracking_confidence=0.1)

    face_video = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1
    )

    face_static = mp_face.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.1
    )

    os.makedirs("data", exist_ok=True)

    face_counts = defaultdict(int)

    with open(HAND_CSV, "w", newline="") as fh, open(FACE_CSV, "w", newline="") as ff:
        hand_writer = csv.writer(fh)
        face_writer = csv.writer(ff)

        for source in ["photos", "videos"]:
            root = os.path.join(DATA_DIR, source)
            for label in os.listdir(root):
                norm_label = label.strip().lower()
                folder = os.path.join(root, label)
                if not os.path.isdir(folder):
                    continue

                print(f"\n📁 Scanning {source}/{label}")

                for item in os.listdir(folder):
                    path = os.path.join(folder, item)

                    if source == "photos":
                        frame = cv2.imread(path)
                        if frame is None:
                            continue

                        # HAND
                        hand_feats = extract_hand(frame, hands)
                        if hand_feats is not None:
                            hand_writer.writerow([norm_label] + hand_feats)

                        # FACE
                        face_feats = extract_face(frame, face_static)
                        if face_feats is not None:
                            face_label = "speed" if norm_label == "speed" else "none"
                            face_writer.writerow([face_label] + face_feats)
                            face_counts[face_label] += 1
                        else:
                            # ⚠ If face NOT detected, still record NEUTRAL baseline
                            if norm_label != "speed":
                                zero_face = [0.0] * 1404
                                face_writer.writerow(["none"] + zero_face)
                                face_counts["none"] += 1

                    else:  # VIDEOS
                        cap = cv2.VideoCapture(path)
                        frame_id = 0

                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            frame_id += 1
                            if frame_id % 3 != 0:
                                continue

                            # HAND
                            hand_feats = extract_hand(frame, hands)
                            if hand_feats is not None:
                                hand_writer.writerow([norm_label] + hand_feats)

                            # FACE
                            face_feats = extract_face(frame, face_video)
                            if face_feats is not None:
                                face_label = "speed" if norm_label == "speed" else "none"
                                face_writer.writerow([face_label] + face_feats)
                                face_counts[face_label] += 1
                            else:
                                if norm_label != "speed":
                                    zero_face = [0.0] * 1404
                                    face_writer.writerow(["none"] + zero_face)
                                    face_counts["none"] += 1

                        cap.release()

    print("\n✅ Feature extraction complete")
    print("Hand data →", HAND_CSV)
    print("Face data →", FACE_CSV)

    print("\n📊 Face samples written:")
    for k, v in face_counts.items():
        print(f"  {k} : {v}")


if __name__ == "__main__":
    main()
