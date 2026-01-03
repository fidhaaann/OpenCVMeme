import cv2
import os
import csv
import mediapipe as mp

DATA_DIR = "data"
OUTPUT_CSV = "data/gestures.csv"

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh


def extract_from_frame(frame, hands, face):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h_res = hands.process(rgb)
    f_res = face.process(rgb)

    # ---------------- HAND FEATURES ----------------
    hand_features = []

    if h_res.multi_hand_landmarks:
        for i in range(2):
            if i < len(h_res.multi_hand_landmarks):
                for lm in h_res.multi_hand_landmarks[i].landmark:
                    hand_features.extend([lm.x, lm.y, lm.z])
            else:
                hand_features.extend([0] * 63)
    else:
        hand_features.extend([0] * 126)

    # ---------------- FACE FEATURES ----------------
    face_features = []

    if f_res.multi_face_landmarks:
        for lm in f_res.multi_face_landmarks[0].landmark:
            face_features.extend([lm.x, lm.y, lm.z])
    else:
        face_features.extend([0] * (468 * 3))

    # ---------------- HACKY SCALING ----------------
    # Boost hands, suppress face
    hand_features = [v * 2.0 for v in hand_features]
    face_features = [v * 0.3 for v in face_features]

    return hand_features + face_features


def main():
    hands = mp_hands.Hands(max_num_hands=2)
    face = mp_face.FaceMesh(max_num_faces=1)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)

        # -------- PHOTOS --------
        photo_root = os.path.join(DATA_DIR, "photos")
        for label in os.listdir(photo_root):
            folder = os.path.join(photo_root, label)
            if not os.path.isdir(folder):
                continue

            for img_name in os.listdir(folder):
                img_path = os.path.join(folder, img_name)
                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                features = extract_from_frame(frame, hands, face)
                writer.writerow([label] + features)

        # -------- VIDEOS --------
        video_root = os.path.join(DATA_DIR, "videos")
        for label in os.listdir(video_root):
            folder = os.path.join(video_root, label)
            if not os.path.isdir(folder):
                continue

            for vid_name in os.listdir(folder):
                cap = cv2.VideoCapture(os.path.join(folder, vid_name))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    features = extract_from_frame(frame, hands, face)
                    writer.writerow([label] + features)

                cap.release()

    print("Feature extraction complete → data/gestures.csv")


if __name__ == "__main__":
    main()
