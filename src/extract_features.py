import cv2
import os
import mediapipe as mp
import csv

DATA_DIR = "data/videos"
OUTPUT_FILE = "data/gestures.csv"

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

def extract_from_video(video_path, label, writer):
    cap = cv2.VideoCapture(video_path)
    hands = mp_hands.Hands(max_num_hands=1)
    face = mp_face.FaceMesh(max_num_faces=1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_res = hands.process(rgb)
        f_res = face.process(rgb)

        if h_res.multi_hand_landmarks and f_res.multi_face_landmarks:
            row = [label]

            for lm in h_res.multi_hand_landmarks[0].landmark:
                row.extend([lm.x, lm.y, lm.z])

            for lm in f_res.multi_face_landmarks[0].landmark:
                row.extend([lm.x, lm.y, lm.z])

            writer.writerow(row)

    cap.release()


def main():
    os.makedirs("data", exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)

        for label in os.listdir(DATA_DIR):
            folder = os.path.join(DATA_DIR, label)
            if not os.path.isdir(folder):
                continue

            print(f"Processing {label}...")

            for video in os.listdir(folder):
                path = os.path.join(folder, video)
                extract_from_video(path, label, writer)

    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
