import cv2
import os
import csv
import mediapipe as mp

DATA_DIR = "data"
HAND_CSV = "data/gestures_hands.csv"
FACE_CSV = "data/gestures_face.csv"

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh


def extract_features(frame, hands, face):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h = hands.process(rgb)
    f = face.process(rgb)

    # ---------- HAND FEATURES ----------
    hand_feat = []
    if h.multi_hand_landmarks:
        for i in range(2):
            if i < len(h.multi_hand_landmarks):
                for lm in h.multi_hand_landmarks[i].landmark:
                    hand_feat.extend([lm.x, lm.y, lm.z])
            else:
                hand_feat.extend([0] * 63)
    else:
        hand_feat.extend([0] * 126)

    # ---------- FACE FEATURES ----------
    face_feat = []
    if f.multi_face_landmarks:
        for lm in f.multi_face_landmarks[0].landmark:
            face_feat.extend([lm.x, lm.y, lm.z])
    else:
        face_feat.extend([0] * (468 * 3))

    return hand_feat, face_feat


def main():
    hands = mp_hands.Hands(max_num_hands=2)
    face = mp_face.FaceMesh(max_num_faces=1)

    with open(HAND_CSV, "w", newline="") as fh, open(FACE_CSV, "w", newline="") as ff:
        wh, wf = csv.writer(fh), csv.writer(ff)

        for source in ["photos", "videos"]:
            root = os.path.join(DATA_DIR, source)
            for label in os.listdir(root):
                folder = os.path.join(root, label)
                if not os.path.isdir(folder):
                    continue

                for item in os.listdir(folder):
                    path = os.path.join(folder, item)

                    if source == "photos":
                        frame = cv2.imread(path)
                        if frame is None:
                            continue
                        h, f = extract_features(frame, hands, face)
                        wh.writerow([label] + h)
                        wf.writerow([label] + f)

                    else:
                        cap = cv2.VideoCapture(path)
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            h, f = extract_features(frame, hands, face)
                            wh.writerow([label] + h)
                            wf.writerow([label] + f)
                        cap.release()

    print("Feature extraction done.")


if __name__ == "__main__":
    main()
