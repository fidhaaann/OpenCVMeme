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

    # ---------------- HAND FEATURES ----------------
    hand_feat = None

    # ACCEPT EVEN IF ONLY ONE HAND IS DETECTED
    if h.multi_hand_landmarks:
        hand_feat = []

        for i in range(2):  # always build 2-hand feature vector
            if i < len(h.multi_hand_landmarks):
                for lm in h.multi_hand_landmarks[i].landmark:
                    hand_feat.extend([lm.x, lm.y, lm.z])
            else:
                # pad missing second hand
                hand_feat.extend([0] * 63)

    # ---------------- FACE FEATURES ----------------
    face_feat = None

    if f.multi_face_landmarks:
        face_feat = []
        for lm in f.multi_face_landmarks[0].landmark:
            face_feat.extend([lm.x, lm.y, lm.z])

    return hand_feat, face_feat


def main():
    # 🔥 ULTRA-PERMISSIVE HAND DETECTOR
    hands = mp_hands.Hands(
        static_image_mode=False,         # keep tracking across frames
        max_num_hands=2,
        min_detection_confidence=0.1,    # MUCH LOWER
        min_tracking_confidence=0.1,     # MUCH LOWER
        model_complexity=1
    )

    # Face model unchanged
    face = mp_face.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

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

                    # ---------- PHOTOS ----------
                    if source == "photos":
                        frame = cv2.imread(path)
                        if frame is None:
                            continue

                        # 🔥 Re-run with STATIC mode for still images
                        static_hands = mp_hands.Hands(
                            static_image_mode=True,
                            max_num_hands=2,
                            min_detection_confidence=0.1,
                            model_complexity=1
                        )

                        hand_f, face_f = extract_features(frame, static_hands, face)

                        if hand_f is not None:
                            wh.writerow([label] + hand_f)

                        if face_f is not None:
                            wf.writerow([label] + face_f)

                        static_hands.close()

                    # ---------- VIDEOS ----------
                    else:
                        cap = cv2.VideoCapture(path)
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            hand_f, face_f = extract_features(frame, hands, face)

                            if hand_f is not None:
                                wh.writerow([label] + hand_f)

                            if face_f is not None:
                                wf.writerow([label] + face_f)

                        cap.release()

    print("✅ Feature extraction complete with ultra-permissive detection.")


if __name__ == "__main__":
    main()
