import cv2
import mediapipe as mp
import numpy as np
import os
import csv

DATA_DIR = "data"
OUT_CSV = "data/gestures.csv"

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
face = mp_face.FaceMesh(static_image_mode=True)

# ---------- HAND FEATURES (20) ----------
def extract_hand_features(hand_landmarks):
    lm = hand_landmarks.landmark
    WRIST = 0
    BASE = 9
    TIPS = [4, 8, 12, 16, 20]

    pts = {i: np.array([lm[i].x, lm[i].y, lm[i].z]) for i in range(21)}
    size = np.linalg.norm(pts[WRIST] - pts[BASE]) + 1e-6
    feats = []

    # distances wrist → fingertips
    for tip in TIPS:
        feats.append(np.linalg.norm(pts[WRIST] - pts[tip]) / size)

    # fingertip-to-fingertip distances
    for a, b in [(4,8),(8,12),(12,16),(16,20),(4,20)]:
        feats.append(np.linalg.norm(pts[a] - pts[b]) / size)

    # angular relationships
    for a, b in [(4,8),(8,12),(12,16),(16,20),(4,20)]:
        va = pts[a] - pts[WRIST]
        vb = pts[b] - pts[WRIST]
        cos = np.dot(va, vb)/(np.linalg.norm(va)*np.linalg.norm(vb)+1e-6)
        feats.append(cos)

    while len(feats) < 20:
        feats.append(0.0)

    return feats

# ---------- FACE FEATURES (4096) ----------
def extract_face_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cx, cy = w//2, h//2
    size = min(w, h)//2

    crop = gray[cy-size//2:cy+size//2, cx-size//2:cx+size//2]
    crop = cv2.resize(crop, (64,64))
    crop = crop.astype("float32") / 255.0
    return crop.flatten().tolist()

# ---------- PROCESS IMAGE ----------
def process_image(path, label, writer):
    img = cv2.imread(path)
    if img is None:
        return

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hand_res = hands.process(rgb)

    hand_feats = [0]*20
    if hand_res.multi_hand_landmarks:
        hand_feats = extract_hand_features(hand_res.multi_hand_landmarks[0])

    face_feats = extract_face_features(img)
    writer.writerow(hand_feats + face_feats + [label])

# ---------- PROCESS VIDEO ----------
def process_video(path, label, writer):
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_res = hands.process(rgb)

        hand_feats = [0]*20
        if hand_res.multi_hand_landmarks:
            hand_feats = extract_hand_features(hand_res.multi_hand_landmarks[0])

        face_feats = extract_face_features(frame)
        writer.writerow(hand_feats + face_feats + [label])

    cap.release()

# ---------- MAIN ----------
def main():
    os.makedirs("data", exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"h{i}" for i in range(20)] + [f"f{i}" for i in range(4096)] + ["label"]
        writer.writerow(header)

        for src in ["photos", "videos"]:
            base = os.path.join(DATA_DIR, src)
            for label in os.listdir(base):
                folder = os.path.join(base, label)
                if not os.path.isdir(folder):
                    continue

                print(f"📂 {src}/{label}")
                for file in os.listdir(folder):
                    path = os.path.join(folder, file)
                    if file.endswith((".jpg", ".png")):
                        process_image(path, label, writer)
                    elif file.endswith(".mp4"):
                        process_video(path, label, writer)

    print("✅ Feature extraction complete → data/gestures.csv")

if __name__ == "__main__":
    main()
