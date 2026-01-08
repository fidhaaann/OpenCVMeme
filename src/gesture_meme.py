import cv2
import mediapipe as mp
from joblib import load
from collections import deque, Counter
import numpy as np

model = load("models/gesture_model.joblib")

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

PRED_WINDOW = 15
pred_buffer = deque(maxlen=PRED_WINDOW)


def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def hand_shape_features(hand_landmarks):
    lm = hand_landmarks.landmark
    WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 0,4,8,12,16,20
    pts = {i: (lm[i].x, lm[i].y, lm[i].z) for i in range(21)}
    hand_size = dist(pts[WRIST], pts[MIDDLE_TIP]) + 1e-6

    features = []
    for tip in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
        features.append(dist(pts[WRIST], pts[tip]) / hand_size)

    for a,b in [(THUMB_TIP,INDEX_TIP),(INDEX_TIP,MIDDLE_TIP),(MIDDLE_TIP,RING_TIP),(RING_TIP,PINKY_TIP),(THUMB_TIP,PINKY_TIP)]:
        features.append(dist(pts[a], pts[b]) / hand_size)

    return features


def detect_speed(face_landmarks):
    """
    Simple rule: mouth open + eyebrows raised
    """
    lm = face_landmarks.landmark
    upper_lip = lm[13].y
    lower_lip = lm[14].y
    mouth_open = abs(lower_lip - upper_lip)

    left_eyebrow = lm[70].y
    right_eyebrow = lm[300].y
    left_eye = lm[159].y
    right_eye = lm[386].y

    eyebrow_raise = (left_eye - left_eyebrow + right_eye - right_eyebrow) / 2

    return mouth_open > 0.03 and eyebrow_raise > 0.02


def majority_vote(buf):
    return Counter(buf).most_common(1)[0][0] if buf else None


cap = cv2.VideoCapture(0)

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.1, min_tracking_confidence=0.1)
face = mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h = hands.process(rgb)
    f = face.process(rgb)

    # Draw
    if h.multi_hand_landmarks:
        for lm in h.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    if f.multi_face_landmarks:
        for lm in f.multi_face_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_face.FACEMESH_TESSELATION)

    final_pred = "none"

    # ---------------- SPEED CHECK ----------------
    if f.multi_face_landmarks:
        if detect_speed(f.multi_face_landmarks[0]):
            final_pred = "speed"
        else:
            # ---------------- HAND MODEL ----------------
            if h.multi_hand_landmarks:
                feats = []
                for i in range(2):
                    if i < len(h.multi_hand_landmarks):
                        feats.extend(hand_shape_features(h.multi_hand_landmarks[i]))
                    else:
                        feats.extend([0] * 10)

                probs = model.predict_proba([feats])[0]
                classes = model.classes_
                pred = classes[np.argmax(probs)]
                pred_buffer.append(pred)
                final_pred = majority_vote(pred_buffer)

    cv2.putText(frame, f"Meme: {final_pred}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gesture Meme System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
