import cv2
import mediapipe as mp
from joblib import load
from collections import deque, Counter

# -------- LOAD MODELS --------
hand_model = load("models/hand_model.joblib")
face_model = load("models/face_model.joblib")

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hand_buffer = deque(maxlen=15)
face_buffer = deque(maxlen=15)


def majority(buffer):
    return Counter(buffer).most_common(1)[0][0] if buffer else None


cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(max_num_hands=2)
face = mp_face.FaceMesh(max_num_faces=1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h = hands.process(rgb)
    f = face.process(rgb)

    hand_pred = None
    face_pred = None

    # -------- HAND PREDICTION --------
    if h.multi_hand_landmarks:
        feat = []
        for i in range(2):
            if i < len(h.multi_hand_landmarks):
                for lm in h.multi_hand_landmarks[i].landmark:
                    feat.extend([lm.x, lm.y, lm.z])
            else:
                feat.extend([0] * 63)

        hand_pred = hand_model.predict([feat])[0]
        hand_buffer.append(hand_pred)

    # -------- FACE PREDICTION --------
    if f.multi_face_landmarks:
        feat = []
        for lm in f.multi_face_landmarks[0].landmark:
            feat.extend([lm.x, lm.y, lm.z])

        face_pred = face_model.predict([feat])[0]
        face_buffer.append(face_pred)

    # -------- FINAL DECISION --------
    final = None
    if hand_buffer:
        final = majority(hand_buffer)
    elif face_buffer:
        final = majority(face_buffer)

    # -------- DRAW LANDMARKS --------
    if h.multi_hand_landmarks:
        for lm in h.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    if f.multi_face_landmarks:
        for lm in f.multi_face_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_face.FACEMESH_TESSELATION)

    # -------- DISPLAY --------
    if final:
        cv2.putText(frame, f"Meme: {final}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Hand: {hand_pred}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Face: {face_pred}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Gesture Meme System (Dual Model)", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
