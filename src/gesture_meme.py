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


def extract_features(frame, hands, face):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h = hands.process(rgb)
    f = face.process(rgb)

    features = []

    # ---------------- HAND FEATURES ----------------
    if h.multi_hand_landmarks:
        for i in range(2):
            if i < len(h.multi_hand_landmarks):
                for lm in h.multi_hand_landmarks[i].landmark:
                    features.extend([lm.x, lm.y, lm.z])
            else:
                features.extend([0] * 63)
    else:
        features.extend([0] * 126)

    # ---------------- FACE FEATURES ----------------
    if f.multi_face_landmarks:
        for lm in f.multi_face_landmarks[0].landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0] * (468 * 3))

    return features, h, f


def majority_vote(buffer):
    return Counter(buffer).most_common(1)[0][0] if buffer else None


cap = cv2.VideoCapture(0)

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    features, h, f = extract_features(frame, hands, face)

    # Draw landmarks
    if h.multi_hand_landmarks:
        for lm in h.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    if f.multi_face_landmarks:
        for lm in f.multi_face_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_face.FACEMESH_TESSELATION)

    # Prediction
    probs = model.predict_proba([features])[0]
    classes = model.classes_
    pred = classes[np.argmax(probs)]
    confidence = np.max(probs)

    pred_buffer.append(pred)
    final_pred = majority_vote(pred_buffer)

    # ---------------- SPEED GATING ----------------
    if final_pred == "speed":
        # Show speed ONLY if confidence is strong
        if confidence < 0.60:
            final_pred = "none"

    # Display
    cv2.putText(frame, f"Meme: {final_pred}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Gesture Meme System", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
