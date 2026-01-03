import cv2
import mediapipe as mp
from joblib import load
from collections import deque, Counter

# ---------------- LOAD MODEL ----------------
model = load("models/gesture_model.joblib")

# ---------------- SMOOTHING ----------------
PREDICTION_WINDOW = 20
prediction_buffer = deque(maxlen=PREDICTION_WINDOW)

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils


def extract_features(frame, hands, face):
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

    # ---------------- SAME SCALING ----------------
    hand_features = [v * 2.0 for v in hand_features]
    face_features = [v * 0.3 for v in face_features]

    return hand_features + face_features, h_res, f_res


def main():
    cap = cv2.VideoCapture(0)

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    face = mp_face.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        features, h_res, f_res = extract_features(frame, hands, face)

        # ---- DRAW LANDMARKS ----
        if h_res.multi_hand_landmarks:
            for hand_landmarks in h_res.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        if f_res.multi_face_landmarks:
            for face_landmarks in f_res.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    frame, face_landmarks, mp_face.FACEMESH_TESSELATION
                )

        # ---- PREDICTION ----
        raw_pred = model.predict([features])[0]
        prediction_buffer.append(raw_pred)

        if len(prediction_buffer) == PREDICTION_WINDOW:
            final_pred = Counter(prediction_buffer).most_common(1)[0][0]
            confidence = Counter(prediction_buffer)[final_pred] / PREDICTION_WINDOW
        else:
            final_pred = raw_pred
            confidence = 0.0

        # ---- DISPLAY ----
        cv2.putText(frame, f"Gesture: {final_pred}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2)

        if confidence > 0:
            cv2.putText(frame, f"Confidence: {confidence:.2f}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2)

        cv2.imshow("Gesture Meme (Hacky Scaled)", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
