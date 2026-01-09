import cv2
import numpy as np
import mediapipe as mp
from joblib import load

# ================== PATH ==================
HAND_MODEL_PATH = "models/hand_gesture_model.joblib"

# ================== LOAD MODEL ==================
hand_model = load(HAND_MODEL_PATH)

# ================== MEDIAPIPE ==================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================== HAND FEATURES ==================
def extract_hand_features(hand_landmarks):
    lm = hand_landmarks.landmark

    WRIST = 0
    TIPS = [4, 8, 12, 16, 20]

    pts = {i: np.array([lm[i].x, lm[i].y, lm[i].z]) for i in range(21)}
    hand_size = np.linalg.norm(pts[WRIST] - pts[12]) + 1e-6

    features = []
    for tip in TIPS:
        features.append(np.linalg.norm(pts[WRIST] - pts[tip]) / hand_size)

    for a, b in [(4,8),(8,12),(12,16),(16,20),(4,20)]:
        features.append(np.linalg.norm(pts[a] - pts[b]) / hand_size)

    return features


def classify_hand(hand_results):
    if not hand_results.multi_hand_landmarks:
        return "none"

    feats = []

    for i in range(2):
        if i < len(hand_results.multi_hand_landmarks):
            feats.extend(extract_hand_features(hand_results.multi_hand_landmarks[i]))
        else:
            feats.extend([0] * 10)

    return hand_model.predict([feats])[0]


# ================== MAIN LOOP ==================
def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---- HAND DETECTION ----
        hand_results = hands.process(rgb)

        if hand_results.multi_hand_landmarks:
            for hl in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        label = classify_hand(hand_results)

        # ---- DISPLAY ----
        cv2.putText(frame, f"Gesture: {label.upper()}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)

        cv2.imshow("Gesture Meme System (Hand Only)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
