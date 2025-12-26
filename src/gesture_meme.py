import cv2
import mediapipe as mp
import numpy as np
from joblib import load
from utils import load_gif_frames, overlay_image_alpha

# Try loading the trained model; fall back to rule-based detection if unavailable
try:
    model = load("models/gesture_model.joblib")
except Exception as e:
    print(f"Warning: failed to load model (using fallback): {e}")
    model = None

GIFS = {
    "cooked": "gifs/cooked.jpg",
    "dicaprio": "gifs/dicaprio.gif",
    "speed": "gifs/speed.gif",
    "think": "gifs/think.jpg",
    "vanish": "gifs/vanish.gif"
}

gif_cache = {k: load_gif_frames(v) for k, v in GIFS.items()}

mp_hands = mp.solutions.hands

# Simple heuristic gesture detector used when model is unavailable
def detect_gesture(pts):
    def up(a, b):
        return a[1] < b[1]

    thumb = up(pts[4], pts[2])
    index = up(pts[8], pts[6])
    middle = up(pts[12], pts[10])
    ring = up(pts[16], pts[14])
    pinky = up(pts[20], pts[18])

    # Map heuristic gestures to existing GIF keys
    if thumb and not index and not middle and not ring and not pinky:
        return "dicaprio"  # thumbs up
    if index and middle and not ring and not pinky:
        return "speed"     # peace
    if thumb and index and middle and ring and pinky:
        return "think"     # open palm
    if not thumb and not index and not middle and not ring and not pinky:
        return "vanish"    # fist
    return None

def main():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            # Prepare features and pixel points
            features = []
            for p in lm.landmark:
                features.extend([p.x, p.y, p.z])
            pts = [(int(p.x * frame.shape[1]), int(p.y * frame.shape[0])) for p in lm.landmark]

            pred_key = None
            if model is not None:
                try:
                    pred_key = model.predict([features])[0]
                except Exception as e:
                    print(f"Warning: model prediction failed, using fallback: {e}")
            if pred_key is None:
                pred_key = detect_gesture(pts)

            overlay = gif_cache.get(pred_key)
            if overlay is not None:
                frame = overlay_image_alpha(frame, overlay[0], 20, 20)

        cv2.imshow("Gesture Meme", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
