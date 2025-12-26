import cv2
import mediapipe as mp
import numpy as np
from joblib import load
from utils import load_gif_frames, overlay_image_alpha

model = load("models/gesture_model.joblib")

GIFS = {
    "cooked": "gifs/cooked.jpg",
    "dicaprio": "gifs/dicaprio.gif",
    "speed": "gifs/speed.gif",
    "think": "gifs/think.jpg",
    "vanish": "gifs/vanish.gif"
}

gif_cache = {k: load_gif_frames(v) for k, v in GIFS.items()}

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

def main():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=2)
    face = mp_face.FaceMesh(max_num_faces=1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h_res = hands.process(rgb)
        f_res = face.process(rgb)

        if h_res.multi_hand_landmarks and f_res.multi_face_landmarks:
            h = h_res.multi_hand_landmarks[0]
            f = f_res.multi_face_landmarks[0]

            features = []
            for lm in h.landmark:
                features.extend([lm.x, lm.y, lm.z])
            for lm in f.landmark:
                features.extend([lm.x, lm.y, lm.z])

            pred = model.predict([features])[0]
            overlay = gif_cache.get(pred)

            if overlay:
                frame = overlay_image_alpha(frame, overlay[0], 20, 20)

        cv2.imshow("Gesture Meme", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
