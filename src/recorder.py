import cv2
import os
import time
import mediapipe as mp

# ================= CONFIG =================
BASE_DIR = "data"
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
PHOTO_DIR = os.path.join(BASE_DIR, "photos")

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(PHOTO_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils


def main():
    print("\n=== Gesture Recorder ===")
    gesture = input("Enter gesture name: ").strip()

    video_path = os.path.join(VIDEO_DIR, gesture)
    photo_path = os.path.join(PHOTO_DIR, gesture)
    os.makedirs(video_path, exist_ok=True)
    os.makedirs(photo_path, exist_ok=True)

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

    print("\nControls:")
    print(" R → Start recording")
    print(" S → Stop recording")
    print(" P → Take photo")
    print(" Q → Quit")

    recording = False
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---- PROCESS DETECTION ----
        hand_results = hands.process(rgb)
        face_results = face.process(rgb)

        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Draw face landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    frame, face_landmarks, mp_face.FACEMESH_TESSELATION
                )

        # UI text
        cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if recording:
            cv2.putText(frame, "RECORDING...", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Gesture Recorder", frame)

        key = cv2.waitKey(1) & 0xFF

        # ---- RECORD VIDEO ----
        if key == ord('r'):
            filename = f"{gesture}_{int(time.time())}.mp4"
            filepath = os.path.join(video_path, filename)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                filepath,
                fourcc,
                20,
                (frame.shape[1], frame.shape[0])
            )
            recording = True
            print(f"Recording started: {filepath}")

        # ---- STOP RECORDING ----
        elif key == ord('s') and recording:
            recording = False
            out.release()
            print("Recording stopped")

        # ---- TAKE PHOTO ----
        elif key == ord('p'):
            filename = f"{gesture}_{int(time.time())}.jpg"
            filepath = os.path.join(photo_path, filename)
            cv2.imwrite(filepath, frame)
            print(f"Photo saved: {filepath}")

        # ---- EXIT ----
        elif key == ord('q'):
            break

        if recording:
            out.write(frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
