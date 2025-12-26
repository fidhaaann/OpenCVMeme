import cv2
import mediapipe as mp
import numpy as np
import os
import time

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

def main():
    # Prepare video recording (optional, toggled by 'r')
    SAVE_DIR = "data/videos"
    os.makedirs(SAVE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    face = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Init writer lazily after camera reports props
        if 'writer' not in locals():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            recording = False
            writer = None
            current_label = None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = hands.process(rgb)
        face_results = face.process(rgb)

        # ---- Draw hands ----
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # ---- Draw face ----
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face.FACEMESH_TESSELATION
                )

        # ---- Write video while recording ----
        if recording and writer is not None:
            writer.write(frame)
            cv2.putText(
                frame,
                f"REC: {current_label}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ---- Status text ----
        hands_count = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0
        faces_count = len(face_results.multi_face_landmarks) if face_results.multi_face_landmarks else 0
        cv2.putText(
            frame,
            f"Hands: {hands_count} | Faces: {faces_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Hands + Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        # Toggle recording with 'r'
        if key == ord('r'):
            if not recording:
                current_label = (input("Gesture name: ") or "unnamed").strip()
                out_dir = os.path.join(SAVE_DIR, current_label)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{int(time.time())}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, float(fps), (width, height))
                recording = True
                print(f"Recording started: {out_path}")
            else:
                recording = False
                if writer is not None:
                    writer.release()
                    writer = None
                print("Recording stopped")

        # ESC to exit
        if key == 27:
            break

    cap.release()
    if 'writer' in locals() and writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
