import cv2
import os
import time
import mediapipe as mp

SAVE_DIR = "data/videos"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

def main():
    cap = cv2.VideoCapture(0)

    # Configure trackers
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

    # Video writer settings
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Press R to start/stop recording | Q to quit")

    recording = False
    writer = None
    current_label = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run trackers
        h_res = hands.process(rgb)
        f_res = face.process(rgb)

        # Draw landmarks
        if h_res.multi_hand_landmarks:
            for hand_lm in h_res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
        if f_res.multi_face_landmarks:
            for face_lm in f_res.multi_face_landmarks:
                mp_draw.draw_landmarks(frame, face_lm, mp_face.FACEMESH_TESSELATION)

        # Status text
        hands_count = len(h_res.multi_hand_landmarks) if h_res.multi_hand_landmarks else 0
        faces_count = len(f_res.multi_face_landmarks) if f_res.multi_face_landmarks else 0
        cv2.putText(frame, f"Hands: {hands_count} | Faces: {faces_count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Write video while recording
        if recording and writer is not None:
            writer.write(frame)
            cv2.putText(frame, f"REC: {current_label}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Recorder", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            if not recording:
                # Start recording
                current_label = input("Gesture name: ").strip() or "unnamed"
                out_dir = os.path.join(SAVE_DIR, current_label)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{int(time.time())}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                recording = True
                print(f"Recording started: {out_path}")
            else:
                # Stop recording
                recording = False
                if writer is not None:
                    writer.release()
                    writer = None
                print("Recording stopped")

        elif key == ord('q'):
            break

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
