import cv2
import mediapipe as mp
import os
import time

DATA_PATH = "data/gestures.csv"  # kept for compatibility but unused for video capture
VIDEO_ROOT = "data/videos"

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

def main():
    os.makedirs(VIDEO_ROOT, exist_ok=True)

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=2)
    face = mp_face.FaceMesh(max_num_faces=1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Press R to start/stop recording video clips | Q to quit")

    recording = False
    writer = None
    current_label = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h_res = hands.process(rgb)
        f_res = face.process(rgb)

        if h_res.multi_hand_landmarks:
            for h in h_res.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, h, mp_hands.HAND_CONNECTIONS
                )
        if f_res.multi_face_landmarks:
            for f in f_res.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, f, mp_face.FACEMESH_TESSELATION
                )

        if recording and writer is not None:
            writer.write(frame)
            cv2.putText(frame, f"REC: {current_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Recorder", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            if not recording:
                current_label = input("Label for clip: ").strip() or "unnamed"
                ts = int(time.time())
                out_dir = os.path.join(VIDEO_ROOT, current_label)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{current_label}_{ts}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                recording = True
                print(f"Recording started: {out_path}")
            else:
                recording = False
                if writer is not None:
                    writer.release()
                    writer = None
                print("Recording stopped")

        elif key == ord("q"):
            break

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
