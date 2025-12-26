import cv2
import mediapipe as mp
import csv
import os

DATA_PATH = "data/gestures.csv"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def flatten_landmarks(landmarks):
    return [v for lm in landmarks.landmark for v in (lm.x, lm.y, lm.z)]

def main():
    os.makedirs("data", exist_ok=True)

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=1)

    with open(DATA_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        print("Press R to record | Q to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Recorder", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r') and results.multi_hand_landmarks:
                label = input("Label (e.g. cooked, think, speed): ")
                landmarks = results.multi_hand_landmarks[0]
                row = [label] + flatten_landmarks(landmarks)
                writer.writerow(row)
                print(f"Saved sample for: {label}")

            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
