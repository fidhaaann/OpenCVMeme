import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)
face = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# ---------------- UTILS ----------------
def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# ---------------- SPEED: FACE MOTION ----------------
prev_face_center = None

def detect_speed(face_bbox):
    global prev_face_center

    x, y, w, h = face_bbox
    cx, cy = x + w // 2, y + h // 2
    curr_center = np.array([cx, cy])

    if prev_face_center is None:
        prev_face_center = curr_center
        return False

    movement = np.linalg.norm(curr_center - prev_face_center)
    prev_face_center = curr_center

    return movement > 12   # 🔧 increase if too sensitive

# ---------------- HAND FEATURES ----------------
def hand_points(hand, w, h):
    lm = hand.landmark
    pts = {}
    for i in range(21):
        pts[i] = (int(lm[i].x * w), int(lm[i].y * h))
    return pts

# ---------------- COOKED (PRAYER HANDS) ----------------
def detect_cooked(h1, h2):
    return distance(h1[0], h2[0]) < 80

# ---------------- DICAPRIO (CLAP) ----------------
def detect_dicaprio(h1, h2):
    return distance(h1[9], h2[9]) < 100

# ---------------- THINK (FINGER TO FOREHEAD) ----------------
def detect_think(hand, face_bbox):
    x, y, w, h = face_bbox
    forehead = (x + w // 2, y + int(h * 0.2))
    index_tip = hand[8]
    return distance(index_tip, forehead) < 80

# ---------------- VANISH (SLANTED PEACE) ----------------
def detect_vanish(hand):
    index = hand[8]
    middle = hand[12]
    ring = hand[16]
    pinky = hand[20]

    spread = distance(index, middle)
    folded = distance(middle, ring) < 40 and distance(ring, pinky) < 40
    slope = abs(index[1] - middle[1])

    return spread > 60 and folded and slope > 30

# ---------------- MAIN ----------------
def main():
    cap = cv2.VideoCapture(0)
    print("🎥 Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = hands.process(rgb)
        face_results = face.process(rgb)

        label = "none"
        face_bbox = None

        # ---- FACE DETECTION ----
        if face_results.detections:
            det = face_results.detections[0]
            bboxC = det.location_data.relative_bounding_box
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            bw = int(bboxC.width * w)
            bh = int(bboxC.height * h)
            face_bbox = (x, y, bw, bh)

            cv2.rectangle(frame, (x,y), (x+bw, y+bh), (255,255,0), 2)

            # SPEED = FACE MOTION
            if detect_speed(face_bbox):
                label = "speed"

        # ---- HAND DETECTION ----
        if hand_results.multi_hand_landmarks:
            all_hands = []
            for hand in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                all_hands.append(hand_points(hand, w, h))

            # Two hands
            if len(all_hands) == 2:
                if detect_cooked(all_hands[0], all_hands[1]):
                    label = "cooked"
                elif detect_dicaprio(all_hands[0], all_hands[1]):
                    label = "dicaprio"

            # One hand
            if len(all_hands) == 1 and face_bbox is not None:
                if detect_think(all_hands[0], face_bbox):
                    label = "think"
                elif detect_vanish(all_hands[0]):
                    label = "vanish"

        # ---- DISPLAY ----
        cv2.putText(frame, f"Gesture: {label.upper()}",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 255, 0), 3)

        cv2.imshow("Gesture System (Stable)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
