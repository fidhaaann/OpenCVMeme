import cv2
import numpy as np
from PIL import Image
import imageio


# --------------------------------------------------
# Load GIF and convert to OpenCV-compatible frames
# --------------------------------------------------
def load_gif_frames(path, size=(200, 200), max_frames=40):
    """
    Loads a GIF and returns a list of BGRA frames.
    """
    frames = []
    reader = imageio.get_reader(path)

    for i, frame in enumerate(reader):
        img = Image.fromarray(frame).convert("RGBA")
        img = img.resize(size)
        frame_np = np.array(img)
        frame_bgra = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGRA)
        frames.append(frame_bgra)

        if i >= max_frames:
            break

    return frames


# --------------------------------------------------
# Overlay PNG/GIF frame onto camera frame
# --------------------------------------------------
def overlay_image_alpha(background, overlay, x, y):
    """
    Overlays an RGBA image on top of a BGR image.
    """
    h, w = overlay.shape[:2]
    bg_h, bg_w = background.shape[:2]

    if x + w > bg_w or y + h > bg_h:
        return background

    alpha = overlay[:, :, 3] / 255.0

    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            alpha * overlay[:, :, c] +
            (1 - alpha) * background[y:y+h, x:x+w, c]
        )

    return background


# --------------------------------------------------
# Normalize landmarks (for ML stability)
# --------------------------------------------------
def normalize_landmarks(landmarks):
    """
    Converts landmark coordinates into normalized form.
    Used before feeding to ML models.
    """
    base_x, base_y, base_z = landmarks[0]
    normalized = []

    for x, y, z in landmarks:
        normalized.extend([
            x - base_x,
            y - base_y,
            z - base_z
        ])

    return normalized
