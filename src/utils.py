import cv2
import numpy as np
from PIL import Image
import imageio

def load_gif_frames(path, size=(200, 200), max_frames=40):
    frames = []
    reader = imageio.get_reader(path)
    for i, frame in enumerate(reader):
        img = Image.fromarray(frame).convert("RGBA")
        img = img.resize(size)
        frames.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA))
        if i >= max_frames:
            break
    return frames


def overlay_image_alpha(background, overlay, x, y):
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
