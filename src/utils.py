"""
Utility Functions for Meme Gesture Recognition System
=====================================================
Provides GIF loading, overlay rendering, landmark normalization,
and other helper functions for the pipeline.
"""

import cv2
import numpy as np
from PIL import Image
import imageio
import os
from typing import List, Tuple, Optional, Dict


# --------------------------------------------------
# Load GIF and convert to OpenCV-compatible frames
# --------------------------------------------------
def load_gif_frames(path: str, size: Tuple[int, int] = (200, 200), 
                    max_frames: int = 100) -> List[np.ndarray]:
    """
    Loads a GIF and returns a list of BGRA frames.
    
    Args:
        path: Path to GIF file
        size: Target size (width, height)
        max_frames: Maximum frames to load
        
    Returns:
        List of BGRA numpy arrays
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"GIF not found: {path}")
    
    frames = []
    ext = os.path.splitext(path)[1].lower()
    
    try:
        if ext == '.gif':
            reader = imageio.get_reader(path)

            for i, frame in enumerate(reader):
                img = Image.fromarray(frame).convert("RGBA")
                img = img.resize(size, Image.Resampling.LANCZOS)
                frame_np = np.array(img)
                frame_bgra = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGRA)
                frames.append(frame_bgra)

                if i >= max_frames:
                    break
            
            reader.close()
        else:
            # Static image (jpg, png)
            img = Image.open(path).convert("RGBA")
            img = img.resize(size, Image.Resampling.LANCZOS)
            frame_np = np.array(img)
            frame_bgra = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGRA)
            frames.append(frame_bgra)
    
    except Exception as e:
        print(f"Error loading {path}: {e}")

    return frames


# --------------------------------------------------
# Overlay PNG/GIF frame onto camera frame
# --------------------------------------------------
def overlay_image_alpha(background: np.ndarray, overlay: np.ndarray, 
                        x: int, y: int) -> np.ndarray:
    """
    Overlays an RGBA/BGRA image on top of a BGR image with alpha blending.
    
    Args:
        background: BGR image (will be modified in place)
        overlay: BGRA overlay image
        x: X position (top-left)
        y: Y position (top-left)
        
    Returns:
        Modified background image
    """
    h, w = overlay.shape[:2]
    bg_h, bg_w = background.shape[:2]

    # Clamp to bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + w)
    y2 = min(bg_h, y + h)
    
    # Calculate overlay region
    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)
    
    if x2 <= x1 or y2 <= y1:
        return background

    # Extract alpha channel
    if overlay.shape[2] == 4:
        alpha = overlay[oy1:oy2, ox1:ox2, 3:4] / 255.0
        overlay_rgb = overlay[oy1:oy2, ox1:ox2, :3]
    else:
        alpha = np.ones((oy2-oy1, ox2-ox1, 1))
        overlay_rgb = overlay[oy1:oy2, ox1:ox2]

    # Alpha blend
    background[y1:y2, x1:x2] = (
        alpha * overlay_rgb +
        (1 - alpha) * background[y1:y2, x1:x2]
    ).astype(np.uint8)

    return background


def overlay_centered(background: np.ndarray, overlay: np.ndarray,
                     center_x: Optional[int] = None,
                     center_y: Optional[int] = None) -> np.ndarray:
    """
    Overlay image centered at position.
    
    Args:
        background: BGR background
        overlay: BGRA overlay
        center_x: X center (default: frame center)
        center_y: Y center (default: frame center)
        
    Returns:
        Modified background
    """
    bg_h, bg_w = background.shape[:2]
    ov_h, ov_w = overlay.shape[:2]
    
    if center_x is None:
        center_x = bg_w // 2
    if center_y is None:
        center_y = bg_h // 2
    
    x = center_x - ov_w // 2
    y = center_y - ov_h // 2
    
    return overlay_image_alpha(background, overlay, x, y)


def overlay_at_corner(background: np.ndarray, overlay: np.ndarray,
                      corner: str = "bottom-right",
                      margin: int = 20) -> np.ndarray:
    """
    Overlay image at specified corner.
    
    Args:
        background: BGR background
        overlay: BGRA overlay
        corner: "top-left", "top-right", "bottom-left", "bottom-right"
        margin: Margin from edge
        
    Returns:
        Modified background
    """
    bg_h, bg_w = background.shape[:2]
    ov_h, ov_w = overlay.shape[:2]
    
    if corner == "top-left":
        x, y = margin, margin
    elif corner == "top-right":
        x, y = bg_w - ov_w - margin, margin
    elif corner == "bottom-left":
        x, y = margin, bg_h - ov_h - margin
    else:  # bottom-right
        x, y = bg_w - ov_w - margin, bg_h - ov_h - margin
    
    return overlay_image_alpha(background, overlay, x, y)


# --------------------------------------------------
# Normalize landmarks (for ML stability)
# --------------------------------------------------
def normalize_landmarks(landmarks: List[Tuple[float, float, float]]) -> List[float]:
    """
    Converts landmark coordinates into normalized form relative to first point.
    
    Args:
        landmarks: List of (x, y, z) tuples
        
    Returns:
        Flattened list of normalized coordinates
    """
    if not landmarks:
        return []
    
    base_x, base_y, base_z = landmarks[0]
    normalized = []

    for x, y, z in landmarks:
        normalized.extend([
            x - base_x,
            y - base_y,
            z - base_z
        ])

    return normalized


def normalize_landmarks_scaled(landmarks: List[Tuple[float, float, float]], 
                                scale_factor: float = 1.0) -> List[float]:
    """
    Normalize landmarks with optional scaling.
    
    Args:
        landmarks: List of (x, y, z) tuples
        scale_factor: Scale to normalize by (e.g., hand size)
        
    Returns:
        Flattened list of normalized coordinates
    """
    if not landmarks or scale_factor == 0:
        return []
    
    base_x, base_y, base_z = landmarks[0]
    normalized = []

    for x, y, z in landmarks:
        normalized.extend([
            (x - base_x) / scale_factor,
            (y - base_y) / scale_factor,
            (z - base_z) / scale_factor
        ])

    return normalized


# --------------------------------------------------
# Drawing utilities
# --------------------------------------------------
def draw_text_with_background(frame: np.ndarray, text: str, 
                               position: Tuple[int, int],
                               font_scale: float = 1.0,
                               color: Tuple[int, int, int] = (255, 255, 255),
                               bg_color: Tuple[int, int, int] = (0, 0, 0),
                               thickness: int = 2,
                               padding: int = 5) -> np.ndarray:
    """
    Draw text with a background rectangle.
    
    Args:
        frame: Image to draw on
        text: Text to render
        position: (x, y) position
        font_scale: Font size
        color: Text color (BGR)
        bg_color: Background color (BGR)
        thickness: Text thickness
        padding: Padding around text
        
    Returns:
        Modified frame
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    cv2.rectangle(frame, 
                  (x - padding, y - text_h - padding),
                  (x + text_w + padding, y + baseline + padding),
                  bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
    
    return frame


def draw_confidence_bar(frame: np.ndarray, confidence: float,
                        position: Tuple[int, int],
                        size: Tuple[int, int] = (200, 20),
                        threshold: float = 0.85) -> np.ndarray:
    """
    Draw a confidence bar indicator.
    
    Args:
        frame: Image to draw on
        confidence: Confidence value (0-1)
        position: (x, y) position
        size: (width, height) of bar
        threshold: Threshold for color change
        
    Returns:
        Modified frame
    """
    x, y = position
    w, h = size
    
    # Background
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    
    # Fill
    fill_w = int(w * min(confidence, 1.0))
    color = (0, 255, 0) if confidence >= threshold else (0, 165, 255)
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)
    
    # Border
    cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
    
    # Threshold marker
    thresh_x = x + int(w * threshold)
    cv2.line(frame, (thresh_x, y), (thresh_x, y + h), (255, 255, 255), 2)
    
    return frame


# --------------------------------------------------
# Performance utilities
# --------------------------------------------------
class FPSCounter:
    """Simple FPS counter with smoothing."""
    
    def __init__(self, smoothing: int = 30):
        self.times = []
        self.smoothing = smoothing
        self.last_time = None
    
    def tick(self) -> float:
        """
        Record a frame and return current FPS.
        
        Returns:
            Smoothed FPS value
        """
        import time
        current_time = time.time()
        
        if self.last_time is not None:
            self.times.append(current_time - self.last_time)
            if len(self.times) > self.smoothing:
                self.times.pop(0)
        
        self.last_time = current_time
        
        if not self.times:
            return 0.0
        
        avg_time = sum(self.times) / len(self.times)
        return 1.0 / max(avg_time, 0.001)


class GestureBuffer:
    """Buffer for temporal gesture smoothing."""
    
    def __init__(self, buffer_size: int = 10):
        self.buffer_size = buffer_size
        self.predictions = []
        self.confidences = []
    
    def add(self, prediction: str, confidence: float):
        """Add a prediction to the buffer."""
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        
        if len(self.predictions) > self.buffer_size:
            self.predictions.pop(0)
            self.confidences.pop(0)
    
    def get_smoothed(self) -> Tuple[str, float]:
        """
        Get smoothed prediction using majority voting.
        
        Returns:
            (prediction, average_confidence) tuple
        """
        if not self.predictions:
            return "none", 0.0
        
        # Count votes weighted by confidence
        votes: Dict[str, float] = {}
        for pred, conf in zip(self.predictions, self.confidences):
            votes[pred] = votes.get(pred, 0) + conf
        
        # Get winner
        best_pred = max(votes.keys(), key=lambda k: votes[k])
        
        # Average confidence for winning class
        matching_confs = [c for p, c in zip(self.predictions, self.confidences) 
                         if p == best_pred]
        avg_conf = sum(matching_confs) / len(matching_confs)
        
        return best_pred, avg_conf
    
    def clear(self):
        """Clear the buffer."""
        self.predictions.clear()
        self.confidences.clear()


# --------------------------------------------------
# File utilities
# --------------------------------------------------
def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def get_available_gifs(gif_dir: str) -> Dict[str, str]:
    """
    Get mapping of available GIF files.
    
    Args:
        gif_dir: Directory to search
        
    Returns:
        Dict mapping name (without extension) to full path
    """
    gifs = {}
    if os.path.exists(gif_dir):
        for f in os.listdir(gif_dir):
            if f.lower().endswith(('.gif', '.jpg', '.png')):
                name = os.path.splitext(f)[0]
                gifs[name] = os.path.join(gif_dir, f)
    return gifs
