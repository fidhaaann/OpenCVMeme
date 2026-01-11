"""
GIF Overlay Utilities
=====================
Efficient GIF loading and overlay functions for real-time video processing.

Features:
- Pre-loaded GIF frame caching
- Alpha-blended overlay
- Position-aware rendering
- FPS-synchronized playback
"""

import cv2
import numpy as np
from PIL import Image
import imageio
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class GifInfo:
    """Container for GIF data and playback state."""
    frames: List[np.ndarray]  # List of BGRA frames
    frame_count: int
    original_fps: float
    current_frame: int = 0
    last_update_time: float = 0.0


class GifOverlayManager:
    """
    Manages multiple GIF overlays for real-time video rendering.
    """
    
    def __init__(self, default_size: Tuple[int, int] = (200, 200)):
        """
        Initialize the GIF manager.
        
        Args:
            default_size: Default (width, height) for GIF rendering
        """
        self.default_size = default_size
        self.gifs: Dict[str, GifInfo] = {}
        self.current_gif: Optional[str] = None
        self.display_start_time: float = 0.0
        
    def load_gif(self, name: str, path: str, size: Optional[Tuple[int, int]] = None,
                 max_frames: int = 100) -> bool:
        """
        Load a GIF file and cache its frames.
        
        Args:
            name: Identifier for this GIF
            path: Path to the GIF file (supports .gif, .jpg, .png)
            size: Optional custom size (width, height)
            max_frames: Maximum frames to load
            
        Returns:
            True if successfully loaded
        """
        if not os.path.exists(path):
            print(f"⚠️ GIF not found: {path}")
            return False
        
        size = size or self.default_size
        frames = []
        fps = 30.0  # Default FPS
        
        try:
            ext = os.path.splitext(path)[1].lower()
            
            if ext == '.gif':
                # Load animated GIF
                reader = imageio.get_reader(path)
                
                # Get GIF metadata for FPS
                try:
                    meta = reader.get_meta_data()
                    duration = meta.get('duration', 100)  # ms per frame
                    fps = 1000.0 / max(duration, 1)
                except:
                    pass
                
                for i, frame in enumerate(reader):
                    if i >= max_frames:
                        break
                    
                    # Convert to RGBA
                    img = Image.fromarray(frame)
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    
                    # Resize
                    img = img.resize(size, Image.Resampling.LANCZOS)
                    
                    # Convert to BGRA for OpenCV
                    frame_np = np.array(img)
                    frame_bgra = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGRA)
                    frames.append(frame_bgra)
                
                reader.close()
                
            else:
                # Load static image (jpg, png)
                img = Image.open(path)
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                img = img.resize(size, Image.Resampling.LANCZOS)
                frame_np = np.array(img)
                frame_bgra = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGRA)
                frames.append(frame_bgra)
                fps = 1.0  # Single frame
            
            if len(frames) == 0:
                print(f"⚠️ No frames loaded from: {path}")
                return False
            
            self.gifs[name] = GifInfo(
                frames=frames,
                frame_count=len(frames),
                original_fps=fps
            )
            
            print(f"✅ Loaded '{name}': {len(frames)} frames @ {fps:.1f} FPS")
            return True
            
        except Exception as e:
            print(f"❌ Error loading GIF '{path}': {e}")
            return False
    
    def load_all_from_mapping(self, mapping: Dict[str, str], 
                               base_dir: str = "",
                               size: Optional[Tuple[int, int]] = None):
        """
        Load multiple GIFs from a mapping dictionary.
        
        Args:
            mapping: Dict mapping gesture names to file paths
            base_dir: Base directory to prepend to paths
            size: Optional custom size
        """
        for name, path in mapping.items():
            full_path = os.path.join(base_dir, path) if base_dir else path
            self.load_gif(name, full_path, size)
    
    def get_current_frame(self, name: str, current_time: float) -> Optional[np.ndarray]:
        """
        Get the current frame for a GIF based on elapsed time.
        
        Args:
            name: GIF identifier
            current_time: Current time in seconds
            
        Returns:
            BGRA frame array or None if not found
        """
        if name not in self.gifs:
            return None
        
        gif = self.gifs[name]
        
        # Calculate which frame to display
        if gif.frame_count == 1:
            return gif.frames[0]
        
        # Time-based frame selection
        elapsed = current_time - gif.last_update_time
        frames_elapsed = int(elapsed * gif.original_fps)
        
        if frames_elapsed > 0:
            gif.current_frame = (gif.current_frame + frames_elapsed) % gif.frame_count
            gif.last_update_time = current_time
        
        return gif.frames[gif.current_frame]
    
    def reset_gif(self, name: str):
        """Reset a GIF to its first frame."""
        if name in self.gifs:
            self.gifs[name].current_frame = 0
            self.gifs[name].last_update_time = 0.0
    
    def reset_all(self):
        """Reset all GIFs to their first frames."""
        for name in self.gifs:
            self.reset_gif(name)


def overlay_image_alpha(background: np.ndarray, overlay: np.ndarray, 
                        x: int, y: int) -> np.ndarray:
    """
    Overlay an RGBA/BGRA image on a BGR background with alpha blending.
    
    Args:
        background: BGR image (will be modified in place)
        overlay: BGRA overlay image
        x: X position (top-left corner)
        y: Y position (top-left corner)
        
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
    
    # Extract alpha channel and normalize
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
    Overlay an image centered at a specific position.
    
    Args:
        background: BGR background image
        overlay: BGRA overlay image
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
    Overlay an image at a specific corner.
    
    Args:
        background: BGR background image
        overlay: BGRA overlay image
        corner: "top-left", "top-right", "bottom-left", "bottom-right"
        margin: Margin from edge in pixels
        
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


def overlay_at_face(background: np.ndarray, overlay: np.ndarray,
                    face_bbox: Tuple[int, int, int, int],
                    position: str = "above",
                    offset: int = 20) -> np.ndarray:
    """
    Overlay an image relative to a detected face.
    
    Args:
        background: BGR background image
        overlay: BGRA overlay image
        face_bbox: (x, y, width, height) of face bounding box
        position: "above", "below", "left", "right", "center"
        offset: Offset from face in pixels
        
    Returns:
        Modified background
    """
    fx, fy, fw, fh = face_bbox
    ov_h, ov_w = overlay.shape[:2]
    
    face_center_x = fx + fw // 2
    face_center_y = fy + fh // 2
    
    if position == "above":
        x = face_center_x - ov_w // 2
        y = fy - ov_h - offset
    elif position == "below":
        x = face_center_x - ov_w // 2
        y = fy + fh + offset
    elif position == "left":
        x = fx - ov_w - offset
        y = face_center_y - ov_h // 2
    elif position == "right":
        x = fx + fw + offset
        y = face_center_y - ov_h // 2
    else:  # center
        x = face_center_x - ov_w // 2
        y = face_center_y - ov_h // 2
    
    return overlay_image_alpha(background, overlay, x, y)


def create_text_overlay(text: str, font_scale: float = 1.0,
                        color: Tuple[int, int, int] = (255, 255, 255),
                        bg_color: Tuple[int, int, int] = (0, 0, 0),
                        padding: int = 10) -> np.ndarray:
    """
    Create a text overlay image with background.
    
    Args:
        text: Text to render
        font_scale: Font size scale
        color: Text color (BGR)
        bg_color: Background color (BGR)
        padding: Padding around text
        
    Returns:
        BGRA image with text
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(font_scale * 2))
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Create image
    img_w = text_w + 2 * padding
    img_h = text_h + baseline + 2 * padding
    
    img = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    
    # Fill background
    img[:, :, :3] = bg_color
    img[:, :, 3] = 200  # Semi-transparent
    
    # Draw text
    text_x = padding
    text_y = padding + text_h
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color + (255,), thickness)
    
    return img


class AnimatedOverlay:
    """
    Manages animated overlays with smooth transitions.
    """
    
    def __init__(self, gif_manager: GifOverlayManager):
        self.gif_manager = gif_manager
        self.current_name: Optional[str] = None
        self.target_name: Optional[str] = None
        self.alpha: float = 1.0
        self.fade_speed: float = 15.0  # Fade per second (faster)
        self.hold_time: float = 0.0
        self.min_hold_time: float = 0.1  # Minimum display time (faster switching)
    
    def set_overlay(self, name: str, current_time: float):
        """
        Set the current overlay with smooth transition.
        
        Args:
            name: GIF identifier
            current_time: Current time in seconds
        """
        if name != self.current_name:
            if self.current_name is None or current_time - self.hold_time > self.min_hold_time:
                self.target_name = name
                self.hold_time = current_time
                self.gif_manager.reset_gif(name)
    
    def clear_overlay(self):
        """Clear the current overlay."""
        self.target_name = None
    
    def update_and_render(self, frame: np.ndarray, current_time: float,
                          position: str = "bottom-right") -> np.ndarray:
        """
        Update transition state and render overlay.
        
        Args:
            frame: BGR frame to overlay on
            current_time: Current time in seconds
            position: Corner position for overlay
            
        Returns:
            Frame with overlay
        """
        # Update transition
        if self.target_name != self.current_name:
            self.alpha -= self.fade_speed * 0.033  # ~30 FPS
            if self.alpha <= 0:
                self.current_name = self.target_name
                self.alpha = 0.0
        else:
            self.alpha = min(1.0, self.alpha + self.fade_speed * 0.033)
        
        # Render current overlay
        if self.current_name is not None and self.alpha > 0:
            gif_frame = self.gif_manager.get_current_frame(self.current_name, current_time)
            if gif_frame is not None:
                # Apply fade alpha
                if self.alpha < 1.0:
                    gif_frame = gif_frame.copy()
                    gif_frame[:, :, 3] = (gif_frame[:, :, 3] * self.alpha).astype(np.uint8)
                
                frame = overlay_at_corner(frame, gif_frame, position)
        
        return frame


if __name__ == "__main__":
    # Test GIF overlay functionality
    print("Testing GIF Overlay System...")
    
    manager = GifOverlayManager(default_size=(150, 150))
    
    # Try to load available GIFs
    gif_dir = "gifs"
    if os.path.exists(gif_dir):
        for f in os.listdir(gif_dir):
            name = os.path.splitext(f)[0]
            path = os.path.join(gif_dir, f)
            manager.load_gif(name, path)
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        import time
        start_time = time.time()
        gif_names = list(manager.gifs.keys())
        current_idx = 0
        
        print("\nPress SPACE to cycle through GIFs, Q to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            current_time = time.time() - start_time
            
            if gif_names:
                gif_frame = manager.get_current_frame(gif_names[current_idx], current_time)
                if gif_frame is not None:
                    frame = overlay_at_corner(frame, gif_frame, "bottom-right")
                
                cv2.putText(frame, f"GIF: {gif_names[current_idx]}", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("GIF Overlay Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and gif_names:
                current_idx = (current_idx + 1) % len(gif_names)
                manager.reset_gif(gif_names[current_idx])
        
        cap.release()
        cv2.destroyAllWindows()
    
    print("Test complete!")
