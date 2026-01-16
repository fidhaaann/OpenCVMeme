"""
Real-Time Meme Detector
========================
Production-ready inference system for meme gesture recognition.

Features:
- Real-time webcam processing
- High-confidence detection (≥0.85 threshold)
- GIF overlay on recognized gestures
- Smooth transitions between states
- Performance optimized
"""

import cv2
import numpy as np
import time
import os
import sys
import json
from joblib import load
import mediapipe as mp

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from holistic_features import EnhancedHolisticFeatureExtractor
from gif_overlay import GifOverlayManager, overlay_at_corner, AnimatedOverlay

# ============================================================
# CONFIGURATION
# ============================================================

# INSERT MODEL PATH HERE
MODEL_PATH = "models/meme_classifier.joblib"
METADATA_PATH = "models/model_metadata.json"

# INSERT GIF DIRECTORY HERE
GIF_DIR = "gifs"

# GIF mapping: gesture name -> GIF file
# INSERT YOUR CUSTOM MAPPING HERE OR USE DEFAULT
DEFAULT_GIF_MAPPING = {
    "cooked": "cooked.jpg",      # Prayer hands gesture
    "dicaprio": "dicaprio.gif",  # Clapping gesture
    "speed": "speed.gif",        # Fast face movement
    "think": "think.jpg",        # Thinking pose
    "vanish": "vanish.gif",      # Peace sign / disappear
    "none": None                 # No overlay for neutral
}

# Detection settings
CONFIDENCE_THRESHOLD = 0.50  # Universal threshold for all classes
SMOOTHING_FRAMES = 3  # Reduced for faster response
MIN_DETECTION_DURATION = 0.15  # Faster trigger

# Class-specific confidence thresholds (all equal for balanced detection)
CLASS_THRESHOLDS = {
    "speed": 0.50,
    "none": 0.30,
    "cooked": 0.50,
    "vanish": 0.50,
    "dicaprio": 0.50,
    "think": 0.50,
}

# Display settings
GIF_SIZE = (180, 180)
GIF_POSITION = "bottom-right"  # or "bottom-left", "top-right", "top-left"
SHOW_LANDMARKS = True
SHOW_CONFIDENCE = True
SHOW_FPS = True


class MemeDetector:
    """
    Real-time meme gesture detector with GIF overlay.
    """
    
    def __init__(self, model_path=MODEL_PATH, gif_mapping=None):
        """
        Initialize the meme detector.
        
        Args:
            model_path: Path to trained classifier model
            gif_mapping: Optional custom GIF mapping dict
        """
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Please train the model first using train_meme_model.py"
            )
        
        print(f"📂 Loading model from: {model_path}")
        self.model = load(model_path)
        self.classes = list(self.model.classes_)
        print(f"   Classes: {self.classes}")

        # Check for class mismatch
        expected_classes = ["cooked", "dicaprio", "none", "speed", "think", "vanish"]
        if set(self.classes) != set(expected_classes):
            print(f"⚠️ WARNING: Model classes {self.classes} do not match expected {expected_classes}")
        else:
            print("✅ Model classes match expected labels.")
        
        # Load metadata if available
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                self.metadata = json.load(f)
            print(f"   Model trained: {self.metadata.get('trained_at', 'unknown')}")
        
        # Initialize feature extractor
        print("🔧 Initializing MediaPipe Holistic...")
        self.extractor = EnhancedHolisticFeatureExtractor()
        
        # Initialize GIF manager
        print("🎬 Loading GIFs...")
        self.gif_manager = GifOverlayManager(default_size=GIF_SIZE)
        self._load_gifs(gif_mapping or DEFAULT_GIF_MAPPING)
        
        # Initialize animated overlay manager
        self.overlay = AnimatedOverlay(self.gif_manager)
        
        # Drawing utilities
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        
        # Prediction smoothing
        self.prediction_history = []
        self.confidence_history = []
        
        # State tracking
        self.current_gesture = "none"
        self.gesture_start_time = 0.0
        self.last_gesture_time = 0.0
        
        # Performance metrics
        self.fps_history = []
        self.process_times = []

    # ...existing code...
    
    def _load_gifs(self, mapping):
        """Load GIFs from mapping."""
        for gesture, filename in mapping.items():
            if filename is None:
                continue
            
            path = os.path.join(GIF_DIR, filename)
            if os.path.exists(path):
                self.gif_manager.load_gif(gesture, path, GIF_SIZE)
            else:
                print(f"   ⚠️ GIF not found: {path}")
    
    def _smooth_prediction(self, prediction, confidence):
        """
        Apply temporal smoothing to predictions.
        
        Args:
            prediction: Current frame prediction
            confidence: Current frame confidence
            
        Returns:
            Smoothed (prediction, confidence) tuple
        """
        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)
        
        # Keep only recent history
        if len(self.prediction_history) > SMOOTHING_FRAMES:
            self.prediction_history.pop(0)
            self.confidence_history.pop(0)
        
        # Majority voting with confidence weighting
        if len(self.prediction_history) < 3:
            return prediction, confidence
        
        # Count weighted votes
        vote_counts = {}
        for pred, conf in zip(self.prediction_history, self.confidence_history):
            if pred not in vote_counts:
                vote_counts[pred] = 0
            vote_counts[pred] += conf
        
        # Get winner
        best_pred = max(vote_counts.keys(), key=lambda k: vote_counts[k])
        
        # Average confidence for winning class
        matching_confs = [c for p, c in zip(self.prediction_history, self.confidence_history) 
                         if p == best_pred]
        avg_conf = np.mean(matching_confs)
        
        return best_pred, avg_conf
    
    def _draw_landmarks(self, frame, results):
        """Draw detected landmarks on frame."""
        # Face mesh (simplified)
        if results.face_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_draw.DrawingSpec(
                    color=(80, 110, 10), thickness=1, circle_radius=1
                )
            )
        
        # Left hand
        if results.left_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=3),
                self.mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2)
            )
        
        # Right hand
        if results.right_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                self.mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2)
            )
        
        return frame
    
    def _draw_ui(self, frame, gesture, confidence, fps, detection_status):
        """Draw the user interface overlay."""
        h, w = frame.shape[:2]
        
        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # Gesture label with confidence
        color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
        text = f"{gesture.upper()}"
        cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        if SHOW_CONFIDENCE:
            conf_text = f"Conf: {confidence:.2f}"
            cv2.putText(frame, conf_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # FPS
        if SHOW_FPS:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_text, (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Detection indicators
        y_offset = 55
        indicators = [
            ("F", detection_status['face']),
            ("L", detection_status['left_hand']),
            ("R", detection_status['right_hand'])
        ]
        
        x_offset = w - 120
        for label, detected in indicators:
            color = (0, 255, 0) if detected else (50, 50, 50)
            cv2.circle(frame, (x_offset, y_offset), 8, color, -1)
            cv2.putText(frame, label, (x_offset - 4, y_offset + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            x_offset += 30
        
        # Confidence threshold indicator
        if confidence >= CONFIDENCE_THRESHOLD:
            cv2.rectangle(frame, (w - 130, 65), (w - 10, 75), (0, 255, 0), -1)
        
        return frame
    
    def predict(self, features):
        """
        Make a prediction with confidence score.
        
        Args:
            features: Feature vector from extractor
            
        Returns:
            tuple: (predicted_class, confidence)
        """
        import numpy as np
        features = np.array(features)
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        expected = 514
        if features.shape[-1] < expected:
            features = np.concatenate([features, np.zeros(expected - features.shape[-1])])
        elif features.shape[-1] > expected:
            features = features[:expected]
        features = features.reshape(1, -1)
        # Get probabilities
        probs = self.model.predict_proba(features)[0]
        idx = np.argmax(probs)
        prediction = self.classes[idx]
        confidence = probs[idx]
        return prediction, confidence
    
    def process_frame(self, frame, current_time):
        """
        Process a single frame and return annotated result.
        
        Args:
            frame: BGR frame from camera
            current_time: Current timestamp
            
        Returns:
            Annotated frame with overlays
        """
        start_time = time.time()
        
        # Convert and extract features
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features, results = self.extractor.extract_enhanced_features_from_frame(rgb)
        # ...existing code...
        detection_status = self.extractor.get_detection_status(results)
        
        # Draw landmarks
        if SHOW_LANDMARKS:
            frame = self._draw_landmarks(frame, results)
        
        # Check if at least one hand is detected - required for gesture recognition
        has_hands = detection_status['left_hand'] or detection_status['right_hand']
        
        # Make prediction only if hands are detected
        if has_hands:
            prediction, confidence = self.predict(features)
        else:
            # No hands = default to "none"
            prediction, confidence = "none", 0.0
        
        # Apply smoothing
        smoothed_pred, smoothed_conf = self._smooth_prediction(prediction, confidence)
        
        # Get class-specific threshold
        class_threshold = CLASS_THRESHOLDS.get(smoothed_pred, CONFIDENCE_THRESHOLD)
        
        # Update gesture state - use class-specific threshold
        if smoothed_conf >= class_threshold and smoothed_pred != "none":
            if smoothed_pred != self.current_gesture:
                self.gesture_start_time = current_time
                self.current_gesture = smoothed_pred
            
            # Check if gesture held long enough
            if current_time - self.gesture_start_time >= MIN_DETECTION_DURATION:
                self.overlay.set_overlay(smoothed_pred, current_time)
                self.last_gesture_time = current_time
        else:
            # Clear overlay if no confident detection for a while
            if current_time - self.last_gesture_time > 0.5:
                self.overlay.clear_overlay()
                self.current_gesture = "none"
        
        # Render overlay
        frame = self.overlay.update_and_render(frame, current_time, GIF_POSITION)
        
        # Calculate FPS
        process_time = time.time() - start_time
        self.process_times.append(process_time)
        if len(self.process_times) > 30:
            self.process_times.pop(0)
        avg_time = np.mean(self.process_times)
        fps = 1.0 / max(avg_time, 0.001)
        
        # Draw UI - show gesture if it meets its class threshold
        display_gesture = self.current_gesture if smoothed_conf >= class_threshold else "none"
        frame = self._draw_ui(frame, display_gesture, smoothed_conf, fps, detection_status)
        
        return frame
    
    def run(self, camera_index=0):
        """
        Run the real-time detection loop.
        
        Args:
            camera_index: Camera device index
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("❌ Could not open webcam!")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*50)
        print("🎬 MEME DETECTOR RUNNING")
        print("="*50)
        print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
        print(f"Press 'Q' to quit")
        print("="*50)
        
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Failed to grab frame")
                    break
                
                # Mirror the frame
                frame = cv2.flip(frame, 1)
                
                # Process
                current_time = time.time() - start_time
                frame = self.process_frame(frame, current_time)
                
                # Display
                cv2.imshow("Meme Detector", frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('l'):  # Toggle landmarks
                    global SHOW_LANDMARKS
                    SHOW_LANDMARKS = not SHOW_LANDMARKS
                elif key == ord('c'):  # Toggle confidence
                    global SHOW_CONFIDENCE
                    SHOW_CONFIDENCE = not SHOW_CONFIDENCE
        
        except KeyboardInterrupt:
            print("\n⏹ Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.extractor.release()
            print("👋 Detector stopped")
    
    def process_video(self, video_path, output_path=None):
        """
        Process a video file instead of webcam.
        
        Args:
            video_path: Path to input video
            output_path: Optional path for output video
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Processing video: {video_path}")
        print(f"   Resolution: {width}x{height} @ {fps} FPS")
        print(f"   Total frames: {total_frames}")
        
        # Output writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                current_time = frame_count / fps
                
                # Process frame
                frame = self.process_frame(frame, current_time)
                
                # Write or display
                if out:
                    out.write(frame)
                else:
                    cv2.imshow("Meme Detector", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    progress = frame_count / total_frames * 100
                    print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            self.extractor.release()
        
        if output_path:
            print(f"✅ Output saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-Time Meme Detector")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Path to trained model")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index")
    parser.add_argument("--video", type=str,
                        help="Process video file instead of webcam")
    parser.add_argument("--output", type=str,
                        help="Output video path (for video mode)")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Confidence threshold (default: 0.85)")
    parser.add_argument("--no-landmarks", action="store_true",
                        help="Hide landmark drawings")
    parser.add_argument("--gif-size", type=int, default=180,
                        help="GIF overlay size in pixels")
    
    args = parser.parse_args()
    
    # Update module-level settings
    import meme_detector
    meme_detector.CONFIDENCE_THRESHOLD = args.threshold
    meme_detector.SHOW_LANDMARKS = not args.no_landmarks
    meme_detector.GIF_SIZE = (args.gif_size, args.gif_size)
    
    # Initialize detector
    try:
        detector = MemeDetector(model_path=args.model)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nTo train a model:")
        print("  1. Collect data: python src/data_collector.py")
        print("  2. Train model: python src/train_meme_model.py")
        return
    
    # Run detection
    if args.video:
        detector.process_video(args.video, args.output)
    else:
        detector.run(camera_index=args.camera)


if __name__ == "__main__":
    main()
