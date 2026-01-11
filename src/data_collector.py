"""
Meme Dataset Collector
======================
Capture face + both hands data for training the meme classifier.
Labels each of the 6 memes and saves feature vectors to CSV.

Usage:
    python data_collector.py

Controls:
    1-6: Select meme class
    SPACE: Start/stop recording
    S: Save single sample
    Q: Quit and save dataset
"""

import cv2
import numpy as np
import pandas as pd
import os
import sys
import time
from datetime import datetime
import mediapipe as mp

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from holistic_features import EnhancedHolisticFeatureExtractor

# ============================================================
# CONFIGURATION
# ============================================================

# INSERT DATASET DIRECTORY HERE
DATASET_DIR = "data/meme_dataset"
CSV_OUTPUT = "data/meme_features.csv"

# 6 Meme classes
MEME_CLASSES = {
    1: "cooked",      # Prayer hands / clasped hands
    2: "dicaprio",    # Clapping / celebration gesture
    3: "speed",       # Fast face movement / head shake
    4: "think",       # Finger to forehead / thinking pose
    5: "vanish",      # Peace sign / disappearing gesture
    6: "none"         # No specific gesture (neutral)
}

# Recording settings
SAMPLES_PER_SECOND = 10  # Samples to capture per second during recording
MIN_SAMPLES_PER_CLASS = 100  # Minimum recommended samples


class MemeDataCollector:
    """
    Interactive data collection tool for meme gesture training.
    """
    
    def __init__(self):
        self.extractor = EnhancedHolisticFeatureExtractor()
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        
        # Data storage
        self.samples = []
        self.labels = []
        
        # State
        self.current_class = 6  # Default to "none"
        self.recording = False
        self.last_sample_time = 0
        
        # Statistics
        self.class_counts = {name: 0 for name in MEME_CLASSES.values()}
        
        # Create output directories
        os.makedirs(DATASET_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
        
        # Load existing data if available
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing dataset if available."""
        if os.path.exists(CSV_OUTPUT):
            try:
                df = pd.read_csv(CSV_OUTPUT)
                if 'label' in df.columns:
                    self.labels = df['label'].tolist()
                    self.samples = df.drop('label', axis=1).values.tolist()
                    
                    # Update counts
                    for label in self.labels:
                        if label in self.class_counts:
                            self.class_counts[label] += 1
                    
                    print(f"📂 Loaded {len(self.samples)} existing samples")
                    self._print_class_counts()
            except Exception as e:
                print(f"⚠️ Could not load existing data: {e}")
    
    def _print_class_counts(self):
        """Print current sample counts per class."""
        print("\n📊 Sample counts:")
        for name, count in self.class_counts.items():
            status = "✅" if count >= MIN_SAMPLES_PER_CLASS else "⚠️"
            print(f"  {status} {name}: {count}")
        print()
    
    def _draw_ui(self, frame, detection_status):
        """Draw the user interface overlay on the frame."""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay for UI
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h-150), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # Current class
        class_name = MEME_CLASSES[self.current_class]
        color = (0, 255, 0) if self.recording else (255, 255, 255)
        cv2.putText(frame, f"Class: {class_name.upper()} ({self.current_class})", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Recording status
        if self.recording:
            cv2.putText(frame, "⏺ RECORDING", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # Blinking dot
            if int(time.time() * 2) % 2:
                cv2.circle(frame, (w - 50, 40), 15, (0, 0, 255), -1)
        
        # Detection status
        y_pos = h - 130
        cv2.putText(frame, "Detection Status:", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_pos += 25
        face_color = (0, 255, 0) if detection_status['face'] else (100, 100, 100)
        cv2.putText(frame, f"Face: {'✓' if detection_status['face'] else '✗'}", 
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)
        
        left_color = (0, 255, 0) if detection_status['left_hand'] else (100, 100, 100)
        cv2.putText(frame, f"Left Hand: {'✓' if detection_status['left_hand'] else '✗'}", 
                    (150, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)
        
        right_color = (0, 255, 0) if detection_status['right_hand'] else (100, 100, 100)
        cv2.putText(frame, f"Right Hand: {'✓' if detection_status['right_hand'] else '✗'}", 
                    (320, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
        
        # Sample counts
        y_pos += 30
        cv2.putText(frame, f"Total samples: {len(self.samples)} | {class_name}: {self.class_counts[class_name]}", 
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls
        y_pos += 30
        cv2.putText(frame, "Controls: 1-6=Select Class | SPACE=Record | S=Single Sample | Q=Quit+Save", 
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Class legend
        y_pos += 25
        legend = " | ".join([f"{k}:{v}" for k, v in MEME_CLASSES.items()])
        cv2.putText(frame, legend, (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return frame
    
    def _draw_landmarks(self, frame, results):
        """Draw detected landmarks on the frame."""
        # Draw face landmarks
        if results.face_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_draw.DrawingSpec(
                    color=(80, 110, 10), thickness=1, circle_radius=1
                )
            )
        
        # Draw left hand
        if results.left_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                self.mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
            )
        
        # Draw right hand
        if results.right_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                self.mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
        return frame
    
    def _add_sample(self, features, label):
        """Add a sample to the dataset."""
        self.samples.append(features.tolist())
        self.labels.append(label)
        self.class_counts[label] += 1
    
    def save_dataset(self):
        """Save the collected dataset to CSV."""
        if len(self.samples) == 0:
            print("⚠️ No samples to save!")
            return
        
        # Find the maximum feature length across all samples
        max_len = max(len(s) for s in self.samples)
        
        # Pad shorter samples with zeros to ensure consistent length
        padded_samples = []
        for s in self.samples:
            if len(s) < max_len:
                s = s + [0.0] * (max_len - len(s))
            padded_samples.append(s)
        
        # Create DataFrame with correct number of columns
        feature_cols = [f"f_{i}" for i in range(max_len)]
        df = pd.DataFrame(padded_samples, columns=feature_cols)
        df['label'] = self.labels
        
        # Save to CSV
        df.to_csv(CSV_OUTPUT, index=False)
        print(f"\n💾 Dataset saved to {CSV_OUTPUT}")
        print(f"   Total samples: {len(self.samples)}")
        self._print_class_counts()
        
        # Save backup with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(DATASET_DIR, f"meme_features_backup_{timestamp}.csv")
        df.to_csv(backup_path, index=False)
        print(f"   Backup saved to {backup_path}")
    
    def run(self):
        """Run the data collection interface."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open webcam!")
            return
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n🎥 Meme Data Collector Started")
        print("=" * 50)
        print("Classes:")
        for key, name in MEME_CLASSES.items():
            print(f"  {key}: {name}")
        print("=" * 50)
        print("\nControls:")
        print("  1-6: Select meme class")
        print("  SPACE: Start/stop continuous recording")
        print("  S: Save single sample")
        print("  D: Delete last sample")
        print("  Q: Quit and save dataset")
        print("=" * 50)
        
        sample_interval = 1.0 / SAMPLES_PER_SECOND
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Extract features
                features, results = self.extractor.extract_enhanced_features_from_frame(rgb)
                detection_status = self.extractor.get_detection_status(results)
                
                # Draw landmarks
                frame = self._draw_landmarks(frame, results)
                
                # Continuous recording
                if self.recording:
                    current_time = time.time()
                    if current_time - self.last_sample_time >= sample_interval:
                        if self.extractor.has_valid_detection(results):
                            label = MEME_CLASSES[self.current_class]
                            self._add_sample(features, label)
                            self.last_sample_time = current_time
                
                # Draw UI
                frame = self._draw_ui(frame, detection_status)
                
                cv2.imshow("Meme Data Collector", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Space - toggle recording
                    self.recording = not self.recording
                    if self.recording:
                        self.last_sample_time = time.time()
                        print(f"🔴 Recording started for: {MEME_CLASSES[self.current_class]}")
                    else:
                        print(f"⏹ Recording stopped. Samples: {self.class_counts[MEME_CLASSES[self.current_class]]}")
                elif key == ord('s'):  # Single sample with 3-second timer
                    print("📸 Get ready! Taking sample in 3 seconds...")
                    countdown_start = time.time()
                    
                    # 3-second countdown loop
                    while time.time() - countdown_start < 3:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame = cv2.flip(frame, 1)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Extract features and draw landmarks
                        features, results = self.extractor.extract_enhanced_features_from_frame(rgb)
                        detection_status = self.extractor.get_detection_status(results)
                        frame = self._draw_landmarks(frame, results)
                        frame = self._draw_ui(frame, detection_status)
                        
                        # Draw countdown
                        remaining = 3 - int(time.time() - countdown_start)
                        h, w = frame.shape[:2]
                        cv2.putText(frame, str(remaining), (w//2 - 50, h//2 + 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
                        cv2.putText(frame, "GET READY!", (w//2 - 150, h//2 - 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                        
                        cv2.imshow("Meme Data Collector", frame)
                        cv2.waitKey(1)
                    
                    # Capture final frame after countdown
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        features, results = self.extractor.extract_enhanced_features_from_frame(rgb)
                        
                        if self.extractor.has_valid_detection(results):
                            label = MEME_CLASSES[self.current_class]
                            self._add_sample(features, label)
                            print(f"✅ Sample saved for: {label}")
                        else:
                            print("⚠️ No valid detection for sample!")
                elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
                    self.current_class = int(chr(key))
                    print(f"🏷️ Selected class: {MEME_CLASSES[self.current_class]}")
                elif key == ord('d'):  # Delete last sample
                    if len(self.samples) > 0:
                        removed_label = self.labels.pop()
                        self.samples.pop()
                        self.class_counts[removed_label] -= 1
                        print(f"🗑️ Deleted last sample ({removed_label}). Total: {len(self.samples)}")
                    else:
                        print("⚠️ No samples to delete!")
                    
        except KeyboardInterrupt:
            print("\n⏹ Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.extractor.release()
            
            # Save dataset
            self.save_dataset()


def collect_from_video(video_path, label, output_csv=CSV_OUTPUT):
    """
    Collect samples from a pre-recorded video file.
    
    Args:
        video_path: Path to the video file
        label: Class label for all frames in this video
        output_csv: Output CSV path
    """
    extractor = EnhancedHolisticFeatureExtractor()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    samples = []
    labels = []
    frame_count = 0
    valid_count = 0
    
    print(f"📹 Processing video: {video_path}")
    print(f"   Label: {label}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Sample every 3rd frame to avoid too similar samples
        if frame_count % 3 != 0:
            continue
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features, results = extractor.extract_enhanced_features_from_frame(rgb)
        
        if extractor.has_valid_detection(results):
            samples.append(features.tolist())
            labels.append(label)
            valid_count += 1
        
        # Progress
        if frame_count % 100 == 0:
            print(f"   Progress: {frame_count}/{total_frames} frames, {valid_count} valid samples")
    
    cap.release()
    extractor.release()
    
    print(f"✅ Extracted {valid_count} samples from {frame_count} frames")
    
    # Load existing data
    existing_samples = []
    existing_labels = []
    
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        if 'label' in df.columns:
            existing_labels = df['label'].tolist()
            existing_samples = df.drop('label', axis=1).values.tolist()
    
    # Merge
    all_samples = existing_samples + samples
    all_labels = existing_labels + labels
    
    # Save
    feature_cols = [f"f_{i}" for i in range(len(all_samples[0]))]
    df = pd.DataFrame(all_samples, columns=feature_cols)
    df['label'] = all_labels
    df.to_csv(output_csv, index=False)
    
    print(f"💾 Saved to {output_csv}")
    print(f"   Total samples: {len(all_samples)}")


def batch_collect_from_videos(video_dir, output_csv=CSV_OUTPUT):
    """
    Batch process videos organized by class folders.
    
    Expected structure:
    video_dir/
        cooked/
            video1.mp4
            video2.mp4
        dicaprio/
            ...
    
    Args:
        video_dir: Root directory containing class folders
        output_csv: Output CSV path
    """
    if not os.path.exists(video_dir):
        print(f"❌ Directory not found: {video_dir}")
        return
    
    print(f"📂 Processing videos from: {video_dir}")
    
    for class_name in os.listdir(video_dir):
        class_path = os.path.join(video_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        if class_name not in MEME_CLASSES.values():
            print(f"⚠️ Skipping unknown class: {class_name}")
            continue
        
        print(f"\n📁 Processing class: {class_name}")
        
        for video_file in os.listdir(class_path):
            if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(class_path, video_file)
                collect_from_video(video_path, class_name, output_csv)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Meme Gesture Data Collector")
    parser.add_argument("--mode", choices=["interactive", "video", "batch"], 
                        default="interactive",
                        help="Collection mode: interactive (webcam), video (single file), batch (folder)")
    parser.add_argument("--video", type=str, help="Video file path (for video mode)")
    parser.add_argument("--label", type=str, help="Label for video mode")
    parser.add_argument("--video-dir", type=str, default="data/videos",
                        help="Video directory for batch mode")
    parser.add_argument("--output", type=str, default=CSV_OUTPUT,
                        help="Output CSV path")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        collector = MemeDataCollector()
        collector.run()
    
    elif args.mode == "video":
        if not args.video or not args.label:
            print("❌ Video mode requires --video and --label arguments")
        else:
            collect_from_video(args.video, args.label, args.output)
    
    elif args.mode == "batch":
        batch_collect_from_videos(args.video_dir, args.output)
