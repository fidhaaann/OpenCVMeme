# OpenCVMeme 🎭

A production-ready Python system for recognizing 6 specific memes based on high-precision facial expressions and dual-hand gestures using OpenCV + MediaPipe Holistic.

## 🎯 Features

- **Real-time face + both hands tracking** using MediaPipe Holistic
- **Movement-invariant features** using relative landmark mapping
- **High-confidence classification** (≥0.85 threshold)
- **GIF overlay** on recognized gestures
- **6 Meme Classes**: cooked, dicaprio, speed, think, vanish, none

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/OpenCVMeme.git
cd OpenCVMeme

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Collect Training Data

```bash
python src/data_collector.py
```

**Controls:**
- `1-6`: Select meme class
- `SPACE`: Start/stop continuous recording
- `S`: Save single sample
- `Q`: Quit and save dataset

### 2. Train the Model

```bash
python src/train_meme_model.py
```

**Options:**
- `--model auto`: Try all classifiers and pick best (default)
- `--model svm`: Train SVM only
- `--model gb`: Train Gradient Boosting only
- `--no-tune`: Skip hyperparameter tuning

### 3. Run Real-Time Detection

```bash
python src/meme_detector.py
```

**Options:**
- `--threshold 0.85`: Set confidence threshold
- `--camera 0`: Select camera device
- `--no-landmarks`: Hide landmark drawings
- `--video path/to/video.mp4`: Process video file

## 🎬 Meme Classes

| Key | Class | Gesture Description |
|-----|-------|---------------------|
| 1 | `cooked` | Prayer hands / clasped hands |
| 2 | `dicaprio` | Clapping / celebration |
| 3 | `speed` | Fast face movement / head shake |
| 4 | `think` | Finger to forehead / thinking pose |
| 5 | `vanish` | Peace sign / disappearing gesture |
| 6 | `none` | Neutral / no specific gesture |

## 📁 Project Structure

```
OpenCVMeme/
├── src/
│   ├── holistic_features.py   # Feature extraction (MediaPipe Holistic)
│   ├── data_collector.py      # Dataset collection tool
│   ├── train_meme_model.py    # Model training script
│   ├── meme_detector.py       # Real-time inference
│   ├── gif_overlay.py         # GIF rendering utilities
│   ├── extract_features.py    # Feature extraction wrapper
│   └── utils.py               # Helper functions
├── data/
│   ├── meme_features.csv      # Training dataset (generated)
│   ├── photos/                # Captured photos
│   └── videos/                # Captured videos
├── models/
│   └── meme_classifier.joblib # Trained model (generated)
├── gifs/                      # GIF files for overlays
│   ├── cooked.jpg
│   ├── dicaprio.gif
│   ├── speed.gif
│   ├── think.jpg
│   └── vanish.gif
├── requirements.txt
└── README.md
```

## 🔧 Feature Extraction

### Face Features
- **Anchor**: Nose bridge (movement invariance)
- **Regions**: Eyes, eyebrows, lips, cheeks, jaw
- **All coordinates** expressed as deltas relative to nose anchor

### Hand Features
- **21 landmarks** per hand
- **Normalized** relative to palm center
- **Scale-invariant** using hand size normalization

### Fused Features
```
[left_hand (63)] + [right_hand (63)] + [face (N*3)] + [additional (52)]
```

## 🎓 Training Pipeline

1. **Data Collection**: Interactive webcam capture with labeling
2. **Feature Extraction**: Relative coordinate mapping
3. **Model Selection**: SVM (RBF), Gradient Boosting, Random Forest
4. **Hyperparameter Tuning**: Grid search with cross-validation
5. **Probability Calibration**: For reliable confidence scores

## ⚙️ Configuration

### GIF Mapping
Edit `src/meme_detector.py`:
```python
DEFAULT_GIF_MAPPING = {
    "cooked": "cooked.jpg",
    "dicaprio": "dicaprio.gif",
    "speed": "speed.gif",
    "think": "think.jpg",
    "vanish": "vanish.gif",
    "none": None
}
```

### Detection Settings
```python
CONFIDENCE_THRESHOLD = 0.85  # Minimum confidence to trigger
SMOOTHING_FRAMES = 5         # Temporal smoothing window
MIN_DETECTION_DURATION = 0.3 # Seconds before overlay appears
```

## 📊 Model Performance

After training, you'll see metrics like:
```
🎯 Best Model: SVM
   F1 Score: 0.9234
   Accuracy: 0.9156
   CV F1 Score: 0.9112 (+/- 0.0234)
```

## 🔄 Batch Processing

### Process videos for training data:
```bash
# Single video
python src/data_collector.py --mode video --video path/to/video.mp4 --label cooked

# Batch from organized folders
python src/data_collector.py --mode batch --video-dir data/videos
```

### Process video for inference:
```bash
python src/meme_detector.py --video input.mp4 --output output.mp4
```

## 🛠️ Troubleshooting

### Low Detection Confidence
- Ensure good lighting
- Face the camera directly
- Collect more training samples for problematic classes

### GIF Not Showing
- Check `gifs/` directory for matching files
- Verify file extensions match the mapping

### High CPU Usage
- Reduce camera resolution
- Set `--no-landmarks` flag
- Use `model_complexity=1` in holistic settings

## 📜 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request