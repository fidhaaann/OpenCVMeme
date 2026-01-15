# OpenCVMeme 🎭

A production-ready real-time AI-powered gesture recognition system that detects your face and hand gestures to display matching memes. Built with Next.js, FastAPI, and MediaPipe.

## 🎨 Color Scheme
- **Primary Black**: `#000000`
- **Primary Purple**: `#9929EA`
- **Accent Pink**: `#FF5FCF`
- **Accent Yellow**: `#FAEB92`

## 🎯 Features

- **Real-time face + both hands tracking** using MediaPipe Holistic
- **Movement-invariant features** using relative landmark mapping
- **High-confidence classification** (≥0.85 threshold)
- **GIF overlay** on recognized gestures
- **Modern Web Interface** with Next.js + Tailwind CSS
- **Real-time WebSocket** streaming for low-latency detection
- **6 Meme Classes**: cooked, dicaprio, speed, think, vanish, none

## 🏗️ Project Structure

```
OpenCVMeme/
├── backend/                 # FastAPI Backend
│   ├── main.py             # API server with WebSocket support
│   └── requirements.txt    # Python dependencies
│
├── frontend/               # Next.js Frontend
│   ├── app/
│   │   ├── globals.css     # Global styles + Tailwind
│   │   ├── layout.tsx      # Root layout
│   │   └── page.tsx        # Main application page
│   ├── package.json        # Node.js dependencies
│   ├── tailwind.config.ts  # Tailwind configuration
│   └── .env.local          # Environment variables
│
├── src/                    # Core ML Components
│   ├── meme_detector.py    # Standalone detector
│   ├── holistic_features.py # MediaPipe feature extraction
│   └── gif_overlay.py      # GIF/image overlay utilities
│
├── models/                 # Trained ML models
│   └── meme_classifier.joblib
│
└── gifs/                   # Meme images/GIFs
```

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

## 🚀 Quick Start - Web App

### 1. Start the Backend (FastAPI)

```bash
# Navigate to backend directory
cd backend

# Install dependencies (if not already)
pip install -r requirements.txt

# Start the FastAPI server
python main.py
```

The API server will start at:
- **API**: `http://localhost:8000`
- **Docs**: `http://localhost:8000/docs`
- **WebSocket**: `ws://localhost:8000/ws`

### 2. Start the Frontend (Next.js)

```bash
# Open a new terminal and navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will start at `http://localhost:3000`

### 3. Use the Web App

1. Open your browser to `http://localhost:3000`
2. Click **"Start Detection"**
3. Allow camera permissions
4. Make gestures and watch the memes appear!

---

## 🎬 Standalone Mode (No Web)

### Run Real-Time Detection (Desktop)

```bash
python src/meme_detector.py
```

**Options:**
- `--threshold 0.85`: Set confidence threshold
- `--camera 0`: Select camera device
- `--no-landmarks`: Hide landmark drawings
- `--video path/to/video.mp4`: Process video file

---

## 🧠 Training Your Own Model

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

## 🎬 Meme Classes

| Key | Class | Gesture Description |
|-----|-------|---------------------|
| 1 | `cooked` | Prayer hands / clasped hands |
| 2 | `dicaprio` | Clapping / celebration |
| 3 | `speed` | Fast face movement / head shake |
| 4 | `think` | Finger to forehead / thinking pose |
| 5 | `vanish` | Peace sign / disappearing gesture |
| 6 | `none` | Neutral / no specific gesture |

## � API Endpoints

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/detect` | POST | Single frame detection |
| `/gif/{filename}` | GET | Serve meme images |
| `/gifs` | GET | List available memes |

### WebSocket

- **Endpoint**: `ws://localhost:8000/ws`
- **Protocol**: Send JSON `{ "image": "base64_data" }`, receive prediction results

## 🎨 UI Features

- **Modern Design**: Dark theme with gradient accents
- **Smooth Animations**: Framer Motion powered transitions
- **Real-time Feedback**: Live FPS counter, detection indicators
- **Responsive Layout**: Works on desktop and tablet
- **Glass Morphism**: Frosted glass effects
- **Glow Effects**: Dynamic hover and active states

## 🛠️ Tech Stack

### Frontend
- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Animations
- **Lucide React** - Icons

### Backend
- **FastAPI** - High-performance API
- **WebSockets** - Real-time communication
- **MediaPipe** - Face & hand detection
- **scikit-learn** - ML classification
- **OpenCV** - Image processing

## 📁 Full Project Structure

```
OpenCVMeme/
├── backend/                   # FastAPI Backend
│   ├── main.py               # WebSocket API server
│   └── requirements.txt      # Python dependencies
├── frontend/                  # Next.js Frontend
│   ├── app/
│   │   ├── globals.css       # Tailwind + custom styles
│   │   ├── layout.tsx        # Root layout
│   │   └── page.tsx          # Main app component
│   ├── package.json          # Node dependencies
│   ├── tailwind.config.ts    # Tailwind config
│   └── .env.local            # Environment variables
├── src/
│   ├── holistic_features.py   # Feature extraction (MediaPipe Holistic)
│   ├── data_collector.py      # Dataset collection tool
│   ├── train_meme_model.py    # Model training script
│   ├── meme_detector.py       # Real-time inference
│   ├── gif_overlay.py         # GIF rendering utilities
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

### Camera not working
- Check browser permissions for camera access
- Ensure no other application is using the camera
- Try a different browser (Chrome recommended)

### WebSocket connection failed
- Verify the backend is running on port 8000
- Check for firewall/antivirus blocking connections
- Ensure CORS is properly configured

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