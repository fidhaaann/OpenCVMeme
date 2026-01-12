"""
Meme Detector Web App
======================
Simple Flask web app that runs the meme detector in a browser.
Camera on left, meme on right - just like meme_detector.py but in a web browser.
"""

import os
import sys
import cv2
import numpy as np
import base64
import json
from flask import Flask, render_template_string, request, jsonify, send_from_directory
from flask_cors import CORS
from joblib import load

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from holistic_features import EnhancedHolisticFeatureExtractor

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "meme_classifier.joblib")
GIF_DIR = os.path.join(BASE_DIR, "gifs")

CONFIDENCE_THRESHOLD = 0.85  # Same as meme_detector.py
EXPECTED_FEATURES = 514

GIF_MAPPING = {
    "cooked": "cooked.jpg",
    "dicaprio": "dicaprio.gif",
    "speed": "speed.gif",
    "think": "think.jpg",
    "vanish": "vanish.gif",
    "none": None
}

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)

# Global model and extractor
model = None
extractor = None

def load_model():
    global model, extractor
    if model is None:
        print(f"📂 Loading model from: {MODEL_PATH}")
        model = load(MODEL_PATH)
        print(f"   Classes: {list(model.classes_)}")
        print("🔧 Initializing MediaPipe...")
        extractor = EnhancedHolisticFeatureExtractor()
        print("✅ Ready!")
    return model is not None

def predict(image_data):
    """Process image and return prediction."""
    if model is None or extractor is None:
        return None
    
    # Convert to RGB
    if len(image_data.shape) == 2:
        rgb = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
    elif image_data.shape[2] == 4:
        rgb = cv2.cvtColor(image_data, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    
    # Extract features
    features, results = extractor.extract_enhanced_features_from_frame(rgb)
    detection_status = extractor.get_detection_status(results)
    
    # Prepare features
    features = features.reshape(1, -1)
    features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Pad/truncate to expected size
    if features.shape[1] < EXPECTED_FEATURES:
        features = np.hstack([features, np.zeros((1, EXPECTED_FEATURES - features.shape[1]))])
    elif features.shape[1] > EXPECTED_FEATURES:
        features = features[:, :EXPECTED_FEATURES]
    
    # Predict
    probs = model.predict_proba(features)[0]
    idx = np.argmax(probs)
    prediction = model.classes_[idx]
    confidence = float(probs[idx])
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'is_confident': confidence >= CONFIDENCE_THRESHOLD,
        'detection_status': detection_status,
        'gif': GIF_MAPPING.get(prediction) if confidence >= CONFIDENCE_THRESHOLD else None
    }

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/detect', methods=['POST'])
def detect():
    load_model()
    
    try:
        data = request.get_json()
        image_b64 = data['image']
        
        # Decode base64
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        
        image_bytes = base64.b64decode(image_b64)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        result = predict(image)
        
        if result:
            # Print to console like meme_detector.py does
            status = "✓" if result['is_confident'] else " "
            print(f"{status} {result['prediction'].upper():10} {result['confidence']:.2f}")
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/gifs/<filename>')
def serve_gif(filename):
    return send_from_directory(GIF_DIR, filename)

# ============================================================
# HTML TEMPLATE (All-in-one)
# ============================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎭 Meme Detector</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            overflow: hidden;
        }
        
        .container {
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        
        /* Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 2rem;
            background: rgba(0,0,0,0.3);
        }
        
        header h1 { color: #F7A5A5; font-size: 1.5rem; }
        
        .stats {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .fps {
            font-family: monospace;
            color: #00ff88;
            background: rgba(0,255,136,0.1);
            padding: 0.3rem 0.8rem;
            border-radius: 4px;
        }
        
        .indicators { display: flex; gap: 0.5rem; }
        
        .indicator {
            width: 28px; height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: bold;
            background: rgba(255,255,255,0.1);
            color: rgba(255,255,255,0.3);
            transition: all 0.2s;
        }
        
        .indicator.active {
            background: #00ff88;
            color: #000;
            box-shadow: 0 0 10px rgba(0,255,136,0.5);
        }
        
        /* Main content */
        main {
            flex: 1;
            display: flex;
            padding: 1rem;
            gap: 1rem;
        }
        
        .camera-side, .meme-side {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        .video-container {
            position: relative;
            flex: 1;
            background: #000;
            border-radius: 16px;
            overflow: hidden;
        }
        
        video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transform: scaleX(-1);
        }
        
        .overlay {
            position: absolute;
            top: 0; left: 0; right: 0;
            padding: 1rem;
            background: linear-gradient(180deg, rgba(0,0,0,0.7) 0%, transparent 100%);
        }
        
        .gesture {
            font-size: 2rem;
            font-weight: bold;
            color: #ffa500;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .gesture.confident { color: #00ff88; }
        
        .confidence {
            font-size: 0.9rem;
            color: rgba(255,255,255,0.7);
            margin-top: 0.25rem;
        }
        
        .detected {
            display: inline-block;
            margin-top: 0.5rem;
            padding: 0.25rem 0.75rem;
            background: #00ff88;
            color: #000;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
            animation: pulse 1s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .meme-side {
            align-items: center;
            justify-content: center;
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            border: 2px dashed rgba(255,255,255,0.1);
        }
        
        .meme-display {
            text-align: center;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.9); }
            to { opacity: 1; transform: scale(1); }
        }
        
        .meme-display img {
            max-width: 90%;
            max-height: 60vh;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        }
        
        .meme-label {
            margin-top: 1rem;
            font-size: 1.5rem;
            font-weight: bold;
            color: #F7A5A5;
            text-transform: uppercase;
        }
        
        .no-meme {
            text-align: center;
            color: rgba(255,255,255,0.5);
        }
        
        .no-meme-icon { font-size: 5rem; opacity: 0.3; margin-bottom: 1rem; }
        
        .hints {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 0.75rem;
            margin-top: 1.5rem;
        }
        
        .hints span {
            padding: 0.5rem 1rem;
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            font-size: 0.85rem;
        }
        
        /* Footer */
        footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 2rem;
            background: rgba(0,0,0,0.3);
        }
        
        .status { font-weight: 600; }
        .status.live { color: #00ff88; }
        
        canvas { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎭 Meme Detector</h1>
            <div class="stats">
                <span class="fps" id="fps">FPS: 0</span>
                <div class="indicators">
                    <span class="indicator" id="ind-f">F</span>
                    <span class="indicator" id="ind-l">L</span>
                    <span class="indicator" id="ind-r">R</span>
                </div>
            </div>
        </header>
        
        <main>
            <div class="camera-side">
                <div class="video-container">
                    <video id="video" autoplay playsinline muted></video>
                    <canvas id="canvas"></canvas>
                    <div class="overlay">
                        <div class="gesture" id="gesture">NONE</div>
                        <div class="confidence" id="conf">Confidence: 0%</div>
                        <div class="detected" id="detected" style="display:none">✓ DETECTED</div>
                    </div>
                </div>
            </div>
            
            <div class="meme-side" id="meme-container">
                <div class="no-meme">
                    <div class="no-meme-icon">🎭</div>
                    <p>Make a gesture!</p>
                    <div class="hints">
                        <span>🙏 Cooked</span>
                        <span>👏 DiCaprio</span>
                        <span>🤔 Think</span>
                        <span>✌️ Vanish</span>
                        <span>💨 Speed</span>
                    </div>
                </div>
            </div>
        </main>
        
        <footer>
            <span class="status live" id="status">● LIVE</span>
            <span>Threshold: 85%</span>
        </footer>
    </div>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        const gestureEl = document.getElementById('gesture');
        const confEl = document.getElementById('conf');
        const detectedEl = document.getElementById('detected');
        const fpsEl = document.getElementById('fps');
        const memeContainer = document.getElementById('meme-container');
        
        let lastTime = Date.now();
        let currentMeme = 'none';
        
        // Start camera
        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 1280, height: 720 },
                    audio: false
                });
                video.srcObject = stream;
                await video.play();
                
                // Start detection loop
                setInterval(detectFrame, 100);
            } catch (err) {
                alert('Could not access camera: ' + err.message);
            }
        }
        
        async function detectFrame() {
            if (!video.videoWidth) return;
            
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            // Mirror image
            ctx.scale(-1, 1);
            ctx.drawImage(video, -canvas.width, 0);
            ctx.scale(-1, 1);
            
            const base64 = canvas.toDataURL('image/jpeg', 0.8);
            
            try {
                const response = await fetch('/api/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64 })
                });
                
                const result = await response.json();
                updateUI(result);
                
                // FPS
                const now = Date.now();
                fpsEl.textContent = 'FPS: ' + Math.round(1000 / (now - lastTime));
                lastTime = now;
                
            } catch (err) {
                console.error('Detection error:', err);
            }
        }
        
        function updateUI(result) {
            if (!result) return;
            
            // Update indicators
            const ds = result.detection_status || {};
            document.getElementById('ind-f').classList.toggle('active', ds.face);
            document.getElementById('ind-l').classList.toggle('active', ds.left_hand);
            document.getElementById('ind-r').classList.toggle('active', ds.right_hand);
            
            // Update gesture display
            const isConfident = result.is_confident && result.prediction !== 'none';
            gestureEl.textContent = result.prediction.toUpperCase();
            gestureEl.classList.toggle('confident', isConfident);
            confEl.textContent = 'Confidence: ' + (result.confidence * 100).toFixed(1) + '%';
            detectedEl.style.display = isConfident ? 'inline-block' : 'none';
            
            // Update meme display
            if (isConfident && result.gif) {
                if (currentMeme !== result.prediction) {
                    currentMeme = result.prediction;
                    memeContainer.innerHTML = `
                        <div class="meme-display">
                            <img src="/gifs/${result.gif}" alt="${result.prediction}">
                            <div class="meme-label">${result.prediction}</div>
                        </div>
                    `;
                }
            } else if (!isConfident && currentMeme !== 'none') {
                currentMeme = 'none';
                memeContainer.innerHTML = `
                    <div class="no-meme">
                        <div class="no-meme-icon">🎭</div>
                        <p>Make a gesture!</p>
                        <div class="hints">
                            <span>🙏 Cooked</span>
                            <span>👏 DiCaprio</span>
                            <span>🤔 Think</span>
                            <span>✌️ Vanish</span>
                            <span>💨 Speed</span>
                        </div>
                    </div>
                `;
            }
        }
        
        // Start on load
        startCamera();
    </script>
</body>
</html>
'''

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🎭 MEME DETECTOR WEB APP")
    print("="*50)
    
    load_model()
    
    print("\n🌐 Open in browser: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
