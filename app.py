"""
Meme Detector Web App
"""
import os
import sys
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from joblib import load

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from holistic_features import EnhancedHolisticFeatureExtractor

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "meme_classifier.joblib")
GIF_DIR = os.path.join(BASE_DIR, "gifs")
CONFIDENCE_THRESHOLD = 0.85
EXPECTED_FEATURES = 514

GIF_MAPPING = {
    "cooked": "cooked.jpg",
    "dicaprio": "dicaprio.gif", 
    "speed": "speed.gif",
    "think": "think.jpg",
    "vanish": "vanish.gif",
}

# Flask app
app = Flask(__name__)
CORS(app)

# Global
model = None
extractor = None

def load_model():
    global model, extractor
    if model is None:
        print("Loading model...")
        model = load(MODEL_PATH)
        extractor = EnhancedHolisticFeatureExtractor()
        print("Model loaded!")

def predict_image(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    features, results = extractor.extract_enhanced_features_from_frame(rgb)
    detection_status = extractor.get_detection_status(results)
    
    features = features.reshape(1, -1)
    features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
    
    if features.shape[1] < EXPECTED_FEATURES:
        features = np.hstack([features, np.zeros((1, EXPECTED_FEATURES - features.shape[1]))])
    elif features.shape[1] > EXPECTED_FEATURES:
        features = features[:, :EXPECTED_FEATURES]
    
    probs = model.predict_proba(features)[0]
    idx = np.argmax(probs)
    prediction = model.classes_[idx]
    confidence = float(probs[idx])
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'is_confident': confidence >= CONFIDENCE_THRESHOLD,
        'detection_status': detection_status,
        'gif': GIF_MAPPING.get(prediction) if confidence >= CONFIDENCE_THRESHOLD and prediction != 'none' else None
    }

# HTML Page
HTML = '''<!DOCTYPE html>
<html>
<head>
    <title>Meme Detector</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: Arial, sans-serif; 
            background: #1a1a2e; 
            color: white;
            min-height: 100vh;
        }
        .container { 
            display: flex; 
            flex-direction: column;
            height: 100vh;
        }
        header {
            padding: 15px 20px;
            background: rgba(0,0,0,0.3);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        header h1 { color: #F7A5A5; font-size: 1.4rem; }
        .stats { display: flex; gap: 15px; align-items: center; }
        .fps { color: #0f0; font-family: monospace; }
        .indicators span {
            display: inline-block;
            width: 24px; height: 24px;
            border-radius: 50%;
            text-align: center;
            line-height: 24px;
            font-size: 12px;
            background: #333;
            margin-left: 5px;
        }
        .indicators span.on { background: #0f0; color: #000; }
        main {
            flex: 1;
            display: flex;
            padding: 15px;
            gap: 15px;
        }
        .left, .right {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        .video-box {
            position: relative;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            flex: 1;
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
            padding: 15px;
            background: linear-gradient(to bottom, rgba(0,0,0,0.7), transparent);
        }
        .gesture { font-size: 28px; font-weight: bold; color: orange; }
        .gesture.detected { color: #0f0; }
        .conf { color: #aaa; font-size: 14px; margin-top: 5px; }
        .right {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            align-items: center;
            justify-content: center;
        }
        .meme-img { max-width: 90%; max-height: 70vh; border-radius: 10px; }
        .meme-name { margin-top: 15px; font-size: 24px; color: #F7A5A5; text-transform: uppercase; }
        .placeholder { text-align: center; color: #666; }
        .placeholder .icon { font-size: 80px; opacity: 0.3; }
        .hints { margin-top: 20px; display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; }
        .hints span { background: rgba(255,255,255,0.1); padding: 8px 15px; border-radius: 20px; font-size: 14px; }
        footer {
            padding: 10px 20px;
            background: rgba(0,0,0,0.3);
            display: flex;
            justify-content: space-between;
        }
        .live { color: #0f0; }
        canvas { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎭 Meme Detector</h1>
            <div class="stats">
                <span class="fps">FPS: <span id="fps">0</span></span>
                <div class="indicators">
                    <span id="f">F</span>
                    <span id="l">L</span>
                    <span id="r">R</span>
                </div>
            </div>
        </header>
        <main>
            <div class="left">
                <div class="video-box">
                    <video id="video" autoplay playsinline muted></video>
                    <canvas id="canvas"></canvas>
                    <div class="overlay">
                        <div class="gesture" id="gesture">NONE</div>
                        <div class="conf">Confidence: <span id="conf">0</span>%</div>
                    </div>
                </div>
            </div>
            <div class="right" id="meme-area">
                <div class="placeholder">
                    <div class="icon">🎭</div>
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
            <span class="live">● LIVE</span>
            <span>Threshold: 85%</span>
        </footer>
    </div>
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let currentMeme = null;
        let lastTime = Date.now();

        // Start camera
        navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false })
            .then(stream => {
                video.srcObject = stream;
                video.play();
                setInterval(capture, 150);
            })
            .catch(err => alert('Camera error: ' + err));

        async function capture() {
            if (!video.videoWidth) return;
            
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.scale(-1, 1);
            ctx.drawImage(video, -canvas.width, 0);
            ctx.scale(-1, 1);
            
            const base64 = canvas.toDataURL('image/jpeg', 0.7);
            
            try {
                const res = await fetch('/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64 })
                });
                const data = await res.json();
                update(data);
            } catch (e) {
                console.error(e);
            }
            
            // FPS
            const now = Date.now();
            document.getElementById('fps').textContent = Math.round(1000 / (now - lastTime));
            lastTime = now;
        }

        function update(data) {
            // Indicators
            const ds = data.detection_status || {};
            document.getElementById('f').className = ds.face ? 'on' : '';
            document.getElementById('l').className = ds.left_hand ? 'on' : '';
            document.getElementById('r').className = ds.right_hand ? 'on' : '';
            
            // Gesture
            const g = document.getElementById('gesture');
            g.textContent = data.prediction.toUpperCase();
            g.className = 'gesture' + (data.is_confident ? ' detected' : '');
            document.getElementById('conf').textContent = (data.confidence * 100).toFixed(1);
            
            // Meme
            const area = document.getElementById('meme-area');
            if (data.is_confident && data.gif && data.prediction !== 'none') {
                if (currentMeme !== data.prediction) {
                    currentMeme = data.prediction;
                    area.innerHTML = '<img class="meme-img" src="/gif/' + data.gif + '"><div class="meme-name">' + data.prediction + '</div>';
                }
            } else if (currentMeme !== null && !data.is_confident) {
                currentMeme = null;
                area.innerHTML = '<div class="placeholder"><div class="icon">🎭</div><p>Make a gesture!</p><div class="hints"><span>🙏 Cooked</span><span>👏 DiCaprio</span><span>🤔 Think</span><span>✌️ Vanish</span><span>💨 Speed</span></div></div>';
            }
        }
    </script>
</body>
</html>'''

@app.route('/')
def home():
    return Response(HTML, mimetype='text/html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        img_b64 = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_bytes = base64.b64decode(img_b64)
        img_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        
        result = predict_image(img)
        print(f"{'✓' if result['is_confident'] else ' '} {result['prediction']:10} {result['confidence']:.2f}")
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/gif/<filename>')
def gif(filename):
    path = os.path.join(GIF_DIR, filename)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = f.read()
        mime = 'image/gif' if filename.endswith('.gif') else 'image/jpeg'
        return Response(data, mimetype=mime)
    return '', 404

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🎭 MEME DETECTOR")
    print("="*50)
    load_model()
    print("\n🌐 Open: http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, threaded=True)
