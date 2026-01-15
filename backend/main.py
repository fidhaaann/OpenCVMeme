"""
FastAPI Backend for Meme Detector
=================================
WebSocket-based real-time meme detection API
"""

import os
import sys
import cv2
import numpy as np
import base64
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from joblib import load
from typing import Optional, Dict, List
from collections import defaultdict
import uvicorn

# Add parent src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from holistic_features import EnhancedHolisticFeatureExtractor

# Configuration
MODEL_PATH = os.path.join(BASE_DIR, "models", "meme_classifier.joblib")
GIF_DIR = os.path.join(BASE_DIR, "gifs")
CONFIDENCE_THRESHOLD = 0.85
EXPECTED_FEATURES = 514

# Smoothing settings (from original meme_detector.py)
SMOOTHING_FRAMES = 3  # Number of frames to smooth predictions

GIF_MAPPING = {
    "cooked": "cooked.jpg",
    "dicaprio": "dicaprio.gif",
    "speed": "speed.gif",
    "think": "think.jpg",
    "vanish": "vanish.gif",
}


class PredictionSmoother:
    """
    Applies temporal smoothing to predictions using majority voting.
    This stabilizes predictions across frames for more accurate results.
    """
    
    def __init__(self, window_size: int = SMOOTHING_FRAMES):
        self.window_size = window_size
        self.prediction_history: List[str] = []
        self.confidence_history: List[float] = []
    
    def smooth(self, prediction: str, confidence: float) -> tuple:
        """
        Apply temporal smoothing to the prediction.
        
        Args:
            prediction: Current frame prediction
            confidence: Current frame confidence
            
        Returns:
            Smoothed (prediction, confidence) tuple
        """
        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)
        
        # Keep only recent history
        if len(self.prediction_history) > self.window_size:
            self.prediction_history.pop(0)
            self.confidence_history.pop(0)
        
        # Need at least 3 frames for smoothing
        if len(self.prediction_history) < 3:
            return prediction, confidence
        
        # Count weighted votes (confidence-weighted majority voting)
        vote_counts: Dict[str, float] = {}
        for pred, conf in zip(self.prediction_history, self.confidence_history):
            if pred not in vote_counts:
                vote_counts[pred] = 0
            vote_counts[pred] += conf
        
        # Get winner
        best_pred = max(vote_counts.keys(), key=lambda k: vote_counts[k])
        
        # Average confidence for winning class
        matching_confs = [c for p, c in zip(self.prediction_history, self.confidence_history) 
                         if p == best_pred]
        avg_conf = float(np.mean(matching_confs))
        
        return best_pred, avg_conf
    
    def reset(self):
        """Clear the smoothing history."""
        self.prediction_history.clear()
        self.confidence_history.clear()


# FastAPI App
app = FastAPI(
    title="Meme Detector API",
    description="Real-time gesture-based meme detection",
    version="1.0.0"
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and extractor
model = None
extractor = None


def load_model():
    """Load the ML model and feature extractor."""
    global model, extractor
    if model is None:
        print("🔧 Loading model...")
        model = load(MODEL_PATH)
        extractor = EnhancedHolisticFeatureExtractor()
        print(f"✅ Model loaded! Classes: {list(model.classes_)}")


def predict_frame(img: np.ndarray) -> dict:
    """
    Run prediction on a single frame.
    
    Args:
        img: BGR image from OpenCV
        
    Returns:
        Detection result dictionary
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    features, results = extractor.extract_enhanced_features_from_frame(rgb)
    detection_status = extractor.get_detection_status(results)
    
    # Prepare features
    features = features.reshape(1, -1)
    features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Pad or truncate features
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
        'gif': GIF_MAPPING.get(prediction) if confidence >= CONFIDENCE_THRESHOLD and prediction != 'none' else None
    }


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Meme Detector API is running"}


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": list(model.classes_) if model else [],
        "threshold": CONFIDENCE_THRESHOLD
    }


@app.post("/detect")
async def detect(data: dict):
    """
    Single frame detection endpoint.
    
    Expects: {"image": "base64_encoded_image"}
    """
    try:
        img_b64 = data.get('image', '')
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]
        
        img_bytes = base64.b64decode(img_b64)
        img_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        result = predict_frame(img)
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming detection.
    
    Client sends base64 frames, server responds with predictions.
    Uses prediction smoothing for stable, accurate results.
    """
    await websocket.accept()
    print("🔌 WebSocket connected")
    
    # Create a smoother for this client session
    smoother = PredictionSmoother(window_size=SMOOTHING_FRAMES)
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_text()
            
            try:
                payload = json.loads(data)
                img_b64 = payload.get('image', '')
                
                if ',' in img_b64:
                    img_b64 = img_b64.split(',')[1]
                
                img_bytes = base64.b64decode(img_b64)
                img_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Get raw prediction
                    raw_result = predict_frame(img)
                    
                    # Apply temporal smoothing
                    smoothed_pred, smoothed_conf = smoother.smooth(
                        raw_result['prediction'], 
                        raw_result['confidence']
                    )
                    
                    # Build smoothed result
                    is_confident = smoothed_conf >= CONFIDENCE_THRESHOLD
                    result = {
                        'prediction': smoothed_pred,
                        'confidence': smoothed_conf,
                        'is_confident': is_confident,
                        'detection_status': raw_result['detection_status'],
                        'gif': GIF_MAPPING.get(smoothed_pred) if is_confident and smoothed_pred != 'none' else None,
                        'raw_prediction': raw_result['prediction'],
                        'raw_confidence': raw_result['confidence']
                    }
                    
                    await websocket.send_json(result)
                else:
                    await websocket.send_json({"error": "Invalid frame"})
                    
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")


@app.get("/gif/{filename}")
async def get_gif(filename: str):
    """Serve GIF/image files."""
    path = os.path.join(GIF_DIR, filename)
    if os.path.exists(path):
        media_type = "image/gif" if filename.endswith('.gif') else "image/jpeg"
        return FileResponse(path, media_type=media_type)
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/gifs")
async def list_gifs():
    """List available GIF mappings."""
    return {
        "mapping": GIF_MAPPING,
        "available": os.listdir(GIF_DIR) if os.path.exists(GIF_DIR) else []
    }


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🎭 MEME DETECTOR API")
    print("=" * 50)
    print("📡 Starting FastAPI server...")
    print("🌐 API: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("=" * 50 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
