"""
Meme Detector - Streamlit App
==============================
Real-time meme detection using webcam in browser.
"""

import streamlit as st
import cv2
import numpy as np
import os
import sys
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from joblib import load

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from holistic_features import EnhancedHolisticFeatureExtractor

# ============================================================
# CONFIGURATION
# ============================================================

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

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="🎭 Meme Detector",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
    .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
    h1, h2, h3 { color: #F7A5A5 !important; }
    .gesture-box {
        background: rgba(0,0,0,0.3);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    .gesture-name {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00ff88;
    }
    .gesture-none { color: #ffa500; }
    .confidence {
        font-size: 1.2rem;
        color: #aaa;
    }
    .meme-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 400px;
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        border: 2px dashed rgba(255,255,255,0.1);
    }
    .hint-box {
        background: rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 8px 16px;
        margin: 5px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL (cached)
# ============================================================

@st.cache_resource
def load_model():
    """Load model and extractor."""
    model = load(MODEL_PATH)
    extractor = EnhancedHolisticFeatureExtractor()
    return model, extractor

# ============================================================
# VIDEO PROCESSOR
# ============================================================

class MemeVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model, self.extractor = load_model()
        self.result = {"prediction": "none", "confidence": 0.0, "is_confident": False}
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror
        img = cv2.flip(img, 1)
        
        # Convert to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract features
        features, results = self.extractor.extract_enhanced_features_from_frame(rgb)
        
        # Prepare features
        features = features.reshape(1, -1)
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if features.shape[1] < EXPECTED_FEATURES:
            features = np.hstack([features, np.zeros((1, EXPECTED_FEATURES - features.shape[1]))])
        elif features.shape[1] > EXPECTED_FEATURES:
            features = features[:, :EXPECTED_FEATURES]
        
        # Predict
        probs = self.model.predict_proba(features)[0]
        idx = np.argmax(probs)
        prediction = self.model.classes_[idx]
        confidence = float(probs[idx])
        
        # Store result
        self.result = {
            "prediction": prediction,
            "confidence": confidence,
            "is_confident": confidence >= CONFIDENCE_THRESHOLD
        }
        
        # Draw on frame
        color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
        cv2.putText(img, f"{prediction.upper()}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(img, f"Conf: {confidence:.2f}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        if confidence >= CONFIDENCE_THRESHOLD and prediction != "none":
            cv2.putText(img, "DETECTED!", (20, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ============================================================
# MAIN APP
# ============================================================

st.title("🎭 Meme Detector")
st.markdown("Make a gesture and see the meme appear!")

# Create columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📹 Camera")
    
    ctx = webrtc_streamer(
        key="meme-detector",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=MemeVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("🎬 Meme")
    
    meme_placeholder = st.empty()
    
    # Show hints
    st.markdown("""
    <div style="text-align: center; margin-top: 20px;">
        <p style="color: #aaa;">Try these gestures:</p>
        <span class="hint-box">🙏 Cooked</span>
        <span class="hint-box">👏 DiCaprio</span>
        <span class="hint-box">🤔 Think</span>
        <span class="hint-box">✌️ Vanish</span>
        <span class="hint-box">💨 Speed</span>
    </div>
    """, unsafe_allow_html=True)

# Update meme display
if ctx.video_processor:
    result = ctx.video_processor.result
    
    if result.get("is_confident", False) and result.get("prediction", "none") != "none":
        gif_file = GIF_MAPPING.get(result["prediction"])
        if gif_file:
            gif_path = os.path.join(GIF_DIR, gif_file)
            if os.path.exists(gif_path):
                meme_placeholder.image(gif_path, caption=result["prediction"].upper(), use_container_width=True)
            else:
                meme_placeholder.info(f"Detected: {result['prediction'].upper()}")
    else:
        meme_placeholder.markdown("""
        <div class="meme-container">
            <div style="text-align: center; color: #666;">
                <div style="font-size: 4rem; opacity: 0.3;">🎭</div>
                <p>Waiting for gesture...</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"**Confidence Threshold:** {CONFIDENCE_THRESHOLD * 100:.0f}%")
