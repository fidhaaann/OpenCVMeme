"""
Holistic Feature Extraction Module
===================================
Extracts stable, movement-invariant features using relative landmark mapping
from MediaPipe Holistic (Face + Both Hands simultaneously).

Features:
- Face landmarks relative to nose bridge anchor
- Hand landmarks normalized to wrist/palm center
- Single fused feature vector per frame
"""

import numpy as np
import mediapipe as mp

# ============================================================
# LANDMARK INDICES FOR OPTIMIZED FACE EXTRACTION
# ============================================================

# Nose landmarks (anchor points)
NOSE_BRIDGE = [6, 197, 195, 5]  # Bridge of nose
NOSE_TIP = [1, 2, 98, 327]  # Tip of nose

# Eye landmarks (upper and lower contours)
LEFT_EYE_UPPER = [159, 145, 153, 144, 163, 7]
LEFT_EYE_LOWER = [33, 133, 160, 161, 246, 173]
RIGHT_EYE_UPPER = [386, 374, 380, 373, 390, 249]
RIGHT_EYE_LOWER = [263, 362, 387, 388, 466, 398]

# Eyebrow landmarks
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295]

# Lip landmarks (inner + outer boundaries)
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

# Cheek peak curvature points
LEFT_CHEEK = [123, 147, 187, 205, 36, 142]
RIGHT_CHEEK = [352, 376, 411, 425, 266, 371]

# Jaw contour for expression context
JAW_CONTOUR = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288]

# Combine all face indices
ALL_FACE_INDICES = (
    NOSE_BRIDGE + NOSE_TIP +
    LEFT_EYE_UPPER + LEFT_EYE_LOWER +
    RIGHT_EYE_UPPER + RIGHT_EYE_LOWER +
    LEFT_EYEBROW + RIGHT_EYEBROW +
    LIPS_OUTER + LIPS_INNER +
    LEFT_CHEEK + RIGHT_CHEEK +
    JAW_CONTOUR
)

# Remove duplicates while preserving order
FACE_INDICES = list(dict.fromkeys(ALL_FACE_INDICES))

# Number of hand landmarks
NUM_HAND_LANDMARKS = 21


class HolisticFeatureExtractor:
    """
    Extracts fused features from face and both hands using MediaPipe Holistic.
    All coordinates are normalized relative to anchor points for movement invariance.
    """

    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,  # Highest accuracy
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            refine_face_landmarks=True,  # 478 face landmarks
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # Feature dimensions
        self.face_feature_dim = len(FACE_INDICES) * 3  # x, y, z per landmark
        self.hand_feature_dim = NUM_HAND_LANDMARKS * 3  # x, y, z per landmark
        self.total_feature_dim = self.face_feature_dim + (self.hand_feature_dim * 2)  # face + 2 hands
        
    def process_frame(self, rgb_frame):
        """
        Process a single RGB frame and return holistic results.
        
        Args:
            rgb_frame: RGB image (numpy array)
            
        Returns:
            MediaPipe Holistic results object
        """
        return self.holistic.process(rgb_frame)
    
    def get_nose_anchor(self, face_landmarks):
        """
        Calculate the nose bridge anchor point (average of nose bridge landmarks).
        
        Args:
            face_landmarks: MediaPipe face landmarks
            
        Returns:
            numpy array [x, y, z] of anchor position
        """
        if face_landmarks is None:
            return np.array([0.5, 0.5, 0.0])  # Default center
        
        anchor_indices = NOSE_BRIDGE[:2]  # Use first 2 nose bridge points
        anchor_coords = []
        
        for idx in anchor_indices:
            lm = face_landmarks.landmark[idx]
            anchor_coords.append([lm.x, lm.y, lm.z])
        
        return np.mean(anchor_coords, axis=0)
    
    def extract_face_features(self, face_landmarks, anchor=None):
        """
        Extract face features relative to nose bridge anchor.
        
        Args:
            face_landmarks: MediaPipe face landmarks (478 points with refinement)
            anchor: Optional pre-computed anchor point
            
        Returns:
            numpy array of normalized face features
        """
        if face_landmarks is None:
            return np.zeros(self.face_feature_dim)
        
        if anchor is None:
            anchor = self.get_nose_anchor(face_landmarks)
        
        features = []
        
        for idx in FACE_INDICES:
            if idx < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[idx]
                # Compute delta relative to anchor
                delta_x = lm.x - anchor[0]
                delta_y = lm.y - anchor[1]
                delta_z = lm.z - anchor[2]
                features.extend([delta_x, delta_y, delta_z])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def get_hand_anchor(self, hand_landmarks):
        """
        Calculate hand anchor point (palm center from wrist and middle finger base).
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            numpy array [x, y, z] of palm center
        """
        if hand_landmarks is None:
            return np.array([0.5, 0.5, 0.0])
        
        # Palm center: average of wrist (0), index base (5), pinky base (17), middle base (9)
        palm_indices = [0, 5, 9, 13, 17]
        palm_coords = []
        
        for idx in palm_indices:
            lm = hand_landmarks.landmark[idx]
            palm_coords.append([lm.x, lm.y, lm.z])
        
        return np.mean(palm_coords, axis=0)
    
    def extract_hand_features(self, hand_landmarks, anchor=None):
        """
        Extract hand features normalized to palm center.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks (21 points)
            anchor: Optional pre-computed anchor point
            
        Returns:
            numpy array of normalized hand features
        """
        if hand_landmarks is None:
            return np.zeros(self.hand_feature_dim)
        
        if anchor is None:
            anchor = self.get_hand_anchor(hand_landmarks)
        
        features = []
        
        # Calculate scale factor based on hand size (wrist to middle finger tip)
        wrist = hand_landmarks.landmark[0]
        middle_tip = hand_landmarks.landmark[12]
        hand_scale = np.sqrt(
            (middle_tip.x - wrist.x)**2 + 
            (middle_tip.y - wrist.y)**2 + 
            (middle_tip.z - wrist.z)**2
        )
        
        # Avoid division by zero
        if hand_scale < 0.001:
            hand_scale = 0.2
        
        for i in range(NUM_HAND_LANDMARKS):
            lm = hand_landmarks.landmark[i]
            # Normalize relative to palm center and scale
            delta_x = (lm.x - anchor[0]) / hand_scale
            delta_y = (lm.y - anchor[1]) / hand_scale
            delta_z = (lm.z - anchor[2]) / hand_scale
            features.extend([delta_x, delta_y, delta_z])
        
        return np.array(features)
    
    def extract_features(self, results):
        """
        Extract complete fused feature vector from holistic results.
        
        Feature order: [left_hand (63), right_hand (63), face (N*3)]
        
        Args:
            results: MediaPipe Holistic results
            
        Returns:
            numpy array of shape (total_feature_dim,)
        """
        # Extract left hand features
        left_hand_features = self.extract_hand_features(results.left_hand_landmarks)
        
        # Extract right hand features
        right_hand_features = self.extract_hand_features(results.right_hand_landmarks)
        
        # Extract face features
        face_features = self.extract_face_features(results.face_landmarks)
        
        # Concatenate all features
        fused_features = np.concatenate([
            left_hand_features,
            right_hand_features,
            face_features
        ])
        
        return fused_features
    
    def extract_features_from_frame(self, rgb_frame):
        """
        Convenience method to extract features directly from an RGB frame.
        
        Args:
            rgb_frame: RGB image (numpy array)
            
        Returns:
            tuple: (features, results) where features is numpy array and results is MediaPipe output
        """
        results = self.process_frame(rgb_frame)
        features = self.extract_features(results)
        return features, results
    
    def has_valid_detection(self, results):
        """
        Check if the results contain valid face or hand detections.
        
        Args:
            results: MediaPipe Holistic results
            
        Returns:
            bool: True if at least face or one hand is detected
        """
        has_face = results.face_landmarks is not None
        has_left_hand = results.left_hand_landmarks is not None
        has_right_hand = results.right_hand_landmarks is not None
        
        return has_face or has_left_hand or has_right_hand
    
    def get_detection_status(self, results):
        """
        Get detailed detection status.
        
        Args:
            results: MediaPipe Holistic results
            
        Returns:
            dict with detection status for each component
        """
        return {
            'face': results.face_landmarks is not None,
            'left_hand': results.left_hand_landmarks is not None,
            'right_hand': results.right_hand_landmarks is not None,
            'pose': results.pose_landmarks is not None
        }
    
    def release(self):
        """Release MediaPipe resources."""
        self.holistic.close()


# ============================================================
# ADDITIONAL FEATURE ENGINEERING FUNCTIONS
# ============================================================

def compute_hand_gesture_features(hand_landmarks):
    """
    Compute additional discriminative features for hand gestures.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        
    Returns:
        numpy array of additional features
    """
    if hand_landmarks is None:
        return np.zeros(15)
    
    lm = hand_landmarks.landmark
    features = []
    
    # Finger tip to palm distances (5 features)
    palm_center = np.array([
        (lm[0].x + lm[5].x + lm[17].x) / 3,
        (lm[0].y + lm[5].y + lm[17].y) / 3,
        (lm[0].z + lm[5].z + lm[17].z) / 3
    ])
    
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
    for tip_idx in finger_tips:
        tip = np.array([lm[tip_idx].x, lm[tip_idx].y, lm[tip_idx].z])
        dist = np.linalg.norm(tip - palm_center)
        features.append(dist)
    
    # Finger curl features (5 features) - tip to pip distance
    finger_pips = [3, 6, 10, 14, 18]  # PIP joints
    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        tip = np.array([lm[tip_idx].x, lm[tip_idx].y, lm[tip_idx].z])
        pip = np.array([lm[pip_idx].x, lm[pip_idx].y, lm[pip_idx].z])
        curl = np.linalg.norm(tip - pip)
        features.append(curl)
    
    # Inter-finger distances (5 features)
    for i in range(len(finger_tips) - 1):
        tip1 = np.array([lm[finger_tips[i]].x, lm[finger_tips[i]].y, lm[finger_tips[i]].z])
        tip2 = np.array([lm[finger_tips[i+1]].x, lm[finger_tips[i+1]].y, lm[finger_tips[i+1]].z])
        dist = np.linalg.norm(tip1 - tip2)
        features.append(dist)
    
    return np.array(features)


def compute_face_expression_features(face_landmarks):
    """
    Compute additional discriminative features for facial expressions.
    
    Args:
        face_landmarks: MediaPipe face landmarks
        
    Returns:
        numpy array of additional features
    """
    if face_landmarks is None:
        return np.zeros(10)
    
    lm = face_landmarks.landmark
    features = []
    
    # Mouth openness (vertical)
    upper_lip = np.array([lm[13].x, lm[13].y, lm[13].z])
    lower_lip = np.array([lm[14].x, lm[14].y, lm[14].z])
    mouth_open = np.linalg.norm(upper_lip - lower_lip)
    features.append(mouth_open)
    
    # Mouth width
    left_corner = np.array([lm[61].x, lm[61].y, lm[61].z])
    right_corner = np.array([lm[291].x, lm[291].y, lm[291].z])
    mouth_width = np.linalg.norm(left_corner - right_corner)
    features.append(mouth_width)
    
    # Smile ratio (mouth width / mouth height)
    smile_ratio = mouth_width / (mouth_open + 0.001)
    features.append(smile_ratio)
    
    # Eye openness (left and right)
    left_eye_top = np.array([lm[159].x, lm[159].y, lm[159].z])
    left_eye_bottom = np.array([lm[145].x, lm[145].y, lm[145].z])
    left_eye_open = np.linalg.norm(left_eye_top - left_eye_bottom)
    features.append(left_eye_open)
    
    right_eye_top = np.array([lm[386].x, lm[386].y, lm[386].z])
    right_eye_bottom = np.array([lm[374].x, lm[374].y, lm[374].z])
    right_eye_open = np.linalg.norm(right_eye_top - right_eye_bottom)
    features.append(right_eye_open)
    
    # Eyebrow raise (left and right)
    left_brow = np.array([lm[105].x, lm[105].y, lm[105].z])
    left_eye_center = np.array([lm[159].x, lm[159].y, lm[159].z])
    left_brow_raise = np.linalg.norm(left_brow - left_eye_center)
    features.append(left_brow_raise)
    
    right_brow = np.array([lm[334].x, lm[334].y, lm[334].z])
    right_eye_center = np.array([lm[386].x, lm[386].y, lm[386].z])
    right_brow_raise = np.linalg.norm(right_brow - right_eye_center)
    features.append(right_brow_raise)
    
    # Nose wrinkle (distance between nose bridge points)
    nose_top = np.array([lm[6].x, lm[6].y, lm[6].z])
    nose_mid = np.array([lm[197].x, lm[197].y, lm[197].z])
    nose_wrinkle = np.linalg.norm(nose_top - nose_mid)
    features.append(nose_wrinkle)
    
    # Cheek raise (left and right)
    left_cheek = np.array([lm[123].x, lm[123].y, lm[123].z])
    features.append(left_cheek[1])  # Y position
    
    right_cheek = np.array([lm[352].x, lm[352].y, lm[352].z])
    features.append(right_cheek[1])  # Y position
    
    return np.array(features)


def compute_two_hand_relation_features(left_hand, right_hand):
    """
    Compute features describing the relationship between two hands.
    
    Args:
        left_hand: MediaPipe left hand landmarks
        right_hand: MediaPipe right hand landmarks
        
    Returns:
        numpy array of relation features
    """
    if left_hand is None or right_hand is None:
        return np.zeros(12)
    
    left_lm = left_hand.landmark
    right_lm = right_hand.landmark
    features = []
    
    # Distance between palms (wrists)
    left_wrist = np.array([left_lm[0].x, left_lm[0].y, left_lm[0].z])
    right_wrist = np.array([right_lm[0].x, right_lm[0].y, right_lm[0].z])
    palm_dist = np.linalg.norm(left_wrist - right_wrist)
    features.append(palm_dist)
    
    # Distance between index fingertips
    left_index = np.array([left_lm[8].x, left_lm[8].y, left_lm[8].z])
    right_index = np.array([right_lm[8].x, right_lm[8].y, right_lm[8].z])
    index_dist = np.linalg.norm(left_index - right_index)
    features.append(index_dist)
    
    # Distance between thumb tips
    left_thumb = np.array([left_lm[4].x, left_lm[4].y, left_lm[4].z])
    right_thumb = np.array([right_lm[4].x, right_lm[4].y, right_lm[4].z])
    thumb_dist = np.linalg.norm(left_thumb - right_thumb)
    features.append(thumb_dist)
    
    # Relative hand positions (x, y, z deltas)
    palm_delta = left_wrist - right_wrist
    features.extend(palm_delta.tolist())
    
    # Average finger distance
    finger_tips = [4, 8, 12, 16, 20]
    avg_finger_dist = 0
    for tip_idx in finger_tips:
        left_tip = np.array([left_lm[tip_idx].x, left_lm[tip_idx].y, left_lm[tip_idx].z])
        right_tip = np.array([right_lm[tip_idx].x, right_lm[tip_idx].y, right_lm[tip_idx].z])
        avg_finger_dist += np.linalg.norm(left_tip - right_tip)
    avg_finger_dist /= len(finger_tips)
    features.append(avg_finger_dist)
    
    # Hand overlap indicator (are hands close together)
    features.append(float(palm_dist < 0.15))
    
    # Hands crossed indicator
    features.append(float(left_wrist[0] > right_wrist[0]))
    
    # Vertical alignment (are hands at same height)
    features.append(abs(left_wrist[1] - right_wrist[1]))
    
    # Horizontal alignment (are hands at same x position)
    features.append(abs(left_wrist[0] - right_wrist[0]))
    
    return np.array(features)


class EnhancedHolisticFeatureExtractor(HolisticFeatureExtractor):
    """
    Enhanced feature extractor with additional discriminative features.
    """
    
    def __init__(self):
        super().__init__()
        # Additional feature dimensions
        self.hand_gesture_dim = 15 * 2  # 15 features per hand
        self.face_expression_dim = 10
        self.two_hand_relation_dim = 12
        self.enhanced_feature_dim = (
            self.total_feature_dim + 
            self.hand_gesture_dim + 
            self.face_expression_dim + 
            self.two_hand_relation_dim
        )
    
    def extract_enhanced_features(self, results):
        """
        Extract complete enhanced feature vector.
        
        Args:
            results: MediaPipe Holistic results
            
        Returns:
            numpy array of enhanced features
        """
        # Base features
        base_features = self.extract_features(results)
        
        # Additional hand gesture features
        left_hand_gesture = compute_hand_gesture_features(results.left_hand_landmarks)
        right_hand_gesture = compute_hand_gesture_features(results.right_hand_landmarks)
        
        # Additional face expression features
        face_expression = compute_face_expression_features(results.face_landmarks)
        
        # Two hand relation features
        two_hand_relation = compute_two_hand_relation_features(
            results.left_hand_landmarks,
            results.right_hand_landmarks
        )
        
        # Concatenate all features
        enhanced_features = np.concatenate([
            base_features,
            left_hand_gesture,
            right_hand_gesture,
            face_expression,
            two_hand_relation
        ])
        
        return enhanced_features
    
    def extract_enhanced_features_from_frame(self, rgb_frame):
        """
        Convenience method to extract enhanced features from an RGB frame.
        
        Args:
            rgb_frame: RGB image (numpy array)
            
        Returns:
            tuple: (features, results)
        """
        results = self.process_frame(rgb_frame)
        features = self.extract_enhanced_features(results)
        return features, results


# ============================================================
# STANDALONE FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(results):
    """
    Standalone function for feature extraction (for backward compatibility).
    
    Args:
        results: MediaPipe Holistic results or object with face_landmarks,
                 left_hand_landmarks, right_hand_landmarks attributes
        
    Returns:
        numpy array of features
    """
    extractor = HolisticFeatureExtractor.__new__(HolisticFeatureExtractor)
    extractor.face_feature_dim = len(FACE_INDICES) * 3
    extractor.hand_feature_dim = NUM_HAND_LANDMARKS * 3
    extractor.total_feature_dim = extractor.face_feature_dim + (extractor.hand_feature_dim * 2)
    
    return extractor.extract_features(results)


if __name__ == "__main__":
    # Test feature extraction
    import cv2
    
    print("Testing Holistic Feature Extraction...")
    print(f"Face indices count: {len(FACE_INDICES)}")
    
    extractor = EnhancedHolisticFeatureExtractor()
    print(f"Base feature dimension: {extractor.total_feature_dim}")
    print(f"Enhanced feature dimension: {extractor.enhanced_feature_dim}")
    
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            features, results = extractor.extract_enhanced_features_from_frame(rgb)
            print(f"Extracted features shape: {features.shape}")
            print(f"Detection status: {extractor.get_detection_status(results)}")
        cap.release()
    
    extractor.release()
    print("Test complete!")
