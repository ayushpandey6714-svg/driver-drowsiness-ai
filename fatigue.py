import numpy as np

# Landmark Indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 0, 17, 13, 14]

# Updated Realistic Thresholds
EYE_AR_THRESH = 0.22      # Slightly strict threshold to prevent fake drowsiness
EYE_AR_CONSEC_FRAMES = 15 # Frame buffer for drowsiness
GAZE_CONSEC_FRAMES = 25   # ~1 second continuous look-away needed to trigger distraction
MOUTH_AR_THRESH = 0.65

def euclidean_dist(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def calculate_ear(eye_landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in eye_landmarks]
    v1 = euclidean_dist(pts[1], pts[5])
    v2 = euclidean_dist(pts[2], pts[4])
    h_dist = euclidean_dist(pts[0], pts[3])
    if h_dist == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h_dist)

def calculate_mar(mouth_landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in mouth_landmarks]
    v = euclidean_dist(pts[4], pts[5])
    h_dist = euclidean_dist(pts[0], pts[1])
    if h_dist == 0:
        return 0.0
    return v / h_dist

def detect_gaze(landmarks, w, h):
    try:
        nose_x = landmarks[1].x
        nose_y = landmarks[1].y
        left_face_x = landmarks[234].x
        right_face_x = landmarks[454].x

        # Horizontal head position ratio
        h_ratio = (nose_x - left_face_x) / (right_face_x - left_face_x + 1e-6)
        
        # Vertical head position ratio
        v_ratio = (nose_y - landmarks[10].y) / (landmarks[152].y - landmarks[10].y + 1e-6)

        # REALISTIC TOLERANCE (Pehle 0.40 & 0.60 tha)
        if h_ratio < 0.28:
            return "Right"
        elif h_ratio > 0.72:
            return "Left"

        if v_ratio < 0.28:
            return "Up"
        elif v_ratio > 0.72:
            return "Down"

        # Fine Iris Check
        r_outer = landmarks[33].x
        r_inner = landmarks[133].x
        r_iris = landmarks[468].x
        iris_ratio = (r_iris - r_outer) / (r_inner - r_outer + 1e-6)

        if iris_ratio < 0.25:
            return "Right"
        elif iris_ratio > 0.75:
            return "Left"

        return "Center"
    except Exception:
        return "Center"

class FatigueDetector:
    def __init__(self):
        self.frame_counter = 0
        self.gaze_counter = 0
        self.no_face_counter = 0
        self.blink_counter = 0
        self.yawn_counter = 0
        self.is_eye_closed = False
        self.is_yawning = False

    def process_landmarks(self, landmarks, frame_shape):
        if landmarks is None:
            self.no_face_counter += 1
            return {
                "ear": 0.0,
                "mar": 0.0,
                "status": "NO FACE DETECTED",
                "blinks": self.blink_counter,
                "yawns": self.yawn_counter,
                "gaze": "N/A",
                "fatigue_index": 1.00 if self.no_face_counter > 10 else 0.50
            }

        self.no_face_counter = 0
        h, w, _ = frame_shape
        
        left_eye_pts = [landmarks[i] for i in LEFT_EYE]
        right_eye_pts = [landmarks[i] for i in RIGHT_EYE]
        mouth_pts = [landmarks[i] for i in MOUTH]

        left_ear = calculate_ear(left_eye_pts, w, h)
        right_ear = calculate_ear(right_eye_pts, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        mar = calculate_mar(mouth_pts, w, h)
        gaze = detect_gaze(landmarks, w, h)

        status = "ATTENTIVE"

        # 1. Eye Drowsiness Check
        if avg_ear < EYE_AR_THRESH:
            self.frame_counter += 1
            self.is_eye_closed = True
            if self.frame_counter >= EYE_AR_CONSEC_FRAMES:
                status = "DROWSY"
        else:
            if self.is_eye_closed and self.frame_counter < EYE_AR_CONSEC_FRAMES:
                self.blink_counter += 1
            self.is_eye_closed = False
            self.frame_counter = 0

        # 2. Distraction Check (Left/Right/Up/Down)
        if status == "ATTENTIVE":
            if gaze in ["Left", "Right", "Up", "Down"]:
                self.gaze_counter += 1
                if self.gaze_counter >= GAZE_CONSEC_FRAMES:
                    status = "DISTRACTED"
            else:
                self.gaze_counter = 0

        # 3. Yawn Logic
        if mar > MOUTH_AR_THRESH:
            if not self.is_yawning:
                self.yawn_counter += 1
                self.is_yawning = True
        else:
            self.is_yawning = False

        fatigue_score = min(1.0, (self.blink_counter * 0.02) + (self.yawn_counter * 0.15) + (0.6 if status == "DROWSY" else 0.0) + (0.35 if status == "DISTRACTED" else 0.0))

        return {
            "ear": round(avg_ear, 3),
            "mar": round(mar, 3),
            "status": status,
            "blinks": self.blink_counter,
            "yawns": self.yawn_counter,
            "gaze": gaze,
            "fatigue_index": round(fatigue_score, 2)
        }