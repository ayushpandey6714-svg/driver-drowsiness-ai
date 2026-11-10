import numpy as np

# MediaPipe Face Mesh landmark indices.
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 0, 17, 13, 14]

EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 15
GAZE_CONSEC_FRAMES = 25
MOUTH_AR_THRESH = 0.65


def euclidean_dist(pt1, pt2):
    return float(np.linalg.norm(np.asarray(pt1, dtype=float) - np.asarray(pt2, dtype=float)))


def calculate_ear(eye_landmarks, w, h):
    pts = [(lm.x * w, lm.y * h) for lm in eye_landmarks]
    vertical1 = euclidean_dist(pts[1], pts[5])
    vertical2 = euclidean_dist(pts[2], pts[4])
    horizontal = euclidean_dist(pts[0], pts[3])
    return (vertical1 + vertical2) / (2.0 * horizontal) if horizontal else 0.0


def calculate_mar(mouth_landmarks, w, h):
    pts = [(lm.x * w, lm.y * h) for lm in mouth_landmarks]
    vertical = euclidean_dist(pts[4], pts[5])
    horizontal = euclidean_dist(pts[0], pts[1])
    return vertical / horizontal if horizontal else 0.0


def detect_gaze(landmarks, w, h):
    """Approximate gaze from face position and iris position when available."""
    try:
        left_face_x = landmarks[234].x
        right_face_x = landmarks[454].x
        face_width = right_face_x - left_face_x
        face_height = landmarks[152].y - landmarks[10].y
        if abs(face_width) < 1e-6 or abs(face_height) < 1e-6:
            return "Center"

        h_ratio = (landmarks[1].x - left_face_x) / face_width
        v_ratio = (landmarks[1].y - landmarks[10].y) / face_height
        if h_ratio < 0.28:
            return "Right"
        if h_ratio > 0.72:
            return "Left"
        if v_ratio < 0.28:
            return "Up"
        if v_ratio > 0.72:
            return "Down"

        # Refined Face Mesh adds iris landmarks; guard for older configurations.
        if len(landmarks) > 468:
            outer, inner, iris = landmarks[33].x, landmarks[133].x, landmarks[468].x
            denominator = inner - outer
            if abs(denominator) > 1e-6:
                iris_ratio = (iris - outer) / denominator
                if iris_ratio < 0.25:
                    return "Right"
                if iris_ratio > 0.75:
                    return "Left"
        return "Center"
    except (IndexError, AttributeError, ZeroDivisionError):
        return "Center"


class FatigueDetector:
    """Stateful frame-based estimates for a single image/video analysis."""

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
                "fatigue_index": 1.0 if self.no_face_counter > 10 else 0.5,
            }

        self.no_face_counter = 0
        h, w = frame_shape[:2]
        if len(landmarks) <= max(max(LEFT_EYE), max(RIGHT_EYE), max(MOUTH)):
            raise ValueError("Face detector returned too few landmarks.")

        left_ear = calculate_ear([landmarks[i] for i in LEFT_EYE], w, h)
        right_ear = calculate_ear([landmarks[i] for i in RIGHT_EYE], w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        mar = calculate_mar([landmarks[i] for i in MOUTH], w, h)
        gaze = detect_gaze(landmarks, w, h)

        status = "ATTENTIVE"
        if avg_ear < EYE_AR_THRESH:
            self.frame_counter += 1
            self.is_eye_closed = True
            if self.frame_counter >= EYE_AR_CONSEC_FRAMES:
                status = "DROWSY"
        else:
            if self.is_eye_closed and 0 < self.frame_counter < EYE_AR_CONSEC_FRAMES:
                self.blink_counter += 1
            self.is_eye_closed = False
            self.frame_counter = 0

        if status == "ATTENTIVE":
            if gaze in ("Left", "Right", "Up", "Down"):
                self.gaze_counter += 1
                if self.gaze_counter >= GAZE_CONSEC_FRAMES:
                    status = "DISTRACTED"
            else:
                self.gaze_counter = 0

        if mar > MOUTH_AR_THRESH:
            if not self.is_yawning:
                self.yawn_counter += 1
                self.is_yawning = True
        else:
            self.is_yawning = False

        fatigue_score = min(
            1.0,
            self.blink_counter * 0.02
            + self.yawn_counter * 0.15
            + (0.6 if status == "DROWSY" else 0.0)
            + (0.35 if status == "DISTRACTED" else 0.0),
        )
        return {
            "ear": round(avg_ear, 3),
            "mar": round(mar, 3),
            "status": status,
            "blinks": self.blink_counter,
            "yawns": self.yawn_counter,
            "gaze": gaze,
            "fatigue_index": round(fatigue_score, 2),
        }
