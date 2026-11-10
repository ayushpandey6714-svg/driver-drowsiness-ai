from __future__ import annotations

import cv2
import mediapipe as mp


def initialize_face_mesh():
    """Create a Face Mesh detector for one Streamlit user session."""
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def process_frame(frame, face_mesh):
    """Process a BGR OpenCV image and return it with MediaPipe results."""
    if frame is None or frame.size == 0:
        raise ValueError("The input frame is empty.")
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = face_mesh.process(rgb_frame)
    return frame, results
