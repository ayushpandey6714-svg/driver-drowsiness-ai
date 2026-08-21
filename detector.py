import cv2
import mediapipe as mp

def initialize_face_mesh():
    """MediaPipe Face Mesh initialize karein."""
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return face_mesh

def process_frame(frame, face_mesh):
    """BGR frame ko RGB me convert karke face landmarks extract karein."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    return frame, results