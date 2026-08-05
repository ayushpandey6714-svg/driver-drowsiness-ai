import streamlit as st
import cv2
import mediapipe as mp
import math
import pandas as pd
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

st.set_page_config(page_title="Driver Drowsiness AI", page_icon="🚨", layout="wide")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        self.sleep_counter = 0
        self.no_face_counter = 0
        self.SLEEP_THRESHOLD = 20
        self.NO_FACE_THRESHOLD = 15
        self.log_data = []
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        eyes_closed = False
        face_detected = False
        
        if results.multi_face_landmarks:
            face_detected = True
            for face_landmarks in results.multi_face_landmarks:
                r_top = (int(face_landmarks.landmark[159].x * w), int(face_landmarks.landmark[159].y * h))
                r_bottom = (int(face_landmarks.landmark[145].x * w), int(face_landmarks.landmark[145].y * h))
                l_top = (int(face_landmarks.landmark[386].x * w), int(face_landmarks.landmark[386].y * h))
                l_bottom = (int(face_landmarks.landmark[374].x * w), int(face_landmarks.landmark[374].y * h))
                
                right_eye_dist = get_distance(r_top, r_bottom)
                left_eye_dist = get_distance(l_top, l_bottom)
                
                if right_eye_dist < 10 and left_eye_dist < 10:
                    eyes_closed = True
        
        if eyes_closed:
            self.sleep_counter += 1
        else:
            self.sleep_counter = 0
            
        if not face_detected:
            self.no_face_counter += 1
        else:
            self.no_face_counter = 0
        
        if self.sleep_counter >= self.SLEEP_THRESHOLD or self.no_face_counter >= self.NO_FACE_THRESHOLD:
            alert_msg = "ALERT: EYES CLOSED!" if self.sleep_counter >= self.SLEEP_THRESHOLD else "ALERT: NO FACE!"
            cv2.putText(img, alert_msg, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            # Log save kar sakte ho
        else:
            cv2.putText(img, "Status: Safe & Active", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🚨 AI Driver Drowsiness Detection")
st.markdown("Browser se camera access karo - cloud compatible!")

webrtc_streamer(
    key="drowsiness",
    video_transformer_factory=DrowsinessTransformer,
    media_stream_constraints={"video": True, "audio": False}
)
