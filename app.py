import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import math
import numpy as np
import av
import threading

# Page Configuration
st.set_page_config(
    page_title="Driver Drowsiness AI Dashboard",
    page_icon="🚨",
    layout="wide"
)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Use a lock and shared state for thread-safe communication
lock = threading.Lock()
class SharedState:
    def __init__(self):
        self.alert_count = 0
        self.last_alert = "None"

shared_state = SharedState()

class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.sleep_counter = 0
        self.no_face_counter = 0
        self.SLEEP_THRESHOLD = 20
        self.NO_FACE_THRESHOLD = 15

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

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
            cv2.putText(img, alert_msg, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            with lock:
                shared_state.last_alert = alert_msg
        else:
            cv2.putText(img, "Status: Safe & Active", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# UI Layout
st.title("🚨 AI Driver Drowsiness Monitoring System")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Feed")
    ctx = webrtc_streamer(
        key="drowsiness-detection",
        video_processor_factory=DrowsinessProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("📊 System Status")
    status_placeholder = st.empty()
    alert_placeholder = st.empty()
    
    if ctx.state.playing:
        status_placeholder.success("System is Active 🟢")
        with lock:
            alert_placeholder.warning(f"Last Alert: {shared_state.last_alert}")
    else:
        status_placeholder.info("System is Idle 😴")

st.sidebar.header("⚙️ Settings")
st.sidebar.info("This app uses WebRTC for real-time video processing in the browser.")
