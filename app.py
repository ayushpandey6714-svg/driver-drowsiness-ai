import streamlit as st
import cv2
import mediapipe as mp
import math
import os
import pandas as pd
from datetime import datetime
import pygame
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# --- Page Config ---
st.set_page_config(page_title="Driver Drowsiness AI Dashboard", page_icon="🚨", layout="wide")

# --- CSS Styling ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); }
    </style>
""", unsafe_allow_html=True)

# --- Sound Initialization (Sirf ek baar) ---
if "sound_initialized" not in st.session_state:
    pygame.mixer.init()
    st.session_state.sound_initialized = True

# --- CSV Log File Setup ---
LOG_FILE = "drowsiness_logs.csv"
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["Timestamp", "Event", "Duration (Frames)"]).to_csv(LOG_FILE, index=False)

# --- Session State for Dashboard ---
if "alert_count" not in st.session_state:
    st.session_state.alert_count = 0

# --- UI Header ---
st.title("🚨 AI-Powered Driver Drowsiness & Safety Monitoring")
st.markdown("Real-time eye closure & head drop detection using MediaPipe + WebRTC.")
st.markdown("---")

# --- Metrics Row ---
col1, col2, col3 = st.columns(3)
with col1:
    status_metric = st.metric(label="System Status", value="Stopped ⏹️")
with col2:
    alert_metric = st.metric(label="Total Alerts Triggered", value=st.session_state.alert_count)
with col3:
    mode_metric = st.metric(label="Detection Mode", value="Eyes + Head Drop")

st.markdown("---")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Control Panel")
run_stream = st.sidebar.checkbox("🟢 Start Live Monitoring", value=True)
st.sidebar.info("💡 **Tip:** Make sure your face is well-lit for better detection.")

# --- Video Processor Class (Yahan saara logic hai) ---
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        # MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        
        # Counters
        self.sleep_counter = 0
        self.no_face_counter = 0
        self.SLEEP_THRESHOLD = 20
        self.NO_FACE_THRESHOLD = 15
        
        # Alarm Sound Load
        self.alarm_sound = None
        if os.path.exists("alarm.wav"):
            self.alarm_sound = pygame.mixer.Sound("alarm.wav")
        
        # Cooldown taaki baar baar alert na aaye
        self.last_alert_time = datetime.now()
        self.alert_cooldown = 3  # seconds

    def get_distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def save_log(self, event_type, duration):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = {"Timestamp": timestamp, "Event": event_type, "Duration (Frames)": duration}
        df = pd.DataFrame([new_entry])
        if os.path.exists(LOG_FILE):
            df_existing = pd.read_csv(LOG_FILE)
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_csv(LOG_FILE, index=False)
        st.session_state.alert_count += 1

    def recv(self, frame):
        # Webcam se frame aaya
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # BGR ko RGB mein convert (MediaPipe ke liye)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        eyes_closed = False
        face_detected = False
        alert_triggered = False
        
        # Agar face detect hua
        if results.multi_face_landmarks:
            face_detected = True
            for face_landmarks in results.multi_face_landmarks:
                # Right Eye landmarks
                r_top = (int(face_landmarks.landmark[159].x * w), int(face_landmarks.landmark[159].y * h))
                r_bottom = (int(face_landmarks.landmark[145].x * w), int(face_landmarks.landmark[145].y * h))
                # Left Eye landmarks
                l_top = (int(face_landmarks.landmark[386].x * w), int(face_landmarks.landmark[386].y * h))
                l_bottom = (int(face_landmarks.landmark[374].x * w), int(face_landmarks.landmark[374].y * h))
                
                # Distance nikaalo
                right_eye_dist = self.get_distance(r_top, r_bottom)
                left_eye_dist = self.get_distance(l_top, l_bottom)
                
                # Agar dono aankhon ke beech distance < 10 hai toh closed maano
                if right_eye_dist < 10 and left_eye_dist < 10:
                    eyes_closed = True
                
                # Markers draw karo (dikhne mein acha lagega)
                cv2.circle(img, r_top, 3, (0, 255, 0), -1)
                cv2.circle(img, r_bottom, 3, (0, 255, 0), -1)
                cv2.circle(img, l_top, 3, (0, 255, 0), -1)
                cv2.circle(img, l_bottom, 3, (0, 255, 0), -1)
        
        # Counter update
        if eyes_closed:
            self.sleep_counter += 1
        else:
            self.sleep_counter = 0
        
        if not face_detected:
            self.no_face_counter += 1
        else:
            self.no_face_counter = 0
        
        # --- Alert Condition Check ---
        current_time = datetime.now()
        if (self.sleep_counter >= self.SLEEP_THRESHOLD or self.no_face_counter >= self.NO_FACE_THRESHOLD):
            
            # Cooldown check (taaki 3 second mein sirf ek baar alert aaye)
            if (current_time - self.last_alert_time).total_seconds() > self.alert_cooldown:
                alert_triggered = True
                self.last_alert_time = current_time
                
                if self.sleep_counter >= self.SLEEP_THRESHOLD:
                    alert_msg = "ALERT: EYES CLOSED!"
                    duration = self.sleep_counter
                else:
                    alert_msg = "ALERT: HEAD DROPPED / NO FACE!"
                    duration = self.no_face_counter
                
                # Log save karo
                self.save_log(alert_msg, duration)
                
                # Alarm bajao
                if self.alarm_sound:
                    pygame.mixer.stop()  # Pehle wala band karo
                    self.alarm_sound.play()
            
            # Frame pe Red alert likho
            if self.sleep_counter >= self.SLEEP_THRESHOLD:
                cv2.putText(img, "ALERT: EYES CLOSED!", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
            else:
                cv2.putText(img, "ALERT: HEAD DROPPED!", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
        else:
            # Safe hai toh alarm band karo aur Green text dikhao
            pygame.mixer.stop()
            cv2.putText(img, "Status: Safe & Active", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # Processed frame return karo
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit WebRTC Streamer ---
if run_stream:
    status_metric.metric(label="System Status", value="Active 🟢")
    
    webrtc_streamer(
        key="driver-drowsiness-cam",
        video_processor_factory=DrowsinessProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": {"facing_mode": "user"}}
    )
else:
    status_metric.metric(label="System Status", value="Stopped ⏹️")
    st.info("👈 Toggle 'Start Live Monitoring' to activate the camera.")
    pygame.mixer.stop()

# --- Bottom Section: Logs aur Download ---
st.markdown("---")
st.subheader("📋 Session Logs & Reports")

if os.path.exists(LOG_FILE):
    log_df = pd.read_csv(LOG_FILE)
    st.dataframe(log_df.tail(10), use_container_width=True)
    
    # Metric ko update karo latest data se
    st.session_state.alert_count = len(log_df)
    alert_metric.metric(label="Total Alerts Triggered", value=st.session_state.alert_count)
    
    col_dl, col_del = st.columns([1, 1])
    with col_dl:
        st.download_button(
            label="📥 Download Full Report (CSV)",
            data=log_df.to_csv(index=False).encode('utf-8'),
            file_name="drowsiness_session_report.csv",
            mime="text/csv",
        )
    with col_del:
        if st.button("🗑️ Clear Logs"):
            pd.DataFrame(columns=["Timestamp", "Event", "Duration (Frames)"]).to_csv(LOG_FILE, index=False)
            st.session_state.alert_count = 0
            st.rerun()
else:
    st.info("No logs generated yet. Start monitoring to collect data.")

mode_metric.metric(label="Detection Mode", value="Eyes + Head Drop")
