import streamlit as st
import cv2
import mediapipe as mp
import math
import os
import pandas as pd
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Driver Drowsiness AI Dashboard",
    page_icon="🚨",
    layout="wide"
)

# Custom CSS for UI styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Session State for Logs & Counters
if "log_data" not in st.session_state:
    st.session_state.log_data = []
if "alert_count" not in st.session_state:
    st.session_state.alert_count = []

# Dashboard Header
st.title("🚨 AI-Powered Driver Drowsiness & Safety Monitoring System")
st.markdown("Monitor driver fatigue, eye closures, and head drops in real-time with advanced computer vision.")
st.markdown("---")

# Sidebar Controls
st.sidebar.header("⚙️ Control Panel")
run_system = st.sidebar.toggle("🟢 Turn On Live Monitoring", value=False)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Pro-Tip:** Ensure proper lighting on your face for optimal MediaPipe facial landmark tracking.")

# Metrics Row
col1, col2, col3 = st.columns(3)
with col1:
    status_metric = st.empty()
    status_metric.metric(label="System Status", value="Idle 😴" if not run_system else "Active 🟢")
with col2:
    alert_metric = st.empty()
    alert_metric.metric(label="Total Alerts Triggered", value=len(st.session_state.log_data))
with col3:
    mode_metric = st.empty()
    mode_metric.metric(label="Detection Mode", value="Eyes + Head Drop")

st.markdown("---")

# Main Feed & Logs layout
video_col, log_col = st.columns([2, 1])

with video_col:
    st.subheader("📹 Live Camera Feed")
    video_placeholder = st.empty()

with log_col:
    st.subheader("📊 Quick Session Stats")

LOG_FILE = "drowsiness_logs.csv"

def save_log(event_type, duration):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {"Timestamp": timestamp, "Event": event_type, "Duration (Frames)": duration}
    st.session_state.log_data.append(new_entry)
    df = pd.DataFrame(st.session_state.log_data)
    df.to_csv(LOG_FILE, index=False)

if run_system:
    status_metric.metric(label="System Status", value="Active 🟢")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    sleep_counter = 0
    no_face_counter = 0
    SLEEP_THRESHOLD = 20
    NO_FACE_THRESHOLD = 15

    while run_system:
        success, frame = cap.read()
        if not success:
            st.error("Failed to access webcam.")
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            sleep_counter += 1
        else:
            sleep_counter = 0

        if not face_detected:
            no_face_counter += 1
        else:
            no_face_counter = 0

        if sleep_counter >= SLEEP_THRESHOLD or no_face_counter >= NO_FACE_THRESHOLD:
            alert_msg = "ALERT: EYES CLOSED!" if sleep_counter >= SLEEP_THRESHOLD else "ALERT: HEAD DROPPED / NO FACE!"
            cv2.putText(frame, alert_msg, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            
            save_log(alert_msg, sleep_counter if sleep_counter >= SLEEP_THRESHOLD else no_face_counter)
            alert_metric.metric(label="Total Alerts Triggered", value=len(st.session_state.log_data))
        else:
            cv2.putText(frame, "Status: Safe & Active", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
else:
    video_placeholder.info("👈 Toggle 'Turn On Live Monitoring' in the sidebar to activate the AI video feed.")

# Bottom Section: Detailed Logs & Report Download
st.markdown("---")
st.subheader("📋 Detailed Session Drowsiness Logs")

if os.path.exists(LOG_FILE):
    log_df = pd.read_csv(LOG_FILE)
    if not log_df.empty:
        log_col.dataframe(log_df.tail(5), use_container_width=True)
        st.dataframe(log_df, use_container_width=True)
        st.download_button(
            label="📥 Download Full Session Report (.CSV)",
            data=log_df.to_csv(index=False).encode('utf-8'),
            file_name="drowsiness_session_report.csv",
            mime="text/csv",
        )
    else:
        st.info("No drowsiness events logged yet.")
else:
    st.info("No log file generated yet.")
