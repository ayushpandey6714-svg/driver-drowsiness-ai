import streamlit as st
import cv2
import mediapipe as mp
import math
import pygame
import os
import time
import pandas as pd
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Driver Drowsiness AI Dashboard",
    page_icon="🚨",
    layout="wide"
)

# Initialize Pygame Mixer for Audio
pygame.mixer.init()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Session State for Logs & Counters
if "log_data" not in st.session_state:
    st.session_state.log_data = []

# UI Styling & Header
st.title("🚨 AI-Powered Driver Drowsiness & Safety Monitoring System")
st.markdown("---")

# Sidebar Controls
st.sidebar.header("⚙️ System Control Panel")
run_system = st.sidebar.checkbox("Start Live Monitoring", value=False)

st.sidebar.markdown("---")
st.sidebar.info("This system tracks eye closure and head drops using MediaPipe and triggers an emergency alarm.")

# Main Layout Columns
col1, col2 = st.sidebar, None # Layout split

video_placeholder = st.empty()
status_placeholder = st.empty()

# CSV Log file initialization
LOG_FILE = "drowsiness_logs.csv"

def save_log(event_type, duration):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {"Timestamp": timestamp, "Event": event_type, "Duration (Frames)": duration}
    st.session_state.log_data.append(new_entry)
    
    # Save to CSV
    df = pd.DataFrame(st.session_state.log_data)
    df.to_csv(LOG_FILE, index=False)

if run_system:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    sleep_counter = 0
    no_face_counter = 0
    SLEEP_THRESHOLD = 20
    NO_FACE_THRESHOLD = 15

    alarm_sound = None
    if os.path.exists("alarm.wav"):
        alarm_sound = pygame.mixer.Sound("alarm.wav")

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

        # Logic tracking
        if eyes_closed:
            sleep_counter += 1
        else:
            sleep_counter = 0

        if not face_detected:
            no_face_counter += 1
        else:
            no_face_counter = 0

        # Trigger Actions
        if sleep_counter >= SLEEP_THRESHOLD or no_face_counter >= NO_FACE_THRESHOLD:
            alert_msg = "ALERT: EYES CLOSED!" if sleep_counter >= SLEEP_THRESHOLD else "ALERT: HEAD DROPPED / NO FACE!"
            cv2.putText(frame, alert_msg, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            
            if alarm_sound and not pygame.mixer.get_busy():
                alarm_sound.play(-1)
            
            # Log event
            save_log(alert_msg, sleep_counter if sleep_counter >= SLEEP_THRESHOLD else no_face_counter)
        else:
            if alarm_sound:
                alarm_sound.stop()
            cv2.putText(frame, "Status: Safe & Active", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert frame to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Stop condition if user unchecks the sidebar toggle
        # (Streamlit loop control check)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if alarm_sound:
        alarm_sound.stop()
    cv2.destroyAllWindows()
else:
    video_placeholder.warning("⚠️ System is paused. Check 'Start Live Monitoring' in the sidebar to launch the dashboard.")

# Session Logs & Summary Section
st.markdown("---")
st.subheader("📊 Session Drowsiness Logs & Report")

if os.path.exists(LOG_FILE):
    log_df = pd.read_csv(LOG_FILE)
    if not log_df.empty:
        st.dataframe(log_df, use_container_width=True)
        
        # Download button for CSV report
        st.download_button(
            label="📥 Download Session Report (CSV)",
            data=log_df.to_csv(index=False).encode('utf-8'),
            file_name="drowsiness_session_report.csv",
            mime="text/csv",
        )
    else:
        st.info("No drowsiness events logged in this session yet.")
else:
    st.info("No log file created yet.")