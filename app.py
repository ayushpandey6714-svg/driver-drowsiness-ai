import streamlit as st
import cv2
import time
import pygame
import os
import numpy as np
import pandas as pd
from datetime import datetime
from detector import initialize_face_mesh, process_frame
from fatigue import FatigueDetector

@st.cache_resource
def load_alarm_sound():
    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=44100, size=-16, channels=1)
    
    if os.path.exists("alarm.wav"):
        return pygame.mixer.Sound("alarm.wav")
    else:
        duration = 0.2
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * 880.0 * t) * 0.4
        sound_array = (tone * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(sound_array)

alarm_sound = load_alarm_sound()

# Page Setup
st.set_page_config(page_title="AI Driver Monitoring System", layout="wide", initial_sidebar_state="expanded")

# UI Styling
st.markdown("""
    <style>
    .stApp { background-color: #080c14; font-family: 'Inter', sans-serif; }
    .main-header-container {
        display: flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        border: 1px solid #1e293b; padding: 18px; border-radius: 16px; margin-bottom: 20px;
    }
    .main-header {
        font-size: 26px; font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: 2px; text-transform: uppercase;
    }
    .metric-card {
        background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px; padding: 16px 20px; margin-bottom: 14px;
    }
    .status-attentive {
        background: rgba(16, 185, 129, 0.12); border: 1.5px solid #10b981; color: #34d399;
        font-size: 18px; font-weight: 800; text-align: center; padding: 14px; border-radius: 12px; margin-bottom: 16px;
    }
    .status-drowsy {
        background: rgba(239, 68, 68, 0.15); border: 1.5px solid #ef4444; color: #f87171;
        font-size: 18px; font-weight: 800; text-align: center; padding: 14px; border-radius: 12px; margin-bottom: 16px;
    }
    .status-distracted {
        background: rgba(245, 158, 11, 0.15); border: 1.5px solid #f59e0b; color: #fbbf24;
        font-size: 18px; font-weight: 800; text-align: center; padding: 14px; border-radius: 12px; margin-bottom: 16px;
    }
    .card-label { color: #94a3b8; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
    .value-highlight { font-size: 30px; font-weight: 800; color: #38bdf8; }
    .value-sub { font-size: 26px; font-weight: 700; color: #f3f4f6; }
    .cam-off-box {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        height: 450px; background: rgba(15, 23, 42, 0.5); border: 2px dashed #1e293b;
        border-radius: 16px; color: #64748b; font-size: 18px; font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header-container"><div class="main-header">🚗 AI Driver Monitoring System</div></div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### ⚙️ Control Panel")
cam_index = st.sidebar.number_input("Camera Source Index", value=0, step=1)
start_system = st.sidebar.toggle("Start Camera Feed", value=False)

# Session State
if "face_mesh" not in st.session_state:
    st.session_state.face_mesh = initialize_face_mesh()
if "detector" not in st.session_state:
    st.session_state.detector = FatigueDetector()
if "session_logs" not in st.session_state:
    st.session_state.session_logs = []
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "distraction_start_time" not in st.session_state:
    st.session_state.distraction_start_time = None

col_video, col_stats = st.columns([2.3, 1])

with col_video:
    video_placeholder = st.empty()

with col_stats:
    status_placeholder = st.empty()
    fatigue_placeholder = st.empty()
    metrics_placeholder = st.empty()
    gaze_placeholder = st.empty()

if start_system:
    if st.session_state.start_time is None:
        st.session_state.start_time = datetime.now()
        st.session_state.session_logs = []  # Reset log for new session

    cap = cv2.VideoCapture(cam_index)
    prev_time = time.time()

    while cap.isOpened() and start_system:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to fetch camera feed.")
            break

        frame, results = process_frame(frame, st.session_state.face_mesh)
        
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30
        prev_time = curr_time

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            status_data = st.session_state.detector.process_landmarks(landmarks, frame.shape)
        else:
            status_data = st.session_state.detector.process_landmarks(None, frame.shape)

        # Record Log Entry
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.session_logs.append({
            "Time": timestamp,
            "Status": status_data["status"],
            "Gaze": status_data["gaze"],
            "EAR": status_data["ear"],
            "MAR": status_data["mar"],
            "Fatigue Index": status_data["fatigue_index"]
        })

        # --- REALISTIC ALARM TRIGGER LOGIC ---
        curr_status = status_data["status"]
        
        if curr_status == "DROWSY":
            st.session_state.distraction_start_time = None
            if not pygame.mixer.get_busy():
                alarm_sound.play()
                
        elif curr_status in ["DISTRACTED", "NO FACE DETECTED"]:
            if st.session_state.distraction_start_time is None:
                st.session_state.distraction_start_time = time.time()
            elif time.time() - st.session_state.distraction_start_time > 1.5:
                if not pygame.mixer.get_busy():
                    alarm_sound.play()
        else:
            st.session_state.distraction_start_time = None
            pygame.mixer.stop()

        # Render FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Dynamic Status UI
        if status_data["status"] == "DROWSY":
            status_placeholder.markdown('<div class="status-drowsy">🚨 STATUS: DROWSY / SLEEPING!</div>', unsafe_allow_html=True)
        elif status_data["status"] == "DISTRACTED":
            status_placeholder.markdown('<div class="status-distracted">⚠️ STATUS: DISTRACTED!</div>', unsafe_allow_html=True)
        elif status_data["status"] == "NO FACE DETECTED":
            status_placeholder.markdown('<div class="status-distracted">⚠️ WARNING: NO FACE DETECTED!</div>', unsafe_allow_html=True)
        else:
            status_placeholder.markdown('<div class="status-attentive">✅ STATUS: ATTENTIVE</div>', unsafe_allow_html=True)

        # Dynamic Cards UI
        fatigue_placeholder.markdown(f'<div class="metric-card"><div class="card-label">Fatigue Index</div><div class="value-highlight">{status_data["fatigue_index"]:.2f}</div></div>', unsafe_allow_html=True)
        metrics_placeholder.markdown(f'<div style="display: flex; gap: 12px;"><div class="metric-card" style="flex: 1;"><div class="card-label">Blinks</div><div class="value-sub">{status_data["blinks"]}</div></div><div class="metric-card" style="flex: 1;"><div class="card-label">Yawns</div><div class="value-sub">{status_data["yawns"]}</div></div></div>', unsafe_allow_html=True)
        gaze_placeholder.markdown(f'<div class="metric-card"><div class="card-label">Gaze Tracking</div><div class="value-highlight">{status_data["gaze"]}</div></div>', unsafe_allow_html=True)

    cap.release()

else:
    # Idle Screen
    video_placeholder.markdown('''
        <div class="cam-off-box">
            <p>📷 Camera is currently OFF</p>
            <p style="font-size: 14px; color: #475569;">Toggle 'Start Camera Feed' in sidebar to begin</p>
        </div>
    ''', unsafe_allow_html=True)

    status_placeholder.markdown('<div class="status-attentive" style="border-color:#334155; color:#94a3b8; background:rgba(30,41,59,0.3);">⏸️ SYSTEM IDLE</div>', unsafe_allow_html=True)
    fatigue_placeholder.markdown('<div class="metric-card"><div class="card-label">Fatigue Index</div><div class="value-highlight">0.00</div></div>', unsafe_allow_html=True)
    metrics_placeholder.markdown('<div style="display: flex; gap: 12px;"><div class="metric-card" style="flex: 1;"><div class="card-label">Blinks</div><div class="value-sub">0</div></div><div class="metric-card" style="flex: 1;"><div class="card-label">Yawns</div><div class="value-sub">0</div></div></div>', unsafe_allow_html=True)
    gaze_placeholder.markdown('<div class="metric-card"><div class="card-label">Gaze Tracking</div><div class="value-highlight">OFF</div></div>', unsafe_allow_html=True)

    # GENERATE REPORT SECTION WHEN CAMERA STOPS
    if len(st.session_state.session_logs) > 0:
        st.markdown("---")
        st.subheader("📊 Session Analytics & Trip Report")

        df = pd.DataFrame(st.session_state.session_logs)
        
        total_frames = len(df)
        drowsy_count = len(df[df["Status"] == "DROWSY"])
        distracted_count = len(df[df["Status"] == "DISTRACTED"])
        avg_fatigue = df["Fatigue Index"].mean()

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("Total Recorded Frames", total_frames)
        col_r2.metric("Drowsy Events", drowsy_count)
        col_r3.metric("Distraction Events", distracted_count)
        col_r4.metric("Avg Fatigue Score", f"{avg_fatigue:.2f}")

        os.makedirs("reports", exist_ok=True)
        report_filename = f"reports/driver_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(report_filename, index=False)

        csv_data = df.to_csv(index=False).encode('utf-8')
        
        col_btn1, col_btn2 = st.columns([1, 2])
        with col_btn1:
            st.download_button(
                label="📥 Download Session Report (CSV)",
                data=csv_data,
                file_name=os.path.basename(report_filename),
                mime="text/csv"
            )
        with col_btn2:
            st.success(f"✅ Report Auto-saved locally to `{report_filename}`")

        with st.expander("🔍 View Raw Log Data"):
            st.dataframe(df, use_container_width=True)