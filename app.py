import os
import time
import threading
from datetime import datetime

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="AI Driver Monitoring", page_icon="🚗", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALARM_PATH = os.path.join(BASE_DIR, "alarm.wav")

# MediaPipe FaceMesh landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


def dist(a, b):
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(p):
    return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2.0 * dist(p[0], p[3]) + 1e-6)


class DriverProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.lock = threading.Lock()

        # Thresholds (changed from the sidebar)
        self.ear_thr = 0.21
        self.mar_thr = 0.60
        self.sleep_secs = 1.5
        self.away_secs = 2.0

        # Internal timers / state
        self.eye_closed_since = None
        self.away_since = None
        self.yawn_since = None
        self.yawn_active = False

        # Metrics
        self.blinks = 0
        self.yawns = 0
        self.alerts = 0
        self.status = "Starting..."
        self.alarm = False
        self.ear_val = 0.0
        self.mar_val = 0.0
        self.gaze = "Center"
        self.events = []
        self.started = time.time()

    def _log(self, kind):
        self.events.append(
            {"time": datetime.now().strftime("%H:%M:%S"), "event": kind}
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        now = time.time()

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        status = "OK"
        alarm = False
        color = (0, 200, 0)
        gaze = "Center"
        ear = 0.0
        mar = 0.0

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            pts = np.array([[p.x * w, p.y * h] for p in lm])

            # ---- Eye Aspect Ratio ----
            ear = (eye_aspect_ratio(pts[LEFT_EYE]) + eye_aspect_ratio(pts[RIGHT_EYE])) / 2.0

            # ---- Mouth Aspect Ratio ----
            mar = dist(pts[13], pts[14]) / (dist(pts[78], pts[308]) + 1e-6)

            # ---- Head direction (simple ratio method) ----
            face_w = pts[454][0] - pts[234][0]
            face_h = pts[152][1] - pts[10][1]
            yaw_ratio = (pts[1][0] - pts[234][0]) / (face_w + 1e-6)
            pitch_ratio = (pts[1][1] - pts[10][1]) / (face_h + 1e-6)

            if yaw_ratio < 0.38:
                gaze = "Left"
            elif yaw_ratio > 0.62:
                gaze = "Right"
            elif pitch_ratio < 0.45:
                gaze = "Up"
            elif pitch_ratio > 0.72:
                gaze = "Down"

            # ---- Eyes closed / blink logic ----
            if ear < self.ear_thr:
                if self.eye_closed_since is None:
                    self.eye_closed_since = now
                closed_for = now - self.eye_closed_since
                if closed_for >= self.sleep_secs:
                    status = "DROWSY! WAKE UP"
                    alarm = True
            else:
                if self.eye_closed_since is not None:
                    if now - self.eye_closed_since < 0.5:
                        self.blinks += 1
                self.eye_closed_since = None

            # ---- Yawn logic ----
            if mar > self.mar_thr:
                if self.yawn_since is None:
                    self.yawn_since = now
                if (now - self.yawn_since) >= 0.8 and not self.yawn_active:
                    self.yawn_active = True
                    self.yawns += 1
                    self._log("Yawn")
            else:
                self.yawn_since = None
                self.yawn_active = False

            # ---- Distraction logic ----
            if gaze != "Center":
                if self.away_since is None:
                    self.away_since = now
                if now - self.away_since >= self.away_secs and not alarm:
                    status = f"DISTRACTED ({gaze})"
                    alarm = True
            else:
                self.away_since = None

            if status == "OK" and self.yawn_active:
                status = "Yawning (fatigue)"
                color = (0, 165, 255)
        else:
            # No face found
            self.eye_closed_since = None
            if self.away_since is None:
                self.away_since = now
            if now - self.away_since >= self.away_secs:
                status = "DRIVER NOT DETECTED"
                alarm = True
            else:
                status = "Searching face..."

        if alarm:
            color = (0, 0, 255)

        # ---- Update shared state ----
        with self.lock:
            if alarm and not self.alarm:
                self.alerts += 1
                self._log(status)
            self.alarm = alarm
            self.status = status
            self.ear_val = ear
            self.mar_val = mar
            self.gaze = gaze

        # ---- Draw overlay ----
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 6 if alarm else 2)
        cv2.putText(img, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(img, f"EAR {ear:.2f}  MAR {mar:.2f}  Gaze {gaze}", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def load_alarm_bytes():
    if os.path.exists(ALARM_PATH):
        with open(ALARM_PATH, "rb") as f:
            return f.read()
    return None


# ------------------------------------------------------------------ UI
st.title("🚗 AI Driver Monitoring & Safety System")
st.caption("Camera allow karo, phir START dabao. Sound ke liye browser ki volume on rakho.")

with st.sidebar:
    st.header("⚙️ Settings")
    ear_thr = st.slider("Eye closed threshold (EAR)", 0.10, 0.35, 0.21, 0.01)
    mar_thr = st.slider("Yawn threshold (MAR)", 0.30, 1.00, 0.60, 0.05)
    sleep_secs = st.slider("Eyes closed seconds for alarm", 0.5, 4.0, 1.5, 0.5)
    away_secs = st.slider("Looking away seconds for alarm", 1.0, 6.0, 2.0, 0.5)

col_video, col_info = st.columns([3, 2])

with col_video:
    ctx = webrtc_streamer(
        key="driver-monitor",
        video_processor_factory=DriverProcessor,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        rtc_configuration=RTC_CONFIG,
        async_processing=True,
    )

with col_info:
    status_ph = st.empty()
    metrics_ph = st.empty()
    audio_ph = st.empty()

alarm_bytes = load_alarm_bytes()

if "events" not in st.session_state:
    st.session_state["events"] = []

if ctx.state.playing:
    alarm_on = False
    while ctx.state.playing:
        vp = ctx.video_processor
        if vp is None:
            time.sleep(0.2)
            continue

        vp.ear_thr = ear_thr
        vp.mar_thr = mar_thr
        vp.sleep_secs = sleep_secs
        vp.away_secs = away_secs

        with vp.lock:
            status = vp.status
            alarm = vp.alarm
            ear_v, mar_v, gaze = vp.ear_val, vp.mar_val, vp.gaze
            blinks, yawns, alerts = vp.blinks, vp.yawns, vp.alerts
            st.session_state["events"] = list(vp.events)
        elapsed = int(time.time() - vp.started)

        if alarm:
            status_ph.error(f"🚨 {status}")
        elif "Yawning" in status:
            status_ph.warning(f"😮 {status}")
        else:
            status_ph.success(f"✅ {status}")

        with metrics_ph.container():
            c1, c2 = st.columns(2)
            c1.metric("EAR (eyes)", f"{ear_v:.2f}")
            c2.metric("MAR (mouth)", f"{mar_v:.2f}")
            c3, c4 = st.columns(2)
            c3.metric("Blinks", blinks)
            c4.metric("Yawns", yawns)
            c5, c6 = st.columns(2)
            c5.metric("Alerts", alerts)
            c6.metric("Session (s)", elapsed)
            st.write(f"**Head direction:** {gaze}")

        # Alarm sound (played in the visitor's browser)
        if alarm and not alarm_on and alarm_bytes:
            audio_ph.audio(alarm_bytes, format="audio/wav", autoplay=True, loop=True)
            alarm_on = True
        elif not alarm and alarm_on:
            audio_ph.empty()
            alarm_on = False

        time.sleep(0.3)

# Report section (shown when camera is stopped)
if not ctx.state.playing and st.session_state["events"]:
    st.subheader("📄 Session Report")
    df = pd.DataFrame(st.session_state["events"])
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "⬇️ Download report (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="driver_session_report.csv",
        mime="text/csv",
    )
