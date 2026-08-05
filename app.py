import streamlit as st
import cv2
import mediapipe as mp
import math
import av
import threading
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Driver Drowsiness AI Dashboard",
    page_icon="🚨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #ff4b4b; text-align: center; }
    .sub-header { text-align: center; color: #666; margin-bottom: 2rem; }
    .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b; }
    .alert-box { background-color: #ffe0e0; padding: 15px; border-radius: 10px; border: 2px solid #ff4b4b; color: #d00; font-weight: bold; }
    .safe-box { background-color: #e0ffe0; padding: 15px; border-radius: 10px; border: 2px solid #00aa00; color: #008800; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==================== MEDIAPIPE SETUP (helper) ====
mp_face_mesh = mp.solutions.face_mesh

def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# ==================== SESSION STATE ====================
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []
if "total_alerts" not in st.session_state:
    st.session_state.total_alerts = 0

# ==================== HEADER ====================
st.markdown('<div class="main-header">🚨 AI Driver Drowsiness Monitor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time fatigue detection using browser camera | Cloud Compatible ☁️</div>', unsafe_allow_html=True)
st.markdown("---")

# ==================== SIDEBAR ====================
st.sidebar.header("⚙️ Control Panel")
st.sidebar.info("💡 This app uses your **browser camera**. Grant permission when prompted.")

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Thresholds")
sleep_threshold = st.sidebar.slider("Eye Closure Threshold (frames)", 10, 60, 20, help="Kitne frames tak aankh band hone pe alert aaye")
no_face_threshold = st.sidebar.slider("No Face Threshold (frames)", 10, 60, 15, help="Kitne frames tak face na dikhe toh alert")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 About")
st.sidebar.markdown("""
- **Right Eye**: Landmarks 159-145  
- **Left Eye**: Landmarks 386-374  
- **Alert**: Red border + text  
- **Safe**: Green border  
""")

# ==================== METRICS ====================
col1, col2, col3, col4 = st.columns(4)
status_placeholder = col1.empty()
alert_count_placeholder = col2.empty()
mode_placeholder = col3.empty()
fps_placeholder = col4.empty()

status_placeholder.metric("System Status", "⏳ Idle")
alert_count_placeholder.metric("Total Alerts", st.session_state.total_alerts)
mode_placeholder.metric("Detection", "Eyes + Face")
fps_placeholder.metric("FPS", "—")

# ==================== VIDEO PROCESSOR ====================
class DrowsinessDetector(VideoTransformerBase):
    def __init__(self):
        self.sleep_counter = 0
        self.no_face_counter = 0
        self.frame_count = 0
        self.alert_active = False
        self.last_alert_time = None

        # Thread-safe queue for alerts to be consumed by main thread
        self.pending_alerts = []
        self.alert_lock = threading.Lock()

        # Create FaceMesh instance per-transformer (safe for worker thread)
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def __del__(self):
        try:
            self.face_mesh.close()
        except Exception:
            pass

    def transform(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # Process frame
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        eyes_closed = False
        face_detected = False

        if results and getattr(results, "multi_face_landmarks", None):
            face_detected = True
            for face_landmarks in results.multi_face_landmarks:
                # Right eye landmarks
                r_top = (int(face_landmarks.landmark[159].x * w), int(face_landmarks.landmark[159].y * h))
                r_bottom = (int(face_landmarks.landmark[145].x * w), int(face_landmarks.landmark[145].y * h))
                # Left eye landmarks
                l_top = (int(face_landmarks.landmark[386].x * w), int(face_landmarks.landmark[386].y * h))
                l_bottom = (int(face_landmarks.landmark[374].x * w), int(face_landmarks.landmark[374].y * h))

                # Calculate distances
                right_eye_dist = get_distance(r_top, r_bottom)
                left_eye_dist = get_distance(l_top, l_bottom)

                # Draw landmarks
                for pt in [r_top, r_bottom, l_top, l_bottom]:
                    cv2.circle(img, pt, 3, (0, 255, 255), -1)

                # Check if eyes closed (threshold: 10 pixels)
                if right_eye_dist < 10 and left_eye_dist < 10:
                    eyes_closed = True

        # Update counters
        if eyes_closed:
            self.sleep_counter += 1
        else:
            self.sleep_counter = 0

        if not face_detected:
            self.no_face_counter += 1
        else:
            self.no_face_counter = 0

        # Alert Logic
        alert_triggered = False
        alert_msg = ""
        color = (0, 255, 0)  # default green
        border_color = (0, 255, 0)
        thickness = 5

        if self.sleep_counter >= sleep_threshold:
            alert_triggered = True
            alert_msg = "⚠️ DROWSINESS DETECTED!"
            color = (0, 0, 255)  # Red
            border_color = (0, 0, 255)
            thickness = 10
        elif self.no_face_counter >= no_face_threshold:
            alert_triggered = True
            alert_msg = "⚠️ NO FACE DETECTED!"
            color = (0, 0, 255)
            border_color = (0, 0, 255)
            thickness = 10
        else:
            alert_msg = "✅ SAFE & ACTIVE"

        # Draw UI overlay
        cv2.putText(img, alert_msg, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(img, f"Sleep Counter: {self.sleep_counter}", (30, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"No Face Counter: {self.no_face_counter}", (30, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Frame: {self.frame_count}", (w - 200, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Border
        cv2.rectangle(img, (0, 0), (w, h), border_color, thickness)

        # Queue alert for main thread to log (do NOT call Streamlit API here)
        if alert_triggered and (self.last_alert_time is None or (datetime.now() - self.last_alert_time).total_seconds() > 3):
            self.last_alert_time = datetime.now()
            if not self.alert_active:
                self.alert_active = True
                alert_entry = {
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Event": "Drowsiness" if self.sleep_counter >= sleep_threshold else "No Face",
                    "Frames": self.sleep_counter if self.sleep_counter >= sleep_threshold else self.no_face_counter
                }
                with self.alert_lock:
                    self.pending_alerts.append(alert_entry)
        else:
            self.alert_active = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==================== VIDEO STREAM ====================
st.subheader("📹 Live Camera Feed")
st.caption("Click **START** to begin monitoring. Works on both local machine and cloud!")

ctx = webrtc_streamer(
    key="drowsiness-detection",
    video_transformer_factory=DrowsinessDetector,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    async_transform=True
)

# ==================== UPDATE METRICS & TRANSFER PENDING ALERTS TO SESSION STATE ====
if ctx.state.playing:
    status_placeholder.metric("System Status", "🟢 Active")
    if ctx.video_transformer:
        # Transfer pending alerts from transformer (worker thread) into Streamlit session state (main thread)
        transformer = ctx.video_transformer
        if hasattr(transformer, "pending_alerts") and hasattr(transformer, "alert_lock"):
            with transformer.alert_lock:
                while transformer.pending_alerts:
                    alert = transformer.pending_alerts.pop(0)
                    st.session_state.alert_log.append(alert)
                    st.session_state.total_alerts += 1

        alert_count_placeholder.metric("Total Alerts", st.session_state.total_alerts)
        fps_placeholder.metric("Frame Count", ctx.video_transformer.frame_count)
else:
    status_placeholder.metric("System Status", "⏳ Idle")

# ==================== LOGS ====================
st.markdown("---")
st.subheader("📋 Alert Log")

if st.session_state.alert_log:
    import pandas as pd
    log_df = pd.DataFrame(st.session_state.alert_log)
    st.dataframe(log_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download CSV",
            data=log_df.to_csv(index=False).encode('utf-8'),
            file_name=f"drowsiness_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    with col2:
        if st.button("🗑️ Clear Logs"):
            st.session_state.alert_log = []
            st.session_state.total_alerts = 0
            st.experimental_rerun()
else:
    st.info("No alerts yet. Start monitoring to see logs here.")

# ==================== FOOTER ====================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + MediaPipe + WebRTC | Deploy anywhere!")
