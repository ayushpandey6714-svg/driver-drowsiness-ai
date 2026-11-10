from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from detector import initialize_face_mesh, process_frame
from fatigue import FatigueDetector


st.set_page_config(
    page_title="Driver Drowsiness Monitor",
    page_icon="🚗",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp { background: #0b1020; }
      .block-container { max-width: 1200px; padding-top: 2rem; }
      [data-testid="stMetric"] { background: #111a2e; padding: 1rem; border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚗 Driver Drowsiness Monitor")
st.caption(
    "A demo that estimates eye closure, head direction, and yawning from images or video. "
    "It is not a certified safety system and must not be used as a substitute for safe driving."
)

if "face_mesh" not in st.session_state:
    st.session_state.face_mesh = initialize_face_mesh()
if "logs" not in st.session_state:
    st.session_state.logs = []


def analyze_frame(frame: np.ndarray, detector: FatigueDetector) -> tuple[np.ndarray, dict]:
    """Run face landmarks and fatigue metrics on one BGR frame."""
    processed, results = process_frame(frame, st.session_state.face_mesh)
    faces = getattr(results, "multi_face_landmarks", None)
    landmarks = faces[0].landmark if faces else None
    metrics = detector.process_landmarks(landmarks, processed.shape)

    if landmarks:
        height, width = processed.shape[:2]
        xs = [int(point.x * width) for point in landmarks]
        ys = [int(point.y * height) for point in landmarks]
        x1, x2 = max(0, min(xs)), min(width - 1, max(xs))
        y1, y2 = max(0, min(ys)), min(height - 1, max(ys))
        color = (40, 200, 120) if metrics["status"] == "ATTENTIVE" else (40, 70, 235)
        cv2.rectangle(processed, (x1, y1), (x2, y2), color, 2)
        for index in (33, 133, 362, 263, 61, 291, 1):
            point = landmarks[index]
            cv2.circle(processed, (int(point.x * width), int(point.y * height)), 3, (0, 220, 255), -1)

    cv2.putText(
        processed,
        metrics["status"],
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (40, 70, 235) if metrics["status"] != "ATTENTIVE" else (40, 200, 120),
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), metrics


def add_log(source: str, frame_number: int | str, metrics: dict) -> None:
    st.session_state.logs.append(
        {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Source": source,
            "Frame": frame_number,
            "Status": metrics["status"],
            "Gaze": metrics["gaze"],
            "EAR": metrics["ear"],
            "MAR": metrics["mar"],
            "Fatigue index": metrics["fatigue_index"],
            "Blinks": metrics["blinks"],
            "Yawns": metrics["yawns"],
        }
    )


def show_metrics(metrics: dict) -> None:
    status = metrics["status"]
    if status == "DROWSY":
        st.error("🚨 Drowsiness indicators detected")
    elif status in ("DISTRACTED", "NO FACE DETECTED"):
        st.warning(f"⚠️ {status}")
    else:
        st.success("✅ No drowsiness indicators detected in this sample")

    cols = st.columns(5)
    cols[0].metric("Fatigue index", f'{metrics["fatigue_index"]:.2f}')
    cols[1].metric("Eye aspect ratio", f'{metrics["ear"]:.3f}')
    cols[2].metric("Mouth aspect ratio", f'{metrics["mar"]:.3f}')
    cols[3].metric("Blinks", metrics["blinks"])
    cols[4].metric("Yawns", metrics["yawns"])
    st.write(f'**Gaze:** {metrics["gaze"]}')


mode = st.radio(
    "Choose an input",
    ("Camera snapshot", "Upload image", "Upload video"),
    horizontal=True,
)

if mode == "Camera snapshot":
    st.info(
        "The browser camera control takes a snapshot (it does not provide a continuous live stream). "
        "For frame-by-frame monitoring, upload a video. Grant camera permission if prompted."
    )
    image_file = st.camera_input("Take a face snapshot")
    if image_file and st.button("Analyze snapshot", type="primary"):
        image = cv2.imdecode(np.frombuffer(image_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            st.error("Could not read the captured image. Please take another snapshot.")
        else:
            annotated, metrics = analyze_frame(image, FatigueDetector())
            left, right = st.columns([1.4, 1])
            with left:
                st.image(annotated, caption="Analyzed snapshot", use_container_width=True)
            with right:
                show_metrics(metrics)
            add_log("Camera snapshot", "-", metrics)

elif mode == "Upload image":
    image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if image_file and st.button("Analyze image", type="primary"):
        image = cv2.imdecode(np.frombuffer(image_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            st.error("Could not read this image. Try a JPG or PNG file.")
        else:
            annotated, metrics = analyze_frame(image, FatigueDetector())
            left, right = st.columns([1.4, 1])
            with left:
                st.image(annotated, caption="Analyzed image", use_container_width=True)
            with right:
                show_metrics(metrics)
            add_log(image_file.name, "-", metrics)

else:
    video_file = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv"])
    sample_every = st.slider("Analyze every Nth frame", min_value=1, max_value=30, value=5)
    max_samples = st.slider("Maximum analyzed frames", min_value=30, max_value=600, value=180, step=30)

    if video_file and st.button("Analyze video", type="primary"):
        suffix = Path(video_file.name).suffix or ".mp4"
        video_path = None
        cap = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_video:
                temp_video.write(video_file.getvalue())
                video_path = temp_video.name

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Could not open this video. Please try an MP4 encoded with H.264.")
            else:
                detector = FatigueDetector()
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress = st.progress(0, text="Analyzing video…")
                preview = st.empty()
                sampled = []
                frame_index = 0
                processed_count = 0

                while processed_count < max_samples:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_index % sample_every == 0:
                        annotated, metrics = analyze_frame(frame, detector)
                        sampled.append(metrics)
                        add_log(video_file.name, frame_index, metrics)
                        preview.image(annotated, caption=f"Analyzed frame {frame_index}", use_container_width=True)
                        processed_count += 1
                        if frame_count > 0:
                            progress.progress(min(frame_index / frame_count, 1.0), text=f"Analyzed {processed_count} frames…")
                    frame_index += 1

                progress.empty()
                if not sampled:
                    st.warning("No video frames could be decoded.")
                else:
                    drowsy = sum(item["status"] == "DROWSY" for item in sampled)
                    distracted = sum(item["status"] in ("DISTRACTED", "NO FACE DETECTED") for item in sampled)
                    average_fatigue = float(np.mean([item["fatigue_index"] for item in sampled]))
                    st.subheader("Video summary")
                    summary_cols = st.columns(4)
                    summary_cols[0].metric("Frames analyzed", len(sampled))
                    summary_cols[1].metric("Drowsy samples", drowsy)
                    summary_cols[2].metric("Distracted / no-face samples", distracted)
                    summary_cols[3].metric("Average fatigue index", f"{average_fatigue:.2f}")
                    st.caption("Results are estimates for the sampled frames; they are not a medical or driving-safety assessment.")
        except Exception as exc:
            st.error(f"Video analysis failed: {exc}")
        finally:
            if cap is not None:
                cap.release()
            if video_path and Path(video_path).exists():
                Path(video_path).unlink(missing_ok=True)

if st.session_state.logs:
    st.divider()
    st.subheader("Analysis history")
    history = pd.DataFrame(st.session_state.logs)
    st.dataframe(history.tail(100), use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV report",
        data=history.to_csv(index=False).encode("utf-8"),
        file_name="driver_drowsiness_report.csv",
        mime="text/csv",
    )
    if st.button("Clear analysis history"):
        st.session_state.logs = []
        st.rerun()

with st.expander("Deployment and privacy notes"):
    st.markdown(
        "- On Streamlit Cloud, camera access is through the browser snapshot control; the server cannot access your computer's webcam directly.\n"
        "- Uploaded media and camera snapshots are processed in the app session and are not intentionally saved to disk, except for a temporary video file removed after processing.\n"
        "- For local use, run `streamlit run app.py` after installing `requirements.txt`.\n"
        "- This project is an educational prototype, not a certified driver-monitoring or emergency-alert system."
    )
