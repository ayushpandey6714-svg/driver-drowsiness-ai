from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from detector import initialize_face_mesh, process_frame
from fatigue import FatigueDetector


st.set_page_config(page_title="AURORA | Driver Monitor", page_icon="🚘", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#e7eefb; --muted:#91a1bb; --cyan:#57e4f2; --blue:#6d8cff; --green:#41e0ad; --amber:#ffbd66; --red:#ff637d; }
    .stApp { background: radial-gradient(ellipse at 12% 0%, #14294b 0, transparent 34%), radial-gradient(ellipse at 90% 15%, #202451 0, transparent 32%), #080d18; color:var(--ink); font-family:'Manrope',sans-serif; }
    .block-container { max-width:1420px; padding:2rem 2rem 3rem; }
    [data-testid="stHeader"] { background:transparent; }
    h1,h2,h3,p,label,span,div { font-family:'Manrope',sans-serif; }
    h1 { letter-spacing:-1.6px; }
    .hero { position:relative; overflow:hidden; padding:26px 30px; border:1px solid #ffffff19; border-radius:22px; background:linear-gradient(120deg,#121f35dd,#11152bdd); box-shadow:0 24px 70px #0005,inset 0 1px #ffffff12; margin:4px 0 22px; }
    .hero:after { content:''; position:absolute; width:270px;height:270px;right:-65px;top:-150px;border-radius:50%;background:#57e4f225;filter:blur(20px); }
    .eyebrow { color:var(--cyan);font:500 11px 'DM Mono',monospace;letter-spacing:2.2px;text-transform:uppercase; }
    .hero-title { color:#f5f8ff;font-size:clamp(28px,4vw,43px);font-weight:800;letter-spacing:-1.8px;margin:9px 0 5px; }
    .hero-copy { color:#9eacc2;font-size:14px;max-width:760px;line-height:1.7; }
    .glass { border:1px solid #ffffff17;border-radius:18px;background:linear-gradient(145deg,#17233acc,#111a2acc);box-shadow:0 16px 38px #0003,inset 0 1px #ffffff0d; padding:18px 20px; }
    .section-label { color:#9babbe;font:500 11px 'DM Mono',monospace;letter-spacing:1.7px;text-transform:uppercase;margin-bottom:10px; }
    .status-line { border:1px solid #57e4f23d;background:linear-gradient(110deg,#14343b,#131f30);color:#abf5ef;border-radius:13px;padding:13px 16px;font-weight:700;margin:10px 0 16px; }
    [data-testid="stMetric"] { background:linear-gradient(145deg,#18253acc,#121a2acc);border:1px solid #ffffff14;border-radius:15px;padding:14px 16px;box-shadow:0 9px 24px #0002; }
    [data-testid="stMetricLabel"] { color:#aab8cd !important; }
    [data-testid="stMetricValue"] { color:#f3f7ff !important; }
    div.stButton>button[kind="primary"] { border:0;border-radius:12px;background:linear-gradient(100deg,#26bdcf,#657fff);color:#061322;font-weight:800;box-shadow:0 7px 22px #3acada2b; }
    div.stButton>button { border-radius:11px; }
    [data-testid="stFileUploader"] { border:1px dashed #7ca0d655;border-radius:15px;background:#111b2b66; }
    [data-testid="stRadio"] label { color:#d5e0f0; }
    .footnote { color:#8292a9;font-size:12px;line-height:1.7; }
    .stProgress > div > div { background:linear-gradient(90deg,#2fcbd5,#7b87ff); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><div class="eyebrow">AURORA · DRIVER ATTENTION SYSTEM</div>'
    '<div class="hero-title">See the signals. Stay alert.</div>'
    '<div class="hero-copy">A polished computer-vision demo for exploring eye closure, head direction, and yawning. '
    'Analyze a camera snapshot, image, or sampled video with on-screen indicators and an optional browser alarm.</div></div>',
    unsafe_allow_html=True,
)

st.warning("Educational prototype only — not a certified safety device. Never rely on this app to decide whether it is safe to drive.")

if "face_mesh" not in st.session_state:
    st.session_state.face_mesh = initialize_face_mesh()
if "logs" not in st.session_state:
    st.session_state.logs = []


def analyze_frame(frame: np.ndarray, detector: FatigueDetector) -> tuple[np.ndarray, dict]:
    """Downscale large frames, run face landmarks, and draw a lightweight overlay."""
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest > 960:
        scale = 960 / longest
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
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
        color = (90, 225, 175) if metrics["status"] == "ATTENTIVE" else (75, 100, 255)
        cv2.rectangle(processed, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        for index in (33, 133, 362, 263, 61, 291, 1):
            point = landmarks[index]
            cv2.circle(processed, (int(point.x * width), int(point.y * height)), 3, (85, 225, 245), -1, cv2.LINE_AA)

    return cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), metrics


def log_result(source: str, frame_number: int | str, metrics: dict) -> None:
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


def show_result(metrics: dict) -> None:
    status = metrics["status"]
    if status == "DROWSY":
        st.error("🚨 Drowsiness indicators detected")
    elif status in ("DISTRACTED", "NO FACE DETECTED"):
        st.warning(f"⚠️ {status}")
    else:
        st.success("✓ No drowsiness indicators detected in this sample")
    cols = st.columns(5)
    cols[0].metric("Fatigue index", f'{metrics["fatigue_index"]:.2f}')
    cols[1].metric("Eye ratio · EAR", f'{metrics["ear"]:.3f}')
    cols[2].metric("Mouth ratio · MAR", f'{metrics["mar"]:.3f}')
    cols[3].metric("Blinks", metrics["blinks"])
    cols[4].metric("Yawns", metrics["yawns"])
    st.markdown(f'<div class="status-line">◉ &nbsp; GAZE DIRECTION &nbsp; / &nbsp; {metrics["gaze"]}</div>', unsafe_allow_html=True)


def alarm_panel(alarm_needed: bool) -> None:
    """Browser-generated two-tone alert; user click is required to satisfy autoplay rules."""
    tone = """
    <div style="font:14px Arial;color:#dce7f8;background:#111b2b;border:1px solid #ffffff20;border-radius:12px;padding:12px 14px">
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
        <span id="label">🔊 Optional browser alarm · adjust volume</span>
        <input id="vol" type="range" min="10" max="100" value="90" style="accent-color:#ff637d">
        <button id="start" style="border:0;border-radius:8px;padding:8px 12px;background:#ff637d;color:#180812;font-weight:700;cursor:pointer">Enable / test alarm</button>
        <button id="stop" style="border:1px solid #ffffff35;border-radius:8px;padding:8px 12px;background:#202d42;color:#e5edf8;cursor:pointer">Stop</button>
      </div>
      <div style="color:#91a1bb;font-size:11px;margin-top:8px">The browser and device control actual speaker loudness. Keep this tab open; browser autoplay policies require a click before sound can play.</div>
    </div>
    <script>
    (() => {
      let ctx = null, timer = null, osc = null, gain = null, phase = 0;
      const volume = () => Number(document.getElementById('vol').value) / 100;
      const stop = () => { if (timer) clearInterval(timer); timer = null; if (osc) { try { osc.stop(); } catch(e) {} osc = null; } if (ctx) { ctx.close(); ctx = null; } document.getElementById('label').textContent = '🔊 Optional browser alarm · stopped'; };
      const start = async () => {
        stop(); ctx = new (window.AudioContext || window.webkitAudioContext)(); await ctx.resume();
        osc = ctx.createOscillator(); gain = ctx.createGain(); osc.type = 'square'; osc.frequency.value = 880; gain.gain.value = 0;
        osc.connect(gain); gain.connect(ctx.destination); osc.start(); phase = 0;
        timer = setInterval(() => { if (!ctx || !gain) return; const active = phase % 8 < 3 || (phase % 8 >= 4 && phase % 8 < 7); gain.gain.setTargetAtTime(active ? Math.min(0.95, volume()) : 0, ctx.currentTime, 0.015); osc.frequency.setTargetAtTime(phase % 8 < 4 ? 880 : 660, ctx.currentTime, 0.02); phase++; }, 125);
        document.getElementById('label').textContent = '🔊 Alarm enabled · adjust volume or stop';
      };
      document.getElementById('start').addEventListener('click', start);
      document.getElementById('stop').addEventListener('click', stop);
      document.getElementById('vol').addEventListener('input', () => { if (gain && ctx) gain.gain.setTargetAtTime(Math.min(0.95, volume()), ctx.currentTime, 0.02); });
      window.addEventListener('pagehide', stop);
    })();
    </script>
    """
    components.html(tone, height=105, scrolling=False)
    if alarm_needed:
        st.info("A drowsiness-related signal was found. Click **Enable / test alarm** above if you want to hear the browser alert.")


with st.sidebar:
    st.markdown("### ◈ CONTROL DECK")
    st.caption("Choose a source to begin analysis.")
    mode = st.radio("Input source", ("Camera snapshot", "Upload image", "Upload video"), label_visibility="collapsed")
    st.divider()
    st.markdown("#### Analysis profile")
    st.caption("Large images are resized for faster inference. Video is sampled to limit CPU use.")
    if mode == "Upload video":
        sample_every = st.slider("Sample every N frames", 1, 30, 5)
        max_samples = st.slider("Maximum sampled frames", 30, 300, 120, step=30)
    else:
        sample_every, max_samples = 5, 120

st.markdown('<div class="section-label">01 / Select media</div>', unsafe_allow_html=True)
if mode == "Camera snapshot":
    st.info("Streamlit Cloud uses a browser camera snapshot, not a continuous server-side webcam stream. For temporal analysis, upload a video.")
    image_file = st.camera_input("Capture a face snapshot")
    if image_file and st.button("Analyze snapshot", type="primary"):
        image = cv2.imdecode(np.frombuffer(image_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            st.error("Could not read the captured image. Please take another snapshot.")
        else:
            annotated, metrics = analyze_frame(image, FatigueDetector())
            st.markdown('<div class="section-label">02 / Analysis</div>', unsafe_allow_html=True)
            left, right = st.columns([1.3, 1])
            with left:
                st.image(annotated, caption="Face landmarks · analyzed snapshot", use_container_width=True)
            with right:
                show_result(metrics)
            alarm_panel(metrics["status"] in ("DROWSY", "DISTRACTED"))
            log_result("Camera snapshot", "-", metrics)

elif mode == "Upload image":
    image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if image_file and st.button("Analyze image", type="primary"):
        image = cv2.imdecode(np.frombuffer(image_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            st.error("Could not read this image. Try a JPG or PNG file.")
        else:
            annotated, metrics = analyze_frame(image, FatigueDetector())
            st.markdown('<div class="section-label">02 / Analysis</div>', unsafe_allow_html=True)
            left, right = st.columns([1.3, 1])
            with left:
                st.image(annotated, caption="Face landmarks · analyzed image", use_container_width=True)
            with right:
                show_result(metrics)
            alarm_panel(metrics["status"] in ("DROWSY", "DISTRACTED"))
            log_result(image_file.name, "-", metrics)

else:
    video_file = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv"])
    if video_file and st.button("Analyze video", type="primary"):
        video_path = None
        cap = None
        try:
            suffix = Path(video_file.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_video:
                temp_video.write(video_file.getvalue())
                video_path = temp_video.name
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Could not open this video. Try a common MP4/H.264 file.")
            else:
                detector = FatigueDetector()
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress = st.progress(0, text="Preparing video analysis…")
                preview = st.empty()
                samples = []
                frame_index = 0
                sampled_count = 0
                while sampled_count < max_samples:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_index % sample_every == 0:
                        annotated, metrics = analyze_frame(frame, detector)
                        samples.append(metrics)
                        log_result(video_file.name, frame_index, metrics)
                        sampled_count += 1
                        # Update preview/progress periodically to avoid slow UI rerenders.
                        if sampled_count == 1 or sampled_count % 5 == 0:
                            preview.image(annotated, caption=f"Sampled frame {frame_index}", use_container_width=True)
                        if frame_count > 0 and (sampled_count == 1 or sampled_count % 5 == 0):
                            progress.progress(min(frame_index / frame_count, 1.0), text=f"Analyzed {sampled_count} frames")
                    frame_index += 1
                progress.empty()
                if not samples:
                    st.warning("No frames could be decoded from this video.")
                else:
                    drowsy = sum(item["status"] == "DROWSY" for item in samples)
                    distracted = sum(item["status"] in ("DISTRACTED", "NO FACE DETECTED") for item in samples)
                    avg_fatigue = float(np.mean([item["fatigue_index"] for item in samples]))
                    st.markdown('<div class="section-label">02 / Session summary</div>', unsafe_allow_html=True)
                    cols = st.columns(4)
                    cols[0].metric("Frames sampled", len(samples))
                    cols[1].metric("Drowsy samples", drowsy)
                    cols[2].metric("Distracted / no face", distracted)
                    cols[3].metric("Average fatigue", f"{avg_fatigue:.2f}")
                    alarm_panel(drowsy > 0 or any(item["status"] == "DISTRACTED" for item in samples))
                    st.caption("Video is analyzed after upload; this is not a real-time warning system. Sample estimates depend on lighting, face visibility, and camera position.")
        except Exception as exc:
            st.error(f"Video analysis failed: {exc}")
        finally:
            if cap is not None:
                cap.release()
            if video_path and Path(video_path).exists():
                Path(video_path).unlink(missing_ok=True)

if st.session_state.logs:
    st.divider()
    st.markdown('<div class="section-label">03 / Session data</div>', unsafe_allow_html=True)
    history = pd.DataFrame(st.session_state.logs)
    st.dataframe(history.tail(100), use_container_width=True, hide_index=True)
    left, right = st.columns([1, 4])
    with left:
        st.download_button("Download CSV report", data=history.to_csv(index=False).encode("utf-8"), file_name="aurora_driver_report.csv", mime="text/csv")
    with right:
        if st.button("Clear session data"):
            st.session_state.logs = []
            st.rerun()

with st.expander("Deployment, sound, and privacy"):
    st.markdown(
        "- The alarm is generated in the visitor's browser, not on the Streamlit server. Click **Enable / test alarm** to satisfy browser audio permissions; set the slider and device volume to the desired level. Browser/OS settings ultimately control loudness.\n"
        "- Since Cloud processes uploaded video after submission, an alert is not continuous or real time. The camera option captures a snapshot only.\n"
        "- Images are processed in memory. Uploaded videos use a temporary file that is deleted after decoding. CSV reports are downloaded rather than saved permanently to Cloud.\n"
        "- This demo can be inaccurate and must not be used to make driving or medical decisions."
    )
