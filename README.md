# Driver Drowsiness Monitor

A small Streamlit computer-vision demo that estimates eye closure, head direction, and yawning using MediaPipe Face Mesh and OpenCV. It supports browser camera snapshots, image uploads, and sampled video analysis, with a downloadable CSV history.

> **Safety:** This is an educational prototype, not a certified driver-monitoring or emergency-alert system. Do not rely on it to determine whether it is safe to drive.

## Run locally

Use Python 3.11, then install dependencies and start Streamlit:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. Allow browser camera permission to take a snapshot, or upload an image/video. Video analysis samples frames to keep processing manageable; use the controls to adjust the sampling interval and maximum number of frames.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from the repository.
3. Set the branch to `main` and the app entry point to `app.py` (or use the branch containing your changes until it is merged).
4. Deploy. The app uses `requirements.txt` and `runtime.txt` to install dependencies and select Python 3.11.

Streamlit Cloud runs in a remote server and cannot access a visitor's webcam directly. The app uses Streamlit's browser-based camera snapshot widget; upload a video if you need temporal, frame-by-frame analysis. It does not play an alarm through the visitor's speakers.

## Privacy and limitations

- Camera snapshots and uploaded images are analyzed in memory. Uploaded videos are written to a temporary file during decoding and removed afterward.
- Video fatigue indicators are heuristic estimates and depend on face visibility, lighting, camera angle, and sampling rate.
- Streamlit Cloud's filesystem is temporary. Reports are provided as a CSV download rather than being saved permanently on the server.

## Project files

- `app.py` — Streamlit interface, image/video input, visual results, and CSV export.
- `detector.py` — MediaPipe Face Mesh setup and frame preprocessing.
- `fatigue.py` — EAR, MAR, gaze, blink, yawn, and fatigue heuristics.
