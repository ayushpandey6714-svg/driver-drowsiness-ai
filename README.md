# AURORA | Driver Attention System

A dark, polished Streamlit computer-vision demo that estimates eye closure, gaze direction, and yawning using MediaPipe Face Mesh. It supports browser camera snapshots, image uploads, sampled video analysis, session metrics, CSV export, and an optional browser-generated alert tone.

> **Safety:** Educational prototype only. It is not a certified safety device, does not provide real-time driver monitoring on Streamlit Cloud, and must not be relied on to determine whether it is safe to drive.

## Run locally

Use Python 3.11:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from this repository.
3. Choose branch `fix/streamlit-cloud-deployment` and entry point `app.py` (or merge this branch and choose `main`).
4. Deploy; the app reads `requirements.txt` and `runtime.txt`.

Cloud does not have access to a visitor's computer webcam. The app uses Streamlit's browser snapshot camera control or uploaded media. Uploaded video is analyzed after submission and sampled to manage CPU use; it is not a real-time alarm system.

## Browser alarm

The optional two-tone alarm is synthesized by the browser, so it can play through the visitor's speakers rather than the server's audio device. Browsers block sound autoplay in many cases: click **Enable / test alarm** to grant audio permission. The in-app volume control sets the web-audio gain, but actual loudness is also controlled by the browser and operating-system volume. Keep the page open. The sound is a convenience for this demo only, not a safety alert.

## Privacy and limitations

- Images are processed in memory. Video uses a temporary file while decoding and removes it afterward.
- Reports are downloaded as CSV; Streamlit Cloud storage is ephemeral.
- Results are heuristic and depend on lighting, face visibility, camera position, and video sampling.

## Files

- `app.py` — Streamlit interface, media analysis, visualization, alarm UI, and CSV export.
- `detector.py` — MediaPipe Face Mesh setup and frame preprocessing.
- `fatigue.py` — EAR, MAR, gaze, blink, yawn, and fatigue heuristics.
