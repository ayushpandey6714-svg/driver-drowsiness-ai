# AURORA | Driver Attention System

A dark Streamlit computer-vision demo for estimating eye closure, gaze direction, and yawning, with optional YOLOv8n object detection, browser-generated alarm UI, sampled video analysis, session metrics, and CSV export.

> **Safety:** Educational prototype only. It is not a certified safety device, does not provide real-time driver monitoring on Streamlit Cloud, and must not be relied on to decide whether it is safe to drive.

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

1. Push this repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from this repository.
3. Choose branch `fix/streamlit-cloud-deployment` and entry point `app.py` (or merge this branch and choose `main`).
4. Deploy. The first dependency installation and first object-detector use may take longer than subsequent launches.

## Object detection

Enable **YOLOv8n object detection** in the sidebar to detect common objects from the COCO dataset (for example, people, phones, cars, and bottles) and draw boxes/labels. It is optional and disabled by default to keep ordinary image analysis faster. On first use, Ultralytics downloads the small `yolov8n.pt` model weights; detection runs on CPU and can slow video processing. This general-purpose model is not a dedicated driver phone-use detector.

## Browser alarm and Cloud limitations

The optional alarm is synthesized by the visitor's browser, not the server. Click **Enable / test alarm** to allow browser audio. The gain control is capped for comfort; device and browser volume controls determine actual loudness. Streamlit Cloud cannot access a visitor's webcam. Its camera widget captures a snapshot, and uploaded videos are processed after submission, not continuously in real time.

## Privacy and limitations

- Images are processed in memory. Video uses a temporary file while decoding and removes it afterward.
- Reports are downloaded as CSV; Streamlit Cloud storage is ephemeral.
- Results are heuristic and depend on lighting, face visibility, camera position, object size, and video sampling.

## Files

- `app.py` — Streamlit interface, media analysis, visualization, optional object detection/alarm, and CSV export.
- `detector.py` — MediaPipe Face Mesh setup and frame preprocessing.
- `fatigue.py` — EAR, MAR, gaze, blink, yawn, and fatigue heuristics.
