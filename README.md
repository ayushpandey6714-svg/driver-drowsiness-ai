# 🚗 AI Driver Monitoring & Safety System

An advanced real-time computer vision application built with **Streamlit**, **OpenCV**, and **MediaPipe**[cite: 3] to monitor driver attentiveness, detect drowsiness, track head pose / gaze[cite: 4, 5], and prevent accidents by triggering smart audio alarms.

---

## 🌟 Key Features

* **Real-time Drowsiness Detection:** Calculates **Eye Aspect Ratio (EAR)** to detect prolonged eye closure/sleeping[cite: 2, 4].
* **Distraction & Gaze Tracking:** Monitors head movements and gaze directions (Left, Right, Up, Down) to check if the driver is looking away from the road[cite: 4, 5].
* **Yawn Detection:** Uses **Mouth Aspect Ratio (MAR)** to track frequent yawning and fatigue levels[cite: 4, 9].
* **Dynamic Alert System:** Triggers automated audio alarms (`pygame` / custom sound) when drowsiness or distraction exceeds safe thresholds[cite: 1, 8].
* **Session Analytics & Trip Reports:** Generates live metrics, tracks performance, auto-saves CSV reports locally, and allows instant downloads[cite: 1, 6].
* **Interactive UI:** Built using Streamlit with dark-mode styling and dynamic status cards.

---

## 🛠️ Tech Stack

* **Python**[cite: 7]
* **Streamlit** (Web Framework)[cite: 1, 7]
* **OpenCV** (Video Capture & Image Processing)[cite: 3, 7]
* **MediaPipe** (Face Mesh & Landmark Extraction)[cite: 3, 7]
* **NumPy & Pandas** (Data Processing & CSV Logging)[cite: 1, 7]
* **Pygame** (Alarm Audio Management)

---

## 📁 Project Structure

```text
├── app.py              # Main Streamlit dashboard application
├── detector.py         # MediaPipe face mesh & frame processing[cite: 3]
├── fatigue.py          # Fatigue index, EAR, MAR & gaze algorithms[cite: 4]
├── blink.py            # Eye blink detection logic[cite: 2]
├── yawn.py             # Yawn detection logic[cite: 9]
├── head_pose.py        # Head pose estimation via solvePnP[cite: 5]
├── utils.py            # Audio warnings & visual overlays[cite: 8]
├── report.py           # Session data logger & CSV exporter[cite: 6]
├── alarm.wav           # Warning sound effect
└── requirements.txt    # Required python dependencies[cite: 7]
