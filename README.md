# SignBridge — Real-Time ASL Recognition

Real-time American Sign Language alphabet recognition using MobileNetV2 and MediaPipe, built to improve communication accessibility for the deaf and hard-of-hearing community.

---

## Demo

▶️ [Watch demo](https://youtu.be/8S9psWXlqLg)

---

## How it works

Each video frame is processed through MediaPipe Hands to extract 21 3D hand landmarks. These 63 values are reshaped into a 7×9 pseudo-image and resized to 32×32, then passed to a fine-tuned MobileNetV2 for classification across 26 ASL alphabet classes.

A debounce system prevents duplicate letter predictions — a letter is only appended to the current word if the same sign is held for a configurable interval, and double letters require a separate hold window. Predictions above a 0.7 confidence threshold are accepted.

The app runs as a Flask + SocketIO server, streaming processed frames and predictions to the browser in real time via WebSockets.

---

## Features

- Real-time ASL alphabet recognition (A–Z) at ~30 FPS
- MediaPipe hand landmark extraction — model never sees raw pixels, only structured keypoints
- Debounce and double-letter detection for accurate word building
- Word reset, backspace, space, and word completion controls
- Spanish translation via the `translate` library (hardcoded to Spanish, configurable in `slr.py` via `self.target_language`)

---

## Stack

- **Model:** MobileNetV2 fine-tuned on ASL gesture landmarks
- **Hand detection:** MediaPipe
- **Backend:** Flask + Flask-SocketIO + Eventlet
- **Frontend:** HTML/CSS/JS with WebSocket client
- **CV:** OpenCV

---

## Model

| Detail | Value |
|---|---|
| Architecture | MobileNetV2 (fine-tuned) |
| Input | 32×32 landmark pseudo-image (1 channel) |
| Classes | 26 (A–Z) |
| Accuracy | 98.7% on test set |
| Confidence threshold | 0.70 |

MobileNetV2 was chosen for its depthwise separable convolutions — lightweight enough for real-time inference on CPU while maintaining high accuracy on the landmark representation.

---

## Run locally

```bash
git clone https://github.com/kshitideshpande/SignBridge
cd SignBridge
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000` in your browser. Allow camera access when prompted.

**Controls:**
| Key | Action |
|---|---|
| Start | Begin camera + recognition |
| R | Reset current word |
| C | Complete word |
| S | Add space |
| T | Translate current word to Spanish |
| Backspace | Remove last letter |
| Q | Stop camera |

---

## Publication

**SignBridge: Real-Time Sign Language Recognition and Translation**
National Conference on Applications of Artificial Intelligence in Engineering, April 2025

📄 [Read the paper](https://drive.google.com/file/d/1DYhcFpoOWs_OwG5knI70VX8p1UWz2Sdw/view)
