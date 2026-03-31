# 😊 Real-Time Facial Emotion Detector

A Computer Vision project that detects human faces through a live webcam feed and classifies facial emotions in real time using DeepFace and OpenCV.

---

## 🧠 What It Does

- Opens your webcam and processes each frame
- Detects faces using OpenCV's built-in detector
- Classifies emotions: **Happy, Sad, Angry, Surprised, Fear, Disgust, Neutral**
- Displays a colored bounding box, dominant emotion label, and a live probability bar chart
- Shows real-time FPS counter
- Allows saving snapshots with the `S` key

---

## 🎯 Problem Statement

Emotion recognition is increasingly relevant in areas like mental wellness apps, online education, and human-computer interaction. This project addresses the challenge of **real-time, accessible emotion analysis** using only a standard webcam — no expensive hardware or large datasets required.

---

## 🖥️ Demo

```
[Webcam Window]
┌───────────────────────────────┐
│ Emotion Scores    FPS: 12.3   │
│ happy   ██████  78.2%         │
│ neutral ██      14.1%         │
│ sad            5.5%           │
│                               │
│   ┌─────────────┐             │
│   │ HAPPY       │  ← label    │
│   │  (face)     │  ← box      │
│   └─────────────┘             │
│ Press Q to quit | S to save   │
└───────────────────────────────┘
```

---

## 📦 Requirements

- Python 3.8 or higher
- A working webcam

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/emotion-detector.git
cd emotion-detector
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** DeepFace will automatically download a pre-trained model (~100 MB) on the first run. This only happens once.

---

## ▶️ Running the Project

```bash
python emotion_detector.py
```

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--camera` | `0` | Camera index (use `1` for external webcam) |
| `--interval` | `0.3` | Seconds between emotion analysis calls |

**Example with external webcam and faster analysis:**
```bash
python emotion_detector.py --camera 1 --interval 0.2
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `Q` | Quit the application |
| `S` | Save a snapshot (saved as `snapshot_001.jpg`, etc.) |

---

## 📁 Project Structure

```
emotion-detector/
├── emotion_detector.py   # Main application script
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🔬 How It Works

1. **Frame Capture** — OpenCV reads frames from the webcam continuously.
2. **Throttled Analysis** — Every 0.3 seconds (configurable), the frame is passed to DeepFace.
3. **Face Detection** — DeepFace uses OpenCV's Haar Cascade internally to locate faces.
4. **Emotion Classification** — A pre-trained mini-Xception CNN model predicts probabilities across 7 emotion classes.
5. **Overlay Rendering** — Results are drawn back onto the live frame before display.

### Model Details

| Property | Value |
|---|---|
| Library | DeepFace |
| Backend Model | mini-Xception (FER-2013 trained) |
| Emotion Classes | 7 (happy, sad, angry, surprise, fear, disgust, neutral) |
| Face Detector | OpenCV Haar Cascade |

---

## ⚡ Performance Tips

- If the webcam feed is laggy, increase `--interval` (e.g., `0.5`)
- For faster performance on a GPU machine, DeepFace will automatically use CUDA if TensorFlow detects it
- Keep good lighting facing your face for best detection accuracy

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot open camera 0` | Try `--camera 1` or check webcam permissions |
| `No face detected` | Ensure good lighting; face the camera directly |
| Slow FPS | Increase `--interval` value |
| DeepFace import error | Run `pip install deepface tf-keras` |

---

## 📚 Course

**Computer Vision** — Bring Your Own Project (BYOP) Submission  
VIT Bhopal University

---

## 📄 License

MIT License — free to use and modify.

Project by: Kopal Khare | VIT Bhopal University | Computer Vision 2026
