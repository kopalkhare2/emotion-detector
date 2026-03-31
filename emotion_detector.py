"""
Real-Time Facial Emotion Detector
===================================
Uses OpenCV for face detection and DeepFace for emotion classification.
Captures webcam feed and overlays detected emotions in real time.

Author: [Your Name]
Course: Computer Vision
"""

import cv2
import time
import argparse
import numpy as np

try:
    from deepface import DeepFace
except ImportError:
    print("[ERROR] DeepFace not installed. Run: pip install deepface")
    exit(1)


# ──────────────────────────────────────────────
# Emotion color palette (BGR for OpenCV)
# ──────────────────────────────────────────────
EMOTION_COLORS = {
    "happy":     (0, 215, 255),   # Gold
    "sad":       (255, 100, 50),  # Blue-ish
    "angry":     (0, 0, 220),     # Red
    "surprise":  (0, 165, 255),   # Orange
    "fear":      (130, 0, 200),   # Purple
    "disgust":   (0, 150, 0),     # Green
    "neutral":   (180, 180, 180), # Gray
}

DEFAULT_COLOR = (255, 255, 255)


def get_emotion_color(emotion: str):
    return EMOTION_COLORS.get(emotion.lower(), DEFAULT_COLOR)


def draw_label(frame, text, x, y, color):
    """Draw a filled background label above the face box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.8, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    # Background rectangle
    cv2.rectangle(frame, (x, y - th - 12), (x + tw + 10, y), color, -1)
    # Text
    cv2.putText(frame, text, (x + 5, y - 6), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def draw_bar_chart(frame, emotions: dict, x_start=20, y_start=60, bar_width=120, bar_height=14):
    """Draw a mini emotion probability bar chart on the frame."""
    sorted_emotions = sorted(emotions.items(), key=lambda e: e[1], reverse=True)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, "Emotion Scores", (x_start, y_start - 10),
                font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    for i, (emo, prob) in enumerate(sorted_emotions):
        y = y_start + i * (bar_height + 8)
        color = get_emotion_color(emo)
        # Background bar
        cv2.rectangle(frame, (x_start, y), (x_start + bar_width, y + bar_height), (60, 60, 60), -1)
        # Filled bar proportional to probability
        filled = int(bar_width * prob / 100)
        cv2.rectangle(frame, (x_start, y), (x_start + filled, y + bar_height), color, -1)
        # Label
        label = f"{emo[:7]:<7} {prob:4.1f}%"
        cv2.putText(frame, label, (x_start + bar_width + 6, y + bar_height - 2),
                    font, 0.4, (220, 220, 220), 1, cv2.LINE_AA)


def draw_fps(frame, fps: float):
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1, cv2.LINE_AA)


def draw_instructions(frame):
    h = frame.shape[0]
    cv2.putText(frame, "Press Q to quit | S to save snapshot",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def analyze_frame(frame, enforce_detection=False):
    """
    Run DeepFace emotion analysis on a frame.
    Returns list of result dicts or empty list.
    """
    try:
        results = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=enforce_detection,
            detector_backend="opencv",
            silent=True,
        )
        return results if isinstance(results, list) else [results]
    except Exception:
        return []


def run_detector(camera_index: int = 0, analysis_interval: float = 0.3):
    """
    Main loop: capture → analyze → overlay → display.
    analysis_interval: seconds between DeepFace calls (to avoid lag).
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] Camera opened. Press Q to quit, S to save snapshot.")

    last_analysis_time = 0.0
    cached_results = []
    frame_count = 0
    fps_timer = time.time()
    fps = 0.0
    snapshot_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed.")
            break

        frame_count += 1
        now = time.time()

        # ── FPS calculation ──────────────────────
        if now - fps_timer >= 1.0:
            fps = frame_count / (now - fps_timer)
            frame_count = 0
            fps_timer = now

        # ── Throttled emotion analysis ───────────
        if now - last_analysis_time >= analysis_interval:
            cached_results = analyze_frame(frame)
            last_analysis_time = now

        # ── Draw results for each face ───────────
        for result in cached_results:
            region = result.get("region", {})
            x, y = region.get("x", 0), region.get("y", 0)
            w, h = region.get("w", 0), region.get("h", 0)

            dominant = result.get("dominant_emotion", "?")
            all_emotions = result.get("emotion", {})
            color = get_emotion_color(dominant)

            # Face bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # Label above box
            draw_label(frame, dominant.upper(), x, y, color)

        # Draw emotion bar chart (from first detected face)
        if cached_results:
            emotions = cached_results[0].get("emotion", {})
            if emotions:
                draw_bar_chart(frame, emotions)

        draw_fps(frame, fps)
        draw_instructions(frame)

        # ── No face fallback ─────────────────────
        if not cached_results:
            cv2.putText(frame, "No face detected", (200, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 220), 2, cv2.LINE_AA)

        cv2.imshow("Real-Time Emotion Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Quitting.")
            break
        elif key == ord("s"):
            snapshot_counter += 1
            fname = f"snapshot_{snapshot_counter:03d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Snapshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Facial Emotion Detector")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--interval", type=float, default=0.3,
                        help="Seconds between emotion analysis calls (default: 0.3)")
    args = parser.parse_args()

    run_detector(camera_index=args.camera, analysis_interval=args.interval)
