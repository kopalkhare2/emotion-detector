"""
utils.py — Utility functions for the Emotion Detector
======================================================
Helper functions for color mapping, text rendering,
frame annotation, and result formatting.
"""

import cv2
import numpy as np

# ──────────────────────────────────────────────
# Emotion → BGR color mapping
# ──────────────────────────────────────────────
EMOTION_COLORS = {
    "happy":     (0, 215, 255),
    "sad":       (255, 100, 50),
    "angry":     (0, 0, 220),
    "surprise":  (0, 165, 255),
    "fear":      (130, 0, 200),
    "disgust":   (0, 150, 0),
    "neutral":   (180, 180, 180),
}


def get_color(emotion: str) -> tuple:
    """Return BGR color for a given emotion string."""
    return EMOTION_COLORS.get(emotion.lower(), (255, 255, 255))


def draw_rounded_rect(frame, x1, y1, x2, y2, color, thickness=2, r=10):
    """Draw a rounded rectangle on the frame."""
    cv2.line(frame, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(frame, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(frame, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(frame, (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(frame, (x1 + r, y2 - r), (r, r),  90, 0, 90, color, thickness)
    cv2.ellipse(frame, (x2 - r, y2 - r), (r, r),   0, 0, 90, color, thickness)


def draw_emotion_label(frame, emotion: str, x: int, y: int):
    """Draw filled label box with emotion text above the face."""
    color = get_color(emotion)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.8, 2
    (tw, th), _ = cv2.getTextSize(emotion.upper(), font, scale, thickness)
    cv2.rectangle(frame, (x, y - th - 14), (x + tw + 12, y), color, -1)
    cv2.putText(frame, emotion.upper(), (x + 6, y - 6),
                font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def draw_probability_bars(frame, emotions: dict, x=20, y=60):
    """Draw a vertical bar chart of emotion probabilities."""
    bar_w, bar_h = 130, 14
    font = cv2.FONT_HERSHEY_SIMPLEX
    sorted_emos = sorted(emotions.items(), key=lambda e: e[1], reverse=True)
    cv2.putText(frame, "Emotion Scores", (x, y - 12),
                font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    for i, (emo, prob) in enumerate(sorted_emos):
        cy = y + i * (bar_h + 8)
        color = get_color(emo)
        cv2.rectangle(frame, (x, cy), (x + bar_w, cy + bar_h), (60, 60, 60), -1)
        filled = int(bar_w * prob / 100)
        cv2.rectangle(frame, (x, cy), (x + filled, cy + bar_h), color, -1)
        cv2.putText(frame, f"{emo[:7]:<7} {prob:4.1f}%",
                    (x + bar_w + 6, cy + bar_h - 2),
                    font, 0.4, (220, 220, 220), 1, cv2.LINE_AA)


def draw_fps(frame, fps: float):
    """Overlay FPS counter on top-right of frame."""
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (frame.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1, cv2.LINE_AA)


def draw_no_face(frame):
    """Show 'No face detected' message in the center."""
    cv2.putText(frame, "No face detected", (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 80, 220), 2, cv2.LINE_AA)


def draw_instructions(frame):
    """Draw keyboard shortcut hint at the bottom of the frame."""
    h = frame.shape[0]
    cv2.putText(frame, "Q = Quit  |  S = Save Snapshot",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (180, 180, 180), 1, cv2.LINE_AA)


def format_results(results: list) -> list:
    """
    Normalize DeepFace output into a consistent list of dicts:
    [{ 'emotion': str, 'emotions': dict, 'region': dict }]
    """
    formatted = []
    for r in results:
        formatted.append({
            "emotion": r.get("dominant_emotion", "neutral"),
            "emotions": r.get("emotion", {}),
            "region": r.get("region", {"x": 0, "y": 0, "w": 0, "h": 0}),
        })
    return formatted
