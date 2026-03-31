"""
config.py — Configuration settings for Emotion Detector
=========================================================
Change these values to tune the application behavior
without touching the main code.
"""

# ── Camera ────────────────────────────────────
CAMERA_INDEX = 0          # 0 = built-in webcam, 1 = external
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ── Analysis ──────────────────────────────────
ANALYSIS_INTERVAL = 0.3   # seconds between DeepFace calls
DETECTOR_BACKEND  = "opencv"   # opencv | ssd | mtcnn | retinaface

# ── Display ───────────────────────────────────
SHOW_FPS         = True
SHOW_PROB_BARS   = True
SHOW_INSTRUCTIONS = True

# ── Snapshot ──────────────────────────────────
SNAPSHOT_PREFIX = "snapshot"  # saved as snapshot_001.jpg, etc.
