# utils.py
import cv2
import time
import os
import pygame
import numpy as np

# Initialize Pygame Mixer for sound
try:
    pygame.mixer.init()
except Exception:
    pass

_last_warning_time = 0.0
_last_danger_time = 0.0

AUDIO_FILE_PATH = "alarm.wav"

def play_sound(file_path):
    """Play custom alarm audio safely."""
    try:
        if os.path.exists(file_path):
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
    except Exception:
        pass

def stop_sound():
    """Immediately stop audio when status returns to Normal."""
    try:
        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    except Exception:
        pass

def play_warning_sound():
    """Play warning audio with 2.0 second cooldown."""
    global _last_warning_time
    current_time = time.time()
    if current_time - _last_warning_time > 2.0:
        play_sound(AUDIO_FILE_PATH)
        _last_warning_time = current_time

def play_danger_sound():
    """Play danger alarm sound with 1.0 second cooldown."""
    global _last_danger_time
    current_time = time.time()
    if current_time - _last_danger_time > 1.0:
        play_sound(AUDIO_FILE_PATH)
        _last_danger_time = current_time

def draw_overlay(frame: np.ndarray, status: str, fps: float) -> np.ndarray:
    """Draw border and FPS on frame."""
    height, width, _ = frame.shape

    if status == "Danger":
        border_color = (0, 0, 255)
    elif status == "Warning":
        border_color = (0, 255, 255)
    else:
        border_color = (0, 255, 0)

    if status in ["Warning", "Danger"]:
        cv2.rectangle(frame, (0, 0), (width-1, height-1), border_color, 6)

    # Top-right FPS Badge
    cv2.rectangle(frame, (width - 130, 10), (width - 10, 45), (15, 23, 42), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (width - 120, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame