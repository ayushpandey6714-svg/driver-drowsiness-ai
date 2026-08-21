# blink.py
# Eye blink detection and Eye Aspect Ratio (EAR) calculation

"""
Mathematical Formula - Eye Aspect Ratio (EAR):
----------------------------------------------
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

Where:
  p1, p4 = Left and right eye corner points (horizontal landmarks)
  p2, p6 = Upper and lower eyelid points (vertical landmarks)
  p3, p5 = Upper and lower eyelid points (vertical landmarks)

Interpretation:
  - When eye is open: EAR ~ 0.3
  - When eye is closed: EAR ~ 0.1
  - A blink is detected when EAR falls below a threshold for consecutive frames

MediaPipe Face Mesh landmark indices for the left eye:
  - Left eye outer corner (p1): landmark 33
  - Left eye upper lid (p2): landmark 159
  - Left eye lower lid (p6): landmark 145
  - Left eye inner corner (p4): landmark 133

MediaPipe Face Mesh landmark indices for the right eye:
  - Right eye outer corner (p4): landmark 263
  - Right eye upper lid (p3): landmark 386
  - Right eye lower lid (p5): landmark 374
  - Right eye inner corner (p1): landmark 362
"""

import cv2
import numpy as np
from collections import deque
from typing import List, Tuple

# EAR threshold for blink detection
EAR_THRESHOLD = 0.25

# Number of consecutive frames EAR must be below threshold to confirm a blink
BLINK_CONSECUTIVE_FRAMES = 3

# Number of consecutive frames EAR must be below threshold for drowsiness detection
DROWSINESS_THRESHOLD_FRAMES = 30

# Landmark indices for left eye (MediaPipe Face Mesh)
LEFT_EYE_INDICES = [33, 160, 159, 158, 157, 173]

# Landmark indices for right eye (MediaPipe Face Mesh)
RIGHT_EYE_INDICES = [362, 385, 386, 387, 388, 390]


class BlinkDetector:
    """
    Detects eye blinks using Eye Aspect Ratio (EAR) calculation.

    EAR Formula:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Where p1-p6 are specific eye landmarks:
        p1, p4 = eye corners (horizontal measurement)
        p2, p6 = upper and lower eyelid points (vertical measurement)
        p3, p5 = inner upper and lower eyelid points (vertical measurement)
    """

    def __init__(self):
        """Initialize blink detector with counters and history."""
        self.blink_frame_counter = 0
        self.total_blinks = 0
        self.ear_history = deque(maxlen=DROWSINESS_THRESHOLD_FRAMES)
        self.last_blink_time = 0

    def calculate_ear(self, landmarks: List[Tuple[float, float]],
                      eye_indices: List[int]) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for a single eye.

        Formula:
            EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        Args:
            landmarks: List of (x, y) tuples for all face landmarks (normalized [0,1])
            eye_indices: List of 6 indices corresponding to the eye landmarks

        Returns:
            EAR value (float between 0 and 1)
        """
        # Extract the 6 eye landmarks
        p1 = landmarks[eye_indices[0]]  # Left corner
        p2 = landmarks[eye_indices[1]]  # Upper
        p3 = landmarks[eye_indices[2]]  # Upper inner
        p4 = landmarks[eye_indices[3]]  # Right corner (or inner)
        p5 = landmarks[eye_indices[4]]  # Lower inner
        p6 = landmarks[eye_indices[5]]  # Lower

        # Calculate vertical distances using Euclidean distance formula
        # Euclidean distance: sqrt((x2-x1)^2 + (y2-y1)^2)
        vertical1 = np.sqrt((p2[0] - p6[0])**2 + (p2[1] - p6[1])**2)
        vertical2 = np.sqrt((p3[0] - p5[0])**2 + (p3[1] - p5[1])**2)

        # Calculate horizontal distance (normalization factor)
        horizontal = np.sqrt((p1[0] - p4[0])**2 + (p1[1] - p4[1])**2)

        # Calculate EAR
        # Avoid division by zero
        if horizontal == 0:
            return 0.0

        ear = (vertical1 + vertical2) / (2.0 * horizontal)

        return ear

    def get_average_ear(self, left_ear: float, right_ear: float) -> float:
        """
        Get the average EAR of both eyes.

        Args:
            left_ear: EAR value for left eye
            right_ear: EAR value for right eye

        Returns:
            Average EAR value
        """
        return (left_ear + right_ear) / 2.0

    def detect_blink(self, ear: float) -> bool:
        """
        Determine if a blink is occurring based on EAR value.

        Args:
            ear: Current EAR value

        Returns:
            True if blink is detected, False otherwise
        """
        if ear < EAR_THRESHOLD:
            self.blink_frame_counter += 1

            # Confirm blink after consecutive frames
            if self.blink_frame_counter >= BLINK_CONSECUTIVE_FRAMES:
                self.total_blinks += 1
                self.blink_frame_counter = 0
                self.last_blink_time = cv2.getTickCount()
                return True
        else:
            self.blink_frame_counter = 0

        return False

    def update_ear_history(self, ear: float):
        """
        Update the EAR history for drowsiness detection.

        Args:
            ear: Current EAR value
        """
        self.ear_history.append(ear)

    def is_drowsy(self) -> bool:
        """
        Check if the driver is drowsy based on prolonged eye closure.

        Returns:
            True if drowsy (eyes closed for extended period), False otherwise
        """
        if len(self.ear_history) < DROWSINESS_THRESHOLD_FRAMES:
            return False

        # If EAR has been below threshold for most of the recent frames
        closed_frames = sum(1 for ear in self.ear_history if ear < EAR_THRESHOLD)
        return closed_frames > (DROWSINESS_THRESHOLD_FRAMES * 0.7)

    def get_blink_count(self) -> int:
        """
        Get the total blink count.

        Returns:
            Total number of blinks detected
        """
        return self.total_blinks

    def reset(self):
        """Reset the blink counter and related state."""
        self.blink_frame_counter = 0
        self.total_blinks = 0
        self.ear_history.clear()
        self.last_blink_time = 0