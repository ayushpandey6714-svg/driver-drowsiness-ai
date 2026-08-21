# yawn.py
# Yawn detection and Mouth Aspect Ratio (MAR) calculation

"""
Mathematical Formula - Mouth Aspect Ratio (MAR):
-----------------------------------------------
MAR = (||p2-p10|| + ||p4-p8|| + ||p6-p0||) / (2 * ||p1-p9||)

Where:
  p0, p9 = Left and right corners of the mouth
  p2, p10 = Upper-left and upper-right side of mouth
  p4, p8 = Lower-left and lower-right side of mouth
  p6 = Middle point on upper lip
  p1-p9 = Corner points controlling the width of mouth

Landmark indices for Mouth (MediaPipe Face Mesh):
  - Mouth corner left (p0): landmark 61
  - Mouth corner right (p9): landmark 146
  - Upper-left mouth (p2): landmark 65
  - Upper-right mouth (p10): landmark 189
  - Lower-left mouth (p4): landmark 13
  - Lower-right mouth (p8): landmark 14
  - Middle upper lip (p6): landmark 123

When mouth is closed, MAR approaches 0. When mouth is open (especially yawning),
MAR increases significantly. A typical threshold for yawning is around 0.5.

Threshold behavior:
  - Opening mouth widely: MAR > 0.35
  - Yawning threshold: MAR > 0.5 for sustained period
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from typing import List, Tuple

# MAR threshold for detecting yawn
YAWN_MAR_THRESHOLD = 0.5
# Consecutive frames mouth must stay open for during yawning
YAWN_DURATION_FRAMES = 15
# MAR threshold for detecting closed mouth
MOUTH_CLOSED_THRESHOLD = 0.2

# Landmark indices for mouth region (closest approximations)
MOUTH_LANDMARKS = [
    61,   # Left corner
    146,  # Right corner
    65,   # Upper left
    189,  # Upper right
    13,   # Lower left
    14,   # Lower right
    123   # Upper lip middle
]

# Mouth detection state
yawn_frame_counter = 0
mouth_open_counter = 0
total_yawns = 0


class YawnDetector:
    """
    Detects yawning using Mouth Aspect Ratio (MAR) calculation.
    """

    def __init__(self):
        """Initialize yawn detector with counters and history."""
        self.yawn_frame_counter = 0
        self.mouth_open_counter = 0
        self.total_yawns = 0

    def calculate_mar(self, landmarks: List[Tuple[float, float]]) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR) using key mouth landmarks.

        Formula:
            MAR = (||p2-p10|| + ||p4-p8|| + ||p6-p0||) / (2 * ||p1-p9||)

        Args:
            landmarks: List of (x, y) tuples for all face landmarks (normalized [0,1])

        Returns:
            MAR value (float between 0 and 1)
        """
        try:
            # Extract key mouth landmarks
            # p1, p9 = left and right mouth corners
            # p2, p10 = upper left and upper right
            # p4, p8 = lower left and lower right
            # p6 = upper middle lip
            # p0 = additional reference point

            # Get key points using standard indices
            left_corner = landmarks[61]     # Left mouth corner (p1)
            right_corner = landmarks[146]   # Right mouth corner (p9)
            upper_left = landmarks[65]      # Upper lip left (p2)
            upper_right = landmarks[189]    # Upper lip right (p10)
            lower_left = landmarks[13]      # Lower lip left (p4)
            lower_right = landmarks[14]     # Lower lip right (p8)
            upper_middle = landmarks[123]   # Upper lip middle (p6)

            # Calculate horizontal width (distance between mouth corners)
            mouth_width = np.sqrt(
                (left_corner[0] - right_corner[0])**2 +
                (left_corner[1] - right_corner[1])**2
            )

            # Calculate vertical heights
            # Height 1: upper middle to lower middle (approximate)
            # Using upper_middle to average of lower left/right
            lower_middle_y = (lower_left[1] + lower_right[1]) / 2.0
            lower_middle_x = (lower_left[0] + lower_right[0]) / 2.0
            upper_middle_point = upper_middle

            height1 = np.sqrt(
                (upper_middle_point[0] - lower_middle_x)**2 +
                (upper_middle_point[1] - lower_middle_y)**2
            )

            # Height 2: upper left to lower left
            height2 = np.sqrt(
                (upper_left[0] - lower_left[0])**2 +
                (upper_left[1] - lower_left[1])**2
            )

            # Height 3: upper right to lower right
            height3 = np.sqrt(
                (upper_right[0] - lower_right[0])**2 +
                (upper_right[1] - lower_right[1])**2
            )

            # Calculate MAR: average of three heights divided by width
            if mouth_width == 0:
                return 0.0

            mar = (height1 + height2 + height3) / (3.0 * mouth_width)
            return mar

        except Exception:
            return 0.0

    def detect_yawn(self, mar: float) -> bool:
        """
        Determine if a yawn is occurring based on MAR value.

        Args:
            mar: Current MAR value

        Returns:
            True if yawn is detected, False otherwise
        """
        if mar > YAWN_MAR_THRESHOLD:
            self.mouth_open_counter += 1

            # Check if mouth has been open long enough to register as yawn
            if self.mouth_open_counter >= YAWN_DURATION_FRAMES:
                self.total_yawns += 1
                self.mouth_open_counter = 0
                return True
        else:
            self.mouth_open_counter = 0

        return False

    def get_yawn_count(self) -> int:
        """
        Get the total yawn count.

        Returns:
            Total number of yawns detected
        """
        return self.total_yawns

    def reset(self):
        """Reset the yawn counter and related state."""
        self.yawn_frame_counter = 0
        self.mouth_open_counter = 0
        self.total_yawns = 0