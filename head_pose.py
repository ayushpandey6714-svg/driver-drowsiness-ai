# head_pose.py
# Head pose estimation using solvePnP

"""
Head Pose Estimation using Perspective-n-Point (PnP) algorithm:
---------------------------------------------------------------

The head pose is estimated by:
1. Using a 3D model of a generic human face (6 key points)
2. Detecting corresponding 2D image points from facial landmarks
3. Using OpenCV's solvePnP to find rotation and translation vectors
4. Converting rotation vector to Euler angles (yaw, pitch, roll)

3D Model Points (in millimeters, standard face model):
  Nose tip:         (0.0, 0.0, 0.0)
  Chin:             (0.0, -330.0, -65.0)
  Left eye corner:  (-225.0, 170.0, -135.0)
  Right eye corner: (225.0, 170.0, -135.0)
  Left mouth corner: (-150.0, -150.0, -125.0)
  Right mouth corner: (150.0, -150.0, -125.0)

2D Image Points (from MediaPipe Face Mesh landmarks):
  Nose tip:         landmark 1
  Chin:             landmark 152
  Left eye corner:  landmark 33
  Right eye corner: landmark 263
  Left mouth corner: landmark 61
  Right mouth corner: landmark 146

Euler Angles:
  - Yaw (rotation around Y-axis): Left/Right head turn
  - Pitch (rotation around X-axis): Up/Down head nod
  - Roll (rotation around Z-axis): Head tilt
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional

# 3D model points for generic human face (in mm)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye left corner
    (225.0, 170.0, -135.0),    # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)    # Right mouth corner
], dtype=np.float32)

# Landmark indices for the corresponding 2D image points
LANDMARK_INDICES = [
    1,    # Nose tip
    152,  # Chin
    33,   # Left eye left corner
    263,  # Right eye right corner
    61,   # Left mouth corner
    146   # Right mouth corner
]

# Thresholds for gaze direction (in degrees)
YAW_THRESHOLD = 20.0      # Left/Right
PITCH_THRESHOLD = 15.0    # Up/Down
ROLL_THRESHOLD = 15.0     # Tilt

def get_camera_matrix(frame_shape: Tuple[int, int]) -> np.ndarray:
    """
    Estimate camera matrix from frame dimensions.

    Args:
        frame_shape: (height, width) of the frame

    Returns:
        3x3 Camera matrix (intrinsic parameters)
    """
    h, w = frame_shape[:2]

    # Assume focal length is equal to frame width (common approximation)
    focal_length = w

    # Camera center is at image center
    center = (w / 2, h / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    return camera_matrix

def extract_2d_points(landmarks: list, frame_shape: Tuple[int, int]) -> np.ndarray:
    """
    Extract 2D image points from facial landmarks.

    Args:
        landmarks: List of normalized (x, y) landmarks
        frame_shape: (height, width) of the frame

    Returns:
        Array of 2D points (Nx2) in pixel coordinates
    """
    h, w = frame_shape[:2]
    image_points = []

    for idx in LANDMARK_INDICES:
        if idx < len(landmarks):
            lm = landmarks[idx]
            x = lm.x * w
            y = lm.y * h
            image_points.append([x, y])

    return np.array(image_points, dtype=np.float32)

def estimate_head_pose(landmarks: list, frame_shape: Tuple[int, int]) -> Tuple[float, float, float, str]:
    """
    Estimate head pose (yaw, pitch, roll) from facial landmarks.

    Args:
        landmarks: List of MediaPipe normalized landmarks
        frame_shape: (height, width) of the frame

    Returns:
        Tuple of (yaw, pitch, roll, gaze_direction) in degrees and string
    """
    # Get 2D image points
    image_points = extract_2d_points(landmarks, frame_shape)

    if len(image_points) < 6:
        return 0.0, 0.0, 0.0, "Center"

    # Get camera matrix
    camera_matrix = get_camera_matrix(frame_shape)

    # Assume zero distortion coefficients
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0, "Center"

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Convert rotation matrix to Euler angles (yaw, pitch, roll)
    # Using the standard XYZ convention
    # Note: OpenCV returns rotation vector in radians

    # Extract Euler angles from rotation matrix
    # This is a simplified extraction for Tait-Bryan angles (XYZ)
    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = 0

    # Convert from radians to degrees
    yaw = np.degrees(yaw)
    pitch = np.degrees(pitch)
    roll = np.degrees(roll)

    # Determine gaze direction
    gaze_direction = get_gaze_direction(yaw, pitch, roll)

    return yaw, pitch, roll, gaze_direction

def get_gaze_direction(yaw: float, pitch: float, roll: float) -> str:
    """
    Classify gaze direction based on head pose angles.

    Args:
        yaw: Yaw angle in degrees (left/right)
        pitch: Pitch angle in degrees (up/down)
        roll: Roll angle in degrees (tilt)

    Returns:
        String description of gaze direction
    """
    directions = []

    # Yaw: positive = right, negative = left
    if yaw > YAW_THRESHOLD:
        directions.append("Right")
    elif yaw < -YAW_THRESHOLD:
        directions.append("Left")

    # Pitch: positive = down, negative = up
    if pitch > PITCH_THRESHOLD:
        directions.append("Down")
    elif pitch < -PITCH_THRESHOLD:
        directions.append("Up")

    # Roll: positive = right tilt, negative = left tilt
    if roll > ROLL_THRESHOLD:
        directions.append("Tilt Right")
    elif roll < -ROLL_THRESHOLD:
        directions.append("Tilt Left")

    if not directions:
        return "Center"

    return " / ".join(directions)

def is_looking_away(yaw: float, pitch: float, roll: float) -> bool:
    """
    Determine if the driver is looking away from the road.

    Args:
        yaw: Yaw angle in degrees
        pitch: Pitch angle in degrees
        roll: Roll angle in degrees

    Returns:
        True if looking away, False if looking at road
    """
    # Looking away if any angle exceeds threshold
    if abs(yaw) > YAW_THRESHOLD:
        return True
    if abs(pitch) > PITCH_THRESHOLD:
        return True
    if abs(roll) > ROLL_THRESHOLD:
        return True

    return False

def draw_head_pose_axes(frame: np.ndarray, landmarks: list, frame_shape: Tuple[int, int],
                        yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Draw 3D coordinate axes on the face to visualize head pose.

    Args:
        frame: Image frame to draw on
        landmarks: Facial landmarks
        frame_shape: (height, width) of the frame
        yaw, pitch, roll: Head pose angles in degrees

    Returns:
        Frame with axes drawn
    """
    # Project 3D axis points to 2D image
    axis_length = 50  # pixels
    axis_3d = np.float32([
        [0, 0, 0],      # Origin (nose tip)
        [axis_length, 0, 0],  # X axis (red)
        [0, axis_length, 0],  # Y axis (green)
        [0, 0, -axis_length]  # Z axis (blue) - negative for forward
    ])

    # Get nose tip position in 2D
    nose_2d = None
    if len(landmarks) > 1:
        nose = landmarks[1]
        nose_2d = (int(nose.x * frame_shape[1]), int(nose.y * frame_shape[0]))

    if nose_2d is None:
        return frame

    # Get rotation vector from angles
    rvec = np.array([np.radians(pitch), np.radians(yaw), np.radians(roll)], dtype=np.float32)

    camera_matrix = get_camera_matrix(frame_shape)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # Project 3D axes to 2D
    axis_2d, _ = cv2.projectPoints(axis_3d, rvec, np.zeros(3), camera_matrix, dist_coeffs)

    # Draw axes
    axis_2d = axis_2d.reshape(-1, 2).astype(int)
    origin = axis_2d[0]

    # X axis (Red) - Right
    cv2.line(frame, tuple(origin), tuple(axis_2d[1]), (0, 0, 255), 3)
    # Y axis (Green) - Down
    cv2.line(frame, tuple(origin), tuple(axis_2d[2]), (0, 255, 0), 3)
    # Z axis (Blue) - Forward
    cv2.line(frame, tuple(origin), tuple(axis_2d[3]), (255, 0, 0), 3)

    return frame