# report.py
# Session report generation and CSV export

"""
Session Report Features:
- Start/End session tracking
- Frame-by-frame data logging
- CSV export for analysis
- Summary statistics generation
"""

import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import time

class SessionReport:
    """
    Class to handle session data collection and report generation.
    """

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.frame_data: List[Dict] = []
        self.is_active: bool = False

    def start(self) -> str:
        """
        Start a new session.

        Returns:
            Session ID string
        """
        self.start_time = time.time()
        self.end_time = None
        self.frame_data = []
        self.is_active = True

        # Generate session ID based on timestamp
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        return session_id

    def end(self) -> Dict:
        """
        End the current session and return summary statistics.

        Returns:
            Dictionary with session summary statistics
        """
        self.end_time = time.time()
        self.is_active = False

        if not self.frame_data:
            return {}

        df = pd.DataFrame(self.frame_data)

        summary = {
            'total_frames': len(df),
            'session_duration': self.get_duration(),
            'avg_fps': 0,
            'total_blinks': int(df['blink_count'].iloc[-1]) if 'blink_count' in df.columns else 0,
            'total_yawns': int(df['yawn_count'].iloc[-1]) if 'yawn_count' in df.columns else 0,
            'final_fatigue_score': df['fatigue_score'].iloc[-1] if 'fatigue_score' in df.columns else 0,
            'avg_fatigue_score': df['fatigue_score'].mean() if 'fatigue_score' in df.columns else 0,
            'max_fatigue_score': df['fatigue_score'].max() if 'fatigue_score' in df.columns else 0,
            'safe_time': len(df[df['driver_status'] == 'Safe']) / len(df) * 100 if 'driver_status' in df.columns else 0,
            'warning_time': len(df[df['driver_status'] == 'Warning']) / len(df) * 100 if 'driver_status' in df.columns else 0,
            'drowsy_time': len(df[df['driver_status'] == 'Drowsy']) / len(df) * 100 if 'driver_status' in df.columns else 0,
            'data_columns': list(df.columns)
        }

        if len(df) > 0:
            summary['avg_fps'] = len(df) / summary['session_duration'] if summary['session_duration'] > 0 else 0

        return summary

    def log_frame(self, frame_timestamp: float, metrics: Dict):
        """
        Log data for a single frame.

        Args:
            frame_timestamp: Timestamp of the frame
            metrics: Dictionary containing frame metrics
        """
        self.frame_data.append({
            'timestamp': frame_timestamp,
            'ear': metrics.get('ear', 0.0),
            'mar': metrics.get('mar', 0.0),
            'yaw': metrics.get('yaw', 0.0),
            'pitch': metrics.get('pitch', 0.0),
            'roll': metrics.get('roll', 0.0),
            'blink_count': metrics.get('blink_count', 0),
            'yawn_count': metrics.get('yawn_count', 0),
            'fatigue_score': metrics.get('fatigue_score', 0.0),
            'driver_status': metrics.get('driver_status', 'Unknown'),
            'fps': metrics.get('fps', 0.0),
            'gaze_direction': metrics.get('gaze_direction', 'Center'),
            'is_blinking': metrics.get('is_blinking', False),
            'is_yawning': metrics.get('is_yawning', False),
            'is_looking_away': metrics.get('is_looking_away', False)
        })

    def get_duration(self) -> float:
        """
        Get session duration in seconds.

        Returns:
            Duration in seconds
        """
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def generate_csv(self, filepath: str) -> bool:
        """
        Generate CSV file from collected frame data.

        Args:
            filepath: Path to save the CSV file

        Returns:
            True if successful, False otherwise
        """
        if not self.frame_data:
            return False

        try:
            df = pd.DataFrame(self.frame_data)
            df.to_csv(filepath, index=False)
            return True
        except Exception as e:
            print(f"Error generating CSV: {e}")
            return False

    def generate_summary_text(self) -> str:
        """
        Generate a text summary of the session.

        Returns:
            String containing session summary
        """
        if not self.is_active and not self.frame_data:
            return "No session data available."

        summary = "\n" + "=" * 50 + "\n"
        summary += "DRIVER SAFETY SESSION REPORT\n"
        summary += "=" * 50 + "\n\n"

        duration = self.get_duration()
        summary += f"Session Duration: {duration:.1f} seconds\n"

        if self.frame_data:
            df = pd.DataFrame(self.frame_data)

            summary += f"Total Frames: {len(df)}\n"
            summary += f"Average FPS: {duration/len(df):.2f}\n\n"

            summary += "-" * 50 + "\n"
            summary += "EVENT COUNTS:\n"
            summary += "-" * 50 + "\n"
            summary += f"Total Blinks: {df['blink_count'].iloc[-1] if 'blink_count' in df.columns else 0}\n"
            summary += f"Total Yawns: {df['yawn_count'].iloc[-1] if 'yawn_count' in df.columns else 0}\n"
            if duration > 0:
                summary += f"Blinks/Minute: {df['blink_count'].iloc[-1] * 60 / duration:.1f}\n"
                summary += f"Yawns/Minute: {df['yawn_count'].iloc[-1] * 60 / duration:.1f}\n"

            summary += "\n" + "-" * 50 + "\n"
            summary += "FATIGUE ANALYSIS:\n"
            summary += "-" * 50 + "\n"
            summary += f"Final Fatigue Score: {df['fatigue_score'].iloc[-1]:.3f}\n"
            summary += f"Average Fatigue Score: {df['fatigue_score'].mean():.3f}\n"
            summary += f"Max Fatigue Score: {df['fatigue_score'].max():.3f}\n\n"

            if 'driver_status' in df.columns:
                safe_pct = len(df[df['driver_status'] == 'Safe']) / len(df) * 100
                warning_pct = len(df[df['driver_status'] == 'Warning']) / len(df) * 100
                drowsy_pct = len(df[df['driver_status'] == 'Drowsy']) / len(df) * 100

                summary += "Driver Status Distribution:\n"
                summary += f"  Safe:   {safe_pct:.1f}%\n"
                summary += f"  Warning: {warning_pct:.1f}%\n"
                summary += f"  Drowsy: {drowsy_pct:.1f}%\n"

        summary += "\n" + "=" * 50 + "\n"
        summary += f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        return summary


# Global session instance
session = SessionReport()

def start_session() -> str:
    """
    Start a new session.

    Returns:
        Session ID string
    """
    return session.start()

def end_session() -> Dict:
    """
    End the current session.

    Returns:
        Session summary statistics
    """
    return session.end()

def log_frame_data(timestamp: float, metrics: Dict):
    """
    Log frame data to the current session.

    Args:
        timestamp: Frame timestamp
        metrics: Dictionary of frame metrics
    """
    session.log_frame(timestamp, metrics)

def generate_csv_report(filepath: str) -> bool:
    """
    Generate CSV report file.

    Args:
        filepath: Path to save CSV

    Returns:
        True if successful
    """
    return session.generate_csv(filepath)

def generate_summary_report() -> str:
    """
    Generate text summary report.

    Returns:
        String containing the report
    """
    return session.generate_summary_text()

def save_report(session_id: str) -> tuple:
    """
    Save reports for a session.

    Args:
        session_id: Unique session identifier

    Returns:
        Tuple of (csv_path, summary_path) or (None, None) on failure
    """
    # Ensure reports directory exists
    reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    # Generate CSV
    csv_path = os.path.join(reports_dir, f"session_{session_id}.csv")
    csv_success = session.generate_csv(csv_path)

    # Generate summary
    summary_path = os.path.join(reports_dir, f"session_{session_id}_summary.txt")
    try:
        summary_text = session.generate_summary_text()
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        summary_success = True
    except:
        summary_success = False

    if csv_success and summary_success:
        return csv_path, summary_path
    return None, None

def create_report_data() -> Dict:
    """
    Create default report data structure.

    Returns:
        Dictionary with default values
    """
    return {
        'timestamp': time.time(),
        'ear': 0.0,
        'mar': 0.0,
        'yaw': 0.0,
        'pitch': 0.0,
        'roll': 0.0,
        'blink_count': 0,
        'yawn_count': 0,
        'fatigue_score': 0.0,
        'driver_status': 'Unknown',
        'fps': 0.0,
        'gaze_direction': 'Center'
    }